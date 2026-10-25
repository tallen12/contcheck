"""``MaskedScorer`` over a Hugging Face masked LM with a byte-level BPE tokenizer.

Built for RoBERTa-style models such as ``distilroberta-base``, whose tokens carry a
leading-space marker (``Ġ``): a word after a space is one ``Ġword`` token, a word at the
start of the text or glued to the token before it is a bare ``word`` token. Only words
that are a single token are predicted or scored.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from contcheck.analysis.models import join_words
from contcheck.lm.models import MaskedSlot, SlotPrediction, WordScore

DEFAULT_MODEL = "distilroberta-base"
SPACE_MARKER = "Ġ"
BATCH_SIZE = 64
MAX_LENGTH = 256


@dataclass(frozen=True, slots=True)
class _Encoded:
    input_ids: list[int]
    mask_index: int
    glued: bool
    candidate_ids: tuple[tuple[str, int], ...]


class HuggingFaceMaskedScorer:
    """Masks one word per slot and reads the model's distribution over its vocabulary."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self._device = torch.device(device)
        self._model = model.eval()
        # PreTrainedModel.to is wrapped in a way ty cannot follow; Module.to is the same call.
        torch.nn.Module.to(self._model, self._device)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_length = max_length
        self._cls_id = _token_id(tokenizer.cls_token_id, "CLS")
        self._sep_id = _token_id(tokenizer.sep_token_id, "SEP")
        self._pad_id = _token_id(tokenizer.pad_token_id, "PAD")
        self._mask_id = _token_id(tokenizer.mask_token_id, "MASK")
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
        spaced, glued = _word_tables(tokenizer, vocab_size)
        self._words = {False: spaced, True: glued}
        self._word_masks = {
            key: _id_mask(table, vocab_size, self._device) for key, table in self._words.items()
        }

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        encoded = [self._encode(slot) for slot in slots]
        predictions: list[SlotPrediction] = []
        for start in range(0, len(encoded), self._batch_size):
            predictions.extend(
                self._predict_batch(encoded[start : start + self._batch_size], top_k)
            )
        return predictions

    def _encode(self, slot: MaskedSlot) -> _Encoded:
        position = slot.position
        glued = position == 0 or not slot.spaces[position - 1]
        left = join_words(slot.words[:position], slot.spaces[:position]).rstrip(" ")
        right = join_words(slot.words[position + 1 :], slot.spaces[position + 1 :])
        if slot.spaces[position] and right:
            right = " " + right
        left_ids = self._ids(left)
        right_ids = self._ids(right)
        # Keep the mask near the middle when the text is longer than the model takes.
        room = self._max_length - 3
        keep_left = min(len(left_ids), max(room // 2, room - len(right_ids)))
        keep_right = min(len(right_ids), room - keep_left)
        input_ids = [
            self._cls_id,
            *left_ids[len(left_ids) - keep_left :],
            self._mask_id,
            *right_ids[:keep_right],
            self._sep_id,
        ]
        prefix = "" if glued else " "
        candidate_ids = tuple(
            (word, ids[0]) for word in slot.candidates if len(ids := self._ids(prefix + word)) == 1
        )
        return _Encoded(input_ids, keep_left + 1, glued, candidate_ids)

    def _ids(self, text: str) -> list[int]:
        if not text:
            return []
        ids: object = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if not isinstance(ids, list):
            raise TypeError("tokenizer returned no id list")
        return [int(i) for i in ids]

    def _predict_batch(self, batch: Sequence[_Encoded], top_k: int) -> list[SlotPrediction]:
        width = max(len(item.input_ids) for item in batch)
        input_ids = torch.full((len(batch), width), self._pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            input_ids[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
            attention[row, : len(item.input_ids)] = 1
        rows = torch.arange(len(batch))
        mask_indices = torch.tensor([item.mask_index for item in batch])
        with torch.inference_mode():
            logits = self._model(
                input_ids=input_ids.to(self._device), attention_mask=attention.to(self._device)
            ).logits
            probabilities = logits[rows.to(self._device), mask_indices.to(self._device)]
            probabilities = probabilities.float().softmax(dim=-1)
        return [
            self._read(item, row, top_k) for item, row in zip(batch, probabilities, strict=True)
        ]

    def _read(self, item: _Encoded, probabilities: torch.Tensor, top_k: int) -> SlotPrediction:
        table = self._words[item.glued]
        top: tuple[WordScore, ...] = ()
        if top_k := min(top_k, len(table)):
            # Probabilities are never negative, so -1 keeps sub-word pieces out of the top.
            words_only = probabilities.masked_fill(~self._word_masks[item.glued], -1.0)
            values, indices = words_only.topk(top_k)
            top = tuple(
                WordScore(table[int(index)], float(value))
                for value, index in zip(values.tolist(), indices.tolist(), strict=True)
                if value >= 0
            )
        scored = tuple(
            WordScore(word, float(probabilities[token_id])) for word, token_id in item.candidate_ids
        )
        return SlotPrediction(top, scored)


def _word_tables(tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> tuple[dict[int, str], ...]:
    """Ids of whole ASCII words after a space, and of bare words, mapped to the word."""
    spaced: dict[int, str] = {}
    glued: dict[int, str] = {}
    tokens: object = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    if not isinstance(tokens, list):
        raise TypeError("tokenizer returned no token list")
    for token_id, token in enumerate(tokens):
        if token.startswith(SPACE_MARKER):
            word = token.removeprefix(SPACE_MARKER)
            if word.isascii() and word.isalpha():
                spaced[token_id] = word
        elif token.isascii() and token.isalpha():
            glued[token_id] = token
    return spaced, glued


def _id_mask(table: dict[int, str], vocab_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[list(table)] = True
    return mask.to(device)


def _token_id(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"tokenizer has no {name} token")
    return value


def load_masked_scorer(name: str = DEFAULT_MODEL, *, device: str) -> HuggingFaceMaskedScorer:
    """Load a masked LM and its tokenizer by Hub name (downloaded once, then cached).

    On a GPU the weights are half precision, which about halves the time per batch;
    probabilities are still computed in float32.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer is None:
        raise ValueError(f"no tokenizer found for {name}")
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    model = AutoModelForMaskedLM.from_pretrained(name, dtype=dtype)
    return HuggingFaceMaskedScorer(model, tokenizer, device=device)
