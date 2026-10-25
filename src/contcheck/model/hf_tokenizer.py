"""``Tokenizer`` over a Hugging Face fast tokenizer, such as ELECTRA's WordPiece."""

from collections.abc import Sequence
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from contcheck.model.models import Encoding


class HuggingFaceTokenizer:
    """Encodes pre-split words with ``is_split_into_words``, keeping word ids."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if not tokenizer.is_fast:
            raise ValueError("word ids need a fast tokenizer")
        self._tokenizer = tokenizer
        self.cls_id = _token_id(tokenizer.cls_token_id, "CLS")
        self.sep_id = _token_id(tokenizer.sep_token_id, "SEP")
        self.pad_id = _token_id(tokenizer.pad_token_id, "PAD")

    def encode_words(self, words: Sequence[str]) -> Encoding:
        encoded = self._tokenizer(list(words), is_split_into_words=True, add_special_tokens=False)
        input_ids: list[int] = []
        word_ids: list[int] = []
        for token_id, word_id in zip(encoded["input_ids"], encoded.word_ids(), strict=True):
            if word_id is None:
                raise ValueError("unexpected special token in a word encoding")
            input_ids.append(int(token_id))
            word_ids.append(word_id)
        return Encoding(tuple(input_ids), tuple(word_ids))

    def save(self, directory: Path) -> None:
        """Save the tokenizer files so a checkpoint loads offline."""
        self._tokenizer.save_pretrained(directory)


def load_tokenizer(name_or_path: str | Path) -> HuggingFaceTokenizer:
    """Load a tokenizer by Hub name (downloaded once, then cached) or from a directory."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer is None:
        raise ValueError(f"no tokenizer found for {name_or_path}")
    return HuggingFaceTokenizer(tokenizer)


def _token_id(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"tokenizer has no {name} token")
    return value
