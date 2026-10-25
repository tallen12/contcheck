import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM

from contcheck.lm.hf_masked_lm import HuggingFaceMaskedScorer
from contcheck.lm.models import MaskedSlot

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
CORPUS = [
    "The dog sleeps in the garden every morning.",
    "A cat walked to the park and the river.",
    "Birds sing while the old man reads a book.",
] * 20


@pytest.fixture(scope="module")
def scorer() -> HuggingFaceMaskedScorer:
    """A random two-layer RoBERTa over a byte-level BPE trained on a few sentences."""
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(CORPUS, vocab_size=600, min_frequency=1, special_tokens=SPECIAL)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    torch.manual_seed(0)
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFaceMaskedScorer(
        RobertaForMaskedLM(config), tokenizer, device="cpu", batch_size=2, max_length=24
    )


WORDS = ("The", "dog", "sleeps", "in", "the", "garden", ".")
SPACES = (True, True, True, True, True, False, False)


def test_top_words_are_whole_words_most_probable_first(scorer: HuggingFaceMaskedScorer) -> None:
    (prediction,) = scorer.predict([MaskedSlot(WORDS, SPACES, 5)], top_k=5)

    probabilities = [score.probability for score in prediction.top]
    assert len(prediction.top) == 5
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(score.word.isalpha() for score in prediction.top)
    assert prediction.scored == ()


@pytest.mark.parametrize("position", [0, 1, 5])
def test_scored_candidates_match_the_top(scorer: HuggingFaceMaskedScorer, position: int) -> None:
    (first,) = scorer.predict([MaskedSlot(WORDS, SPACES, position)], top_k=3)
    best = first.top[0]

    (second,) = scorer.predict([MaskedSlot(WORDS, SPACES, position, (best.word,))], top_k=0)

    assert second.top == ()
    assert second.probability(best.word) == pytest.approx(best.probability)


def test_words_that_are_not_one_token_are_not_scored(scorer: HuggingFaceMaskedScorer) -> None:
    (prediction,) = scorer.predict(
        [MaskedSlot(WORDS, SPACES, 1, ("dog", "xylophonequartz"))], top_k=0
    )

    assert [score.word for score in prediction.scored] == ["dog"]
    assert 0 < (prediction.probability("dog") or 0) < 1


def test_batches_and_padding_do_not_change_predictions(scorer: HuggingFaceMaskedScorer) -> None:
    slots = [
        MaskedSlot(WORDS, SPACES, 1, ("dog",)),
        MaskedSlot(("A", "cat", "walked", "."), (True, True, False, False), 2, ("walked",)),
        MaskedSlot(WORDS, SPACES, 5, ("garden",)),
    ]

    together = scorer.predict(slots, top_k=2)
    alone = [scorer.predict([slot], top_k=2)[0] for slot in slots]

    for joint, single in zip(together, alone, strict=True):
        assert [s.word for s in joint.top] == [s.word for s in single.top]
        assert joint.scored[0].probability == pytest.approx(single.scored[0].probability, abs=1e-5)


def test_long_text_keeps_the_mask(scorer: HuggingFaceMaskedScorer) -> None:
    words = ("the", "dog") * 30
    slot = MaskedSlot(words, (True,) * len(words), 41, ("dog",))

    (prediction,) = scorer.predict([slot], top_k=1)

    assert len(prediction.top) == 1
    assert prediction.probability("dog") is not None
