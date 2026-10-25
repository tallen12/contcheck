import pytest

from contcheck.model.hf_tokenizer import load_tokenizer


@pytest.mark.models
def test_electra_tokenizer_splits_unknown_words() -> None:
    tokenizer = load_tokenizer("google/electra-small-discriminator")

    encoding = tokenizer.encode_words(["Yesterday", "I", "buyed", "apples"])

    assert len(encoding.input_ids) > 4
    assert encoding.word_ids[0] == 0
    assert encoding.word_ids[-1] == 3
