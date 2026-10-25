from pathlib import Path

import pytest
from transformers import BertTokenizer

from contcheck.model.hf_tokenizer import HuggingFaceTokenizer, load_tokenizer

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "went", "go", "##ne", "home", "."]


@pytest.fixture
def bert() -> BertTokenizer:
    return BertTokenizer(vocab={token: i for i, token in enumerate(VOCAB)})


def test_encodes_words_with_word_ids(bert: BertTokenizer) -> None:
    tokenizer = HuggingFaceTokenizer(bert)

    encoding = tokenizer.encode_words(["I", "gone", "home", "."])

    assert encoding.input_ids == (5, 7, 8, 9, 10)
    assert encoding.word_ids == (0, 1, 1, 2, 3)
    assert (tokenizer.cls_id, tokenizer.sep_id, tokenizer.pad_id) == (2, 3, 0)


def test_saved_tokenizer_loads_from_its_directory(bert: BertTokenizer, tmp_path: Path) -> None:
    HuggingFaceTokenizer(bert).save(tmp_path / "saved")

    loaded = load_tokenizer(tmp_path / "saved")

    assert loaded.encode_words(["went"]).input_ids == (6,)
