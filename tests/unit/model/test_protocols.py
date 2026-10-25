from contcheck.model.protocols import Tokenizer
from tests.conftest import FakeTokenizer


def test_tokenizer_contract(tokenizer: FakeTokenizer) -> None:
    implementation: Tokenizer = tokenizer

    encoding = implementation.encode_words(["Hello", "world", "!"])

    assert len({implementation.cls_id, implementation.sep_id, implementation.pad_id}) == 3
    assert encoding.word_ids == (0, 0, 1, 1, 2)
    assert implementation.encode_words(["hello"]).input_ids == encoding.input_ids[:2]
