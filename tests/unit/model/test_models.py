import pytest

from contcheck.model.models import IGNORE_INDEX, Encoding, Window


def test_encoding_needs_one_word_id_per_token() -> None:
    Encoding((5, 6), (0, 0))
    with pytest.raises(ValueError, match="word id"):
        Encoding((5, 6), (0,))


def test_window_columns_must_line_up() -> None:
    Window("d", (1, 5, 2), (None, 0, None), (IGNORE_INDEX, 3, IGNORE_INDEX), start=0)
    with pytest.raises(ValueError, match="one entry per token"):
        Window("d", (1, 5, 2), (None, 0, None), (IGNORE_INDEX,), start=0)
