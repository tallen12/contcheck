import pytest

from contcheck.training.misspellings import load_misspellings, parse_misspellings


def test_parse_maps_correct_words_to_misspellings() -> None:
    parsed = parse_misspellings(["# comment", "", "Tomorrow: tomorow tommorrow  # note"])

    assert parsed == {"tomorrow": ("tomorow", "tommorrow")}


@pytest.mark.parametrize("line", ["no colon here", ": orphan", "word:"])
def test_parse_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_misspellings([line])


def test_bundled_list_loads() -> None:
    misspellings = load_misspellings()

    assert misspellings["occurred"] == ("occured",)
    assert len(misspellings) > 80
