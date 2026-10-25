from pathlib import Path

import pytest

from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.readers import normalize_whitespace, read_paragraphs


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a\n b\t\tc ") == "a b c"


def test_text_files_split_on_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("First line\nstill first.\n\n  \nSecond.\n", encoding="utf-8")

    assert list(read_paragraphs(path)) == [
        RawParagraph(source_id="story", text="First line still first."),
        RawParagraph(source_id="story", text="Second."),
    ]


def test_directory_reads_txt_and_jsonl_in_name_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("From b.", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"source_id":"x","text":"From  a."}\n', encoding="utf-8")
    (tmp_path / "ignored.md").write_text("No.", encoding="utf-8")

    texts = [p.text for p in read_paragraphs(tmp_path)]

    assert texts == ["From a.", "From b."]


def test_unknown_file_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        list(read_paragraphs(path))
