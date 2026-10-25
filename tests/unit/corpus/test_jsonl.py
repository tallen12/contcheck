from pathlib import Path

import pytest
from pydantic import BaseModel

from contcheck.corpus.jsonl import read_jsonl, write_jsonl


class Item(BaseModel):
    name: str


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "items.jsonl"

    count = write_jsonl(path, [Item(name="a"), Item(name="b")])

    assert count == 2
    assert path.read_text(encoding="utf-8") == '{"name":"a"}\n{"name":"b"}\n'
    assert list(read_jsonl(path, Item)) == [Item(name="a"), Item(name="b")]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text('{"name":"a"}\n\n', encoding="utf-8")

    assert list(read_jsonl(path, Item)) == [Item(name="a")]


def test_invalid_line_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text('{"name":"a"}\n{"nope":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"items\.jsonl:2"):
        list(read_jsonl(path, Item))
