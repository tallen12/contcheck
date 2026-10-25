"""Readers for raw clean text: plain-text files and JSONL files of paragraphs."""

import re
from collections.abc import Iterator
from pathlib import Path

from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.jsonl import read_jsonl

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
WHITESPACE = re.compile(r"\s+")
SUFFIXES = (".txt", ".jsonl")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces, so tokens rebuild the text."""
    return WHITESPACE.sub(" ", text).strip()


def read_text_file(path: Path) -> Iterator[RawParagraph]:
    """Paragraphs of a plain-text file, separated by blank lines; the file stem is the source."""
    text = path.read_text(encoding="utf-8")
    for block in PARAGRAPH_BREAK.split(text):
        paragraph = normalize_whitespace(block)
        if paragraph:
            yield RawParagraph(source_id=path.stem, text=paragraph)


def read_jsonl_file(path: Path) -> Iterator[RawParagraph]:
    """Paragraphs of a JSONL file with one ``RawParagraph`` per line."""
    for paragraph in read_jsonl(path, RawParagraph):
        text = normalize_whitespace(paragraph.text)
        if text:
            yield RawParagraph(source_id=paragraph.source_id, text=text)


def read_paragraphs(path: Path) -> Iterator[RawParagraph]:
    """Paragraphs of one ``.txt`` or ``.jsonl`` file, or of every such file in a directory.

    Directory entries are read in name order so the output is deterministic.
    """
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in SUFFIXES)
    else:
        files = [path]
    for file in files:
        match file.suffix:
            case ".txt":
                yield from read_text_file(file)
            case ".jsonl":
                yield from read_jsonl_file(file)
            case _:
                raise ValueError(f"{file}: expected a .txt or .jsonl file")
