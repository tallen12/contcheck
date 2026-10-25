from pathlib import Path

import pytest

from contcheck.cli.corrupt import corrupt
from contcheck.corpus.jsonl import read_jsonl
from contcheck.corpus.record import Record
from contcheck.corpus.report import build_report

STORY = """\
The old man walked to the harbour every morning. He watched the boats and counted the gulls.

One day a stranger arrived and asked him about the sea. They talked for hours, and the
stranger left before sunset.

The next morning the man returned to the harbour. The boats were gone, and the gulls were quiet.
"""
CONFIG = """\
seed: 5
variants_per_paragraph: 3
errors_per_paragraph: {1: 0.5, 2: 0.5}
kind_weights: {default: 1}
"""


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "story.txt").write_text(STORY, encoding="utf-8")
    return directory


def test_corrupt_writes_valid_round_tripping_records(raw: Path, tmp_path: Path) -> None:
    config = tmp_path / "gen.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    corrupt(config=config, in_=raw, out=first)
    corrupt(config=config, in_=raw, out=second)

    records = list(read_jsonl(first, Record))
    report = build_report(records)
    assert len(records) == 9
    assert report.round_trip_failures == 0
    assert sum(report.kinds.values()) > 0
    assert first.read_bytes() == second.read_bytes()
