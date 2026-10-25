from pathlib import Path

import pytest

from contcheck.cli.tag import tag


def test_tag_runs_spacy_and_the_checkpoint(
    checkpoint: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Random weights flag nothing above a threshold of 1, but the full path runs.
    tag("Yesterday I go to the store.", checkpoint=checkpoint, threshold=1.0)

    assert capsys.readouterr().out.strip() == "no errors flagged"
