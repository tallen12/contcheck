from pathlib import Path

import pytest

from contcheck.cli.train import train


def test_document_vectors_are_refused_until_m4(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("use_doc: true\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="M4"):
        train(config=config, device="cpu")
