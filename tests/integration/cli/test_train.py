import json
from pathlib import Path

import pytest

from contcheck.cli.train import train
from contcheck.corpus.jsonl import write_jsonl
from contcheck.model.checkpoint import load_checkpoint
from tests.conftest import make_record


@pytest.mark.models
def test_train_saves_a_loadable_checkpoint(tmp_path: Path) -> None:
    records = [make_record(i, noisy=i % 2 == 0) for i in range(8)]
    write_jsonl(tmp_path / "train.jsonl", records)
    write_jsonl(tmp_path / "val.jsonl", records)
    config = tmp_path / "train.yaml"
    config.write_text(
        f"epochs: 1\nbatch_size: 4\ntrain: {tmp_path / 'train.jsonl'}\n"
        f"val: {tmp_path / 'val.jsonl'}\nout: {tmp_path / 'run'}\n",
        encoding="utf-8",
    )

    train(config=config, device="cpu")

    loaded = load_checkpoint(tmp_path / "run")
    assert loaded.info.epoch == 1
    (line,) = (tmp_path / "run" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    report = json.loads(line)
    assert report["epoch"] == 1
    assert report["train_loss"] > 0
    assert report["val"]["name"] == "epoch-1"
