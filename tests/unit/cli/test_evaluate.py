from pathlib import Path

import pytest

from contcheck.cli.evaluate import evaluate
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.training.models import EvalResult
from tests.conftest import make_record


def test_evaluate_prints_and_logs_model_and_baseline(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "val.jsonl"
    write_jsonl(data, [make_record(i, noisy=i % 2 == 0) for i in range(4)])
    results = tmp_path / "results.jsonl"

    evaluate(checkpoint=checkpoint, data=data, results=results, device="cpu")

    output = capsys.readouterr().out
    assert f"{checkpoint} on {data}" in output
    assert "symspell-baseline" in output
    logged = list(read_jsonl(results, EvalResult))
    assert [r.name for r in logged] == [str(checkpoint), "symspell-baseline"]


def test_evaluate_without_baseline(checkpoint: Path, tmp_path: Path) -> None:
    data = tmp_path / "val.jsonl"
    write_jsonl(data, [make_record(0, noisy=True)])
    results = tmp_path / "results.jsonl"

    evaluate(checkpoint=checkpoint, data=data, results=results, device="cpu", baseline=False)

    assert len(results.read_text(encoding="utf-8").splitlines()) == 1
