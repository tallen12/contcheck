from pathlib import Path

import pytest

from contcheck.cli.evaluate import build_checks, evaluate
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.training.models import EvalResult
from contcheck.training.probes import CheckName
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


def test_build_checks_adds_held_out_pairs_only_when_some_are_withheld(tmp_path: Path) -> None:
    records = [make_record(i, noisy=i % 2 == 0) for i in range(4)]
    base = "seed: 13\nvariants_per_paragraph: 1\nerrors_per_paragraph: {1: 1}\n"
    base += "kind_weights: {default: 1}\n"
    none_held = tmp_path / "none.yaml"
    none_held.write_text(base, encoding="utf-8")
    some_held = tmp_path / "some.yaml"
    some_held.write_text(base + "held_out_confusion_share: 0.5\n", encoding="utf-8")

    assert set(build_checks(records, gen_config=none_held)) == {
        CheckName.REAL_MISSPELLINGS,
        CheckName.SEEN_CONFUSIONS,
    }
    checks = build_checks(records, gen_config=some_held)
    assert CheckName.HELD_OUT_CONFUSIONS in checks
    # "to" is in the to/too confusion set and occurs in the clean fixture records.
    assert sum(len(c.probes) for c in checks.values()) > 0
