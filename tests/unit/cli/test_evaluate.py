from pathlib import Path

import pytest

from contcheck.cli.evaluate import build_checks, evaluate
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.confusables import hand_written_map
from contcheck.corrupt.real_word import load_confusion_sets
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


def test_build_checks_adds_held_out_pairs_only_when_some_are_withheld() -> None:
    records = [make_record(i, noisy=i % 2 == 0) for i in range(4)]
    confusions = hand_written_map(load_confusion_sets())

    def gen_config(share: float) -> GenConfig:
        return GenConfig(
            seed=13,
            variants_per_paragraph=1,
            errors_per_paragraph={1: 1.0},
            kind_weights={"default": 1},
            held_out_confusion_share=share,
        )

    none_held = build_checks(records, gen_config=gen_config(0.0), confusions=confusions)
    some_held = build_checks(records, gen_config=gen_config(0.5), confusions=confusions)

    assert set(none_held) == {CheckName.REAL_MISSPELLINGS, CheckName.SEEN_CONFUSIONS}
    assert CheckName.HELD_OUT_CONFUSIONS in some_held
    # "to" is in the to/too confusion set and occurs in the clean fixture records.
    assert sum(len(c.probes) for c in some_held.values()) > 0
