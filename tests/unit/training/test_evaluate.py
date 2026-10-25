from pathlib import Path

import numpy as np
import numpy.typing as npt

from contcheck.corpus.record import Record
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.taxonomy.labels import LABEL2ID, LABELS
from contcheck.training.evaluate import (
    append_result,
    evaluate_tagger,
    format_result,
    symspell_baseline,
)
from contcheck.training.models import EvalResult
from contcheck.training.probes import CheckName, ProbeSet, substitution_probes
from tests.conftest import FakeDictionary, FakeTokenizer

GO = FakeTokenizer().piece_id("go")


class PerfectTenseTagger:
    """Flags every subword ``go`` as B-VerbTense, which is right for the fixture records."""

    def logits(
        self, input_ids: npt.NDArray[np.int64], attention_mask: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        logits = np.zeros((*input_ids.shape, len(LABELS)), dtype=np.float32)
        logits[..., LABEL2ID["O"]] = 5.0
        logits[input_ids == GO, LABEL2ID["B-VerbTense"]] = 10.0
        return logits


def test_evaluate_tagger_scores_against_record_labels(
    records: list[Record], tokenizer: FakeTokenizer
) -> None:
    result = evaluate_tagger(
        records,
        tagger=PerfectTenseTagger(),
        tokenizer=tokenizer,
        window=16,
        stride=8,
        batch_size=4,
        name="perfect",
        dataset="fixture",
    )

    assert result.per_kind[ErrorKind.VERB_TENSE].f05 == 1.0
    assert result.false_alarms_per_1000 == 0.0
    assert result.clean_words == 8 * 7


def test_symspell_baseline_flags_unknown_lowercase_words(records: list[Record]) -> None:
    result = symspell_baseline(records, dictionary=FakeDictionary(), dataset="fixture")

    # "store" and "yesterday" are known; "go" is known, so the tense errors are missed.
    assert result.name == "symspell-baseline"
    assert result.per_kind[ErrorKind.VERB_TENSE].recall == 0.0
    assert result.overall.false_positives == 0


def test_results_log_appends_lines(tmp_path: Path, records: list[Record]) -> None:
    result = symspell_baseline(records, dictionary=FakeDictionary(), dataset="fixture")
    path = tmp_path / "runs" / "results.jsonl"

    append_result(path, result)
    append_result(path, result)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert EvalResult.model_validate_json(lines[0]) == result


def test_format_result_lists_kinds_and_summary(
    records: list[Record], tokenizer: FakeTokenizer
) -> None:
    result = evaluate_tagger(
        records,
        tagger=PerfectTenseTagger(),
        tokenizer=tokenizer,
        window=16,
        stride=8,
        batch_size=4,
        name="perfect",
        dataset="fixture",
    )

    text = format_result(result)

    assert text.splitlines()[0] == "perfect on fixture"
    assert "VerbTense               1.000  1.000  1.000       8" in text
    assert "false alarms per 1,000 clean words: 0.00 (56 words)" in text


def test_checks_are_scored_and_shown(records: list[Record], tokenizer: FakeTokenizer) -> None:
    misspelled = substitution_probes(records, {"store": ("stor",), "yesterday": ("yesturday",)})
    checks = {CheckName.REAL_MISSPELLINGS: ProbeSet(ErrorKind.SPELLING, misspelled)}

    tagged = evaluate_tagger(
        records,
        tagger=PerfectTenseTagger(),
        tokenizer=tokenizer,
        window=16,
        stride=8,
        batch_size=4,
        name="perfect",
        dataset="fixture",
        checks=checks,
    )
    baseline = symspell_baseline(
        records, dictionary=FakeDictionary(), dataset="fixture", checks=checks
    )

    tagged_check = tagged.checks[CheckName.REAL_MISSPELLINGS]
    baseline_check = baseline.checks[CheckName.REAL_MISSPELLINGS]
    assert tagged_check.probes == baseline_check.probes == len(misspelled) > 0
    assert tagged_check.detected == 0.0
    assert baseline_check.correct_kind == 1.0
    assert "real-misspellings: flagged 1.000, with the right kind 1.000" in format_result(baseline)
    plain = symspell_baseline(records, dictionary=FakeDictionary(), dataset="x")
    assert "real-misspellings" not in format_result(plain)
