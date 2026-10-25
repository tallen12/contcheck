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
