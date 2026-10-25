import pytest

from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.models import EvalResult, Scores


def test_scores_weight_precision_over_recall() -> None:
    scores = Scores(true_positives=6, false_positives=2, false_negatives=6)

    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == pytest.approx(0.5)
    assert scores.f05 == pytest.approx(1.25 * 0.75 * 0.5 / (0.25 * 0.75 + 0.5))


def test_empty_scores_are_zero() -> None:
    assert (Scores().precision, Scores().recall, Scores().f05) == (0.0, 0.0, 0.0)


def test_scores_add() -> None:
    total = Scores(true_positives=1) + Scores(false_positives=2, false_negatives=3)

    assert total == Scores(true_positives=1, false_positives=2, false_negatives=3)


def test_eval_result_round_trips_through_json() -> None:
    result = EvalResult(
        name="m",
        dataset="val",
        per_kind={ErrorKind.SPELLING: Scores(true_positives=1)},
        overall=Scores(true_positives=1),
        detection=Scores(true_positives=1),
        confusion={"Spelling": {"Spelling": 1}},
        clean_words=10,
        false_alarms_per_1000=0.0,
    )

    restored = EvalResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert '"f05":1.0' in result.model_dump_json()
