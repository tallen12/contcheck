import pytest

from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.metrics import ParagraphSpans, score

SPELL = ErrorKind.SPELLING
TENSE = ErrorKind.VERB_TENSE
FORM = ErrorKind.SUBJECT_VERB_AGREEMENT


def test_hits_need_the_same_span_and_kind() -> None:
    paragraphs = [
        ParagraphSpans(
            10,
            gold=(WordSpan(0, 1, SPELL), WordSpan(3, 4, TENSE), WordSpan(6, 7, SPELL)),
            predicted=(WordSpan(0, 1, SPELL), WordSpan(3, 4, FORM), WordSpan(8, 9, SPELL)),
        )
    ]

    result = score(paragraphs, name="m", dataset="val")

    assert result.per_kind[SPELL].true_positives == 1
    assert result.per_kind[SPELL].false_positives == 1
    assert result.per_kind[SPELL].false_negatives == 1
    assert result.per_kind[TENSE].false_negatives == 1
    assert result.per_kind[FORM].false_positives == 1
    assert result.overall.true_positives == 1
    assert result.overall.false_positives == 2
    assert result.detection.true_positives == 2
    assert result.detection.false_positives == 1
    assert result.confusion == {
        "Spelling": {"Spelling": 1, "O": 1},
        "VerbTense": {"SubjectVerbAgreement": 1},
        "O": {"Spelling": 1},
    }


def test_false_alarms_count_only_clean_paragraphs() -> None:
    paragraphs = [
        ParagraphSpans(500, gold=(), predicted=(WordSpan(0, 1, SPELL),)),
        ParagraphSpans(1500, gold=(), predicted=(WordSpan(0, 1, SPELL), WordSpan(4, 5, TENSE))),
        ParagraphSpans(100, gold=(WordSpan(0, 1, SPELL),), predicted=(WordSpan(2, 3, SPELL),)),
    ]

    result = score(paragraphs, name="m", dataset="val")

    assert result.clean_words == 2000
    assert result.false_alarms_per_1000 == pytest.approx(1.5)


def test_empty_input_scores_zero() -> None:
    result = score([], name="m", dataset="val")

    assert result.overall.f05 == 0.0
    assert result.per_kind == {}
    assert result.false_alarms_per_1000 == 0.0
