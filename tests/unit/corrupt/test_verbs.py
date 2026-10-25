import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.verbs import Subject, be_past_to_present, present_tag_for, subject_of
from tests.conftest import FakeAnalyzer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I walked", Subject.FIRST_SINGULAR),
        ("She walked", Subject.THIRD_SINGULAR),
        ("They walked", Subject.OTHER),
        ("The dog walked", Subject.THIRD_SINGULAR),
        ("The dogs walked", Subject.OTHER),
        ("She then walked", Subject.THIRD_SINGULAR),
        ("The dogs in London walked", Subject.THIRD_SINGULAR),
        ("The big walked", Subject.UNKNOWN),
        ("walked", Subject.UNKNOWN),
    ],
)
def test_subject_of(analyzer: FakeAnalyzer, text: str, expected: Subject) -> None:
    tags = {"then": ("RB", "then")}
    tokens = analyzer.analyze(text).tokens
    tokens = tuple(
        t
        if t.text not in tags
        else type(t)(t.text, "RB", t.lemma, t.space_after, t.sentence_start, t.finite)
        for t in tokens
    )
    context = CorruptionContext(tokens, TenseHistogram(1.0, 0.0, 0.0, 0.0))

    assert subject_of(context, len(tokens) - 1) is expected


def test_present_tag_for() -> None:
    assert present_tag_for(Subject.THIRD_SINGULAR) == "VBZ"
    assert present_tag_for(Subject.OTHER) == "VBP"
    assert present_tag_for(Subject.FIRST_SINGULAR) == "VBP"
    assert present_tag_for(Subject.UNKNOWN) is None


def test_be_past_to_present() -> None:
    assert be_past_to_present("were", Subject.UNKNOWN) == "are"
    assert be_past_to_present("Was", Subject.FIRST_SINGULAR) == "am"
    assert be_past_to_present("was", Subject.THIRD_SINGULAR) == "is"
    assert be_past_to_present("was", Subject.UNKNOWN) is None
