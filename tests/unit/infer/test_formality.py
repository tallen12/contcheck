from contcheck.infer.formality import classify_formality, keep_formalities
from contcheck.infer.models import SpanError
from contcheck.taxonomy.kinds import ErrorKind, Formality
from tests.unit.infer.conftest import FlatAdverbs

WORDS = ["we", "worked", "quick", "and", "careful"]
ERRORS = [
    SpanError(2, 3, ErrorKind.ADJECTIVE_ADVERB, 0.9),
    SpanError(4, 5, ErrorKind.ADJECTIVE_ADVERB, 0.8),
]


def test_decoded_errors_start_as_errors() -> None:
    assert all(error.formality is Formality.ERROR for error in ERRORS)


def test_classify_attaches_formality() -> None:
    classified = classify_formality(WORDS, ERRORS, FlatAdverbs())

    assert [e.formality for e in classified] == [Formality.INFORMAL, Formality.ERROR]
    assert [(e.start, e.probability) for e in classified] == [(2, 0.9), (4, 0.8)]


def test_keep_formalities_hides_informal_flags() -> None:
    classified = classify_formality(WORDS, ERRORS, FlatAdverbs())

    assert [e.start for e in keep_formalities(classified, {Formality.ERROR})] == [4]
    assert len(keep_formalities(classified, set(Formality))) == 2
