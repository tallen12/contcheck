from contcheck.infer.models import SpanError
from contcheck.infer.protocols import FormalityClassifier
from contcheck.taxonomy.kinds import ErrorKind, Formality
from tests.unit.infer.conftest import FlatAdverbs


def test_formality_classifier_contract() -> None:
    classifier: FormalityClassifier = FlatAdverbs()
    words = ["we", "worked", "quick", "and", "careful"]

    assert (
        classifier.classify(words, SpanError(2, 3, ErrorKind.ADJECTIVE_ADVERB, 0.9))
        is Formality.INFORMAL
    )
    assert (
        classifier.classify(words, SpanError(4, 5, ErrorKind.ADJECTIVE_ADVERB, 0.9))
        is Formality.ERROR
    )
