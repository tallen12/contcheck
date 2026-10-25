from contcheck.cli.corrupt import build_registry, keep_every_confusion
from contcheck.corrupt.confusables import Confusion, ConfusionSource
from contcheck.taxonomy.kinds import ErrorKind


def test_registry_has_the_proof_of_concept_kinds() -> None:
    assert build_registry().kinds() == (
        ErrorKind.SPELLING,
        ErrorKind.REAL_WORD,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.VERB_TENSE,
        ErrorKind.DETERMINER,
    )


def test_every_confusion_is_kept_by_default() -> None:
    confusion = Confusion("then", "than", 1.0, ConfusionSource.HAND_WRITTEN)

    assert keep_every_confusion(confusion)
