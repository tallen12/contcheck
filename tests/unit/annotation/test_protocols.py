from contcheck.annotation.models import Annotated, Unmappable
from contcheck.annotation.protocols import Annotator
from contcheck.corpus.record import revert_edits
from contcheck.taxonomy.kinds import ErrorKind
from tests.conftest import FakeAnnotator

NOISY = ("I", "go", "home", ".")
CLEAN = ("I", "went", "home", ".")


def test_annotator_contract() -> None:
    implementation: Annotator = FakeAnnotator(
        {("went",): ErrorKind.VERB_TENSE}, unmappable=frozenset({("gone",)})
    )

    result = implementation.annotate(NOISY, CLEAN)
    rejected = implementation.annotate(NOISY, ("I", "gone", "home", "."))

    assert isinstance(result, Annotated)
    assert [edit.kind for edit in result.edits] == [ErrorKind.VERB_TENSE]
    assert revert_edits(NOISY, result.edits) == CLEAN
    assert isinstance(rejected, Unmappable)
    assert list(implementation.annotate_many([(CLEAN, CLEAN), (NOISY, CLEAN)])) == [
        Annotated(()),
        result,
    ]
