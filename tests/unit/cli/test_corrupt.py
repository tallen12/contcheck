from contcheck.cli.corrupt import build_registry
from contcheck.taxonomy.kinds import ErrorKind


def test_registry_has_the_proof_of_concept_kinds() -> None:
    assert build_registry().kinds() == (
        ErrorKind.SPELLING,
        ErrorKind.REAL_WORD,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.VERB_TENSE,
        ErrorKind.PRONOUN_FORM,
        ErrorKind.DETERMINER,
        ErrorKind.PREPOSITION,
        ErrorKind.CONJUNCTION,
    )
