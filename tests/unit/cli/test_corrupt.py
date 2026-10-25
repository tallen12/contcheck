from contcheck.cli.corrupt import RULE_KINDS, build_registry
from contcheck.taxonomy.kinds import ErrorKind


def test_registry_has_the_proof_of_concept_kinds() -> None:
    assert build_registry().kinds() == (
        ErrorKind.SPELLING,
        ErrorKind.REAL_WORD,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.VERB_TENSE,
        ErrorKind.DETERMINER,
    )


def test_rule_kinds_match_the_registry() -> None:
    assert set(RULE_KINDS) == set(build_registry().kinds())
