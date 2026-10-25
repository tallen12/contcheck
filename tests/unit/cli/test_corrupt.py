from contcheck.cli.corrupt import RULE_KINDS, build_registry, confusion_split
from contcheck.corrupt.config import GenConfig
from contcheck.taxonomy.kinds import ErrorKind


def config(share: float) -> GenConfig:
    return GenConfig(
        seed=13,
        variants_per_paragraph=1,
        errors_per_paragraph={1: 1.0},
        kind_weights={"default": 1},
        held_out_confusion_share=share,
    )


def test_registry_has_the_proof_of_concept_kinds() -> None:
    assert build_registry(confusion_split(config(0.0)).train).kinds() == (
        ErrorKind.SPELLING,
        ErrorKind.REAL_WORD,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.VERB_TENSE,
        ErrorKind.DETERMINER,
    )


def test_rule_kinds_match_the_registry() -> None:
    assert set(RULE_KINDS) == set(build_registry(confusion_split(config(0.0)).train).kinds())


def test_confusion_split_follows_the_config() -> None:
    nothing_held = confusion_split(config(0.0))
    some_held = confusion_split(config(0.3))

    assert nothing_held.held_out == {}
    assert some_held.held_out
    assert confusion_split(config(0.3)) == some_held
