import pytest

from contcheck.cli.corrupt import RULE_KINDS, build_registry, confusion_split
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.confusables import ConfusionMap, hand_written_map
from contcheck.corrupt.real_word import load_confusion_sets
from contcheck.spelling.symspell_dictionary import SymSpellDictionary, load_symspell_dictionary
from contcheck.taxonomy.kinds import ErrorKind


def config(share: float) -> GenConfig:
    return GenConfig(
        seed=13,
        variants_per_paragraph=1,
        errors_per_paragraph={1: 1.0},
        kind_weights={"default": 1},
        held_out_confusion_share=share,
    )


@pytest.fixture(scope="module")
def dictionary() -> SymSpellDictionary:
    return load_symspell_dictionary()


@pytest.fixture
def confusions() -> ConfusionMap:
    return hand_written_map(load_confusion_sets())


def test_registry_has_every_kind_with_a_corruptor(
    confusions: ConfusionMap, dictionary: SymSpellDictionary
) -> None:
    assert build_registry(confusions, dictionary=dictionary).kinds() == (
        ErrorKind.SPELLING,
        ErrorKind.REAL_WORD,
        ErrorKind.NOUN_NUMBER,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.VERB_TENSE,
        ErrorKind.VERB_FORM,
        ErrorKind.ADJECTIVE_ADVERB,
        ErrorKind.PRONOUN_FORM,
        ErrorKind.DETERMINER,
        ErrorKind.PREPOSITION,
        ErrorKind.CONJUNCTION,
    )


def test_rule_kinds_match_the_registry(
    confusions: ConfusionMap, dictionary: SymSpellDictionary
) -> None:
    assert set(RULE_KINDS) == set(build_registry(confusions, dictionary=dictionary).kinds())


def test_confusion_split_follows_the_config(confusions: ConfusionMap) -> None:
    nothing_held = confusion_split(config(0.0), confusions)
    some_held = confusion_split(config(0.3), confusions)

    assert nothing_held.held_out == {}
    assert some_held.held_out
    assert confusion_split(config(0.3), confusions) == some_held


def test_training_map_keeps_only_training_pairs_with_weights(confusions: ConfusionMap) -> None:
    split = confusion_split(config(0.3), confusions)

    training = confusions.restrict(split.train)

    assert training.as_sets() == {w: alts for w, alts in split.train.items()}
    assert all(c.weight > 0 for c in training)
