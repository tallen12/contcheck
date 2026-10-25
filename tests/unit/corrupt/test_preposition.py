import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.preposition import (
    CONFUSIONS,
    PrepositionCorruptor,
    add_preposition,
    drop_preposition,
    swap_preposition,
)
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import DEFAULT_LEXICON, FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
LEXICON = {
    **DEFAULT_LEXICON,
    "for": ("IN", "for"),
    "at": ("IN", "at"),
    "because": ("IN", "because"),
    "waited": ("VBD", "wait"),
    "discussed": ("VBD", "discuss"),
    "gave": ("VBD", "give"),
    "him": ("PRP", "he"),
    "plan": ("NN", "plan"),
    "about": ("IN", "about"),
    "into": ("IN", "into"),
    "book": ("NN", "book"),
}
ANALYZER = FakeAnalyzer(LEXICON)


def context(text: str) -> CorruptionContext:
    return CorruptionContext(ANALYZER.analyze(text).tokens, NO_TENSE)


def test_swap_picks_a_common_confusion(rng: random.Random) -> None:
    edit = swap_preposition(context("I waited at the park."), rng)

    assert edit is not None
    assert (edit.start, edit.op, edit.orig) == (2, Operation.REPLACEMENT, ("at",))
    assert edit.new[0] in CONFUSIONS["at"]


def test_swap_keeps_subordinators_and_infinitives_out(rng: random.Random) -> None:
    assert swap_preposition(context("I walk because I want to go."), rng) is None


def test_swap_takes_a_to_tagged_to_before_a_noun_phrase(rng: random.Random) -> None:
    edit = swap_preposition(context("I went to the park."), rng)

    assert edit is not None
    assert edit.orig == ("to",)


def test_drop_removes_a_preposition_between_verb_and_object(rng: random.Random) -> None:
    assert drop_preposition(context("I waited for the dog."), rng) == Edit(
        2, 3, ErrorKind.PREPOSITION, Operation.MISSING, ("for",), ()
    )


def test_drop_skips_pronoun_objects_and_undroppable_words(rng: random.Random) -> None:
    assert drop_preposition(context("I gave to him. I walked in the park."), rng) is None


def test_add_puts_the_learner_preposition_after_a_transitive_verb(rng: random.Random) -> None:
    assert add_preposition(context("We discussed the plan."), rng) == Edit(
        2, 2, ErrorKind.PREPOSITION, Operation.UNNECESSARY, (), ("about",)
    )


def test_add_needs_a_listed_verb(rng: random.Random) -> None:
    assert add_preposition(context("We walked the dog."), rng) is None


def test_corruptor_prefers_swaps() -> None:
    corruptor = PrepositionCorruptor()
    ctx = context("I waited for the dog.")

    ops = [e.op for seed in range(200) if (e := corruptor.corrupt(ctx, random.Random(seed)))]

    assert len(ops) == 200
    assert ops.count(Operation.REPLACEMENT) > ops.count(Operation.MISSING) > 0


def test_corruptor_falls_back_to_an_operation_with_a_site() -> None:
    corruptor = PrepositionCorruptor()
    ctx = context("We discussed the plan.")

    edits = [corruptor.corrupt(ctx, random.Random(seed)) for seed in range(20)]

    assert all(e is not None and e.op is Operation.UNNECESSARY for e in edits)


def test_corruptor_without_any_site(rng: random.Random) -> None:
    assert PrepositionCorruptor().corrupt(context("Dogs walk."), rng) is None


@pytest.mark.parametrize("text", ["I read a book about dogs.", "I walk into the park."])
def test_swaps_that_often_fit_are_not_made(text: str, rng: random.Random) -> None:
    # about/on and into/in usually both fit, so swapping them would label good English.
    assert swap_preposition(context(text), rng) is None


def test_about_can_still_be_dropped(rng: random.Random) -> None:
    assert drop_preposition(context("I waited about the plan."), rng) == Edit(
        2, 3, ErrorKind.PREPOSITION, Operation.MISSING, ("about",), ()
    )
