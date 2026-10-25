import random

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.determiner import (
    DeterminerCorruptor,
    add_article,
    drop_article,
    swap_article,
)
from contcheck.corrupt.edit import Edit
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)


def context(analyzer: FakeAnalyzer, text: str) -> CorruptionContext:
    return CorruptionContext(analyzer.analyze(text).tokens, NO_TENSE)


def test_drop_skips_sentence_initial_articles(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    edit = drop_article(context(analyzer, "The dog is in the park."), rng)

    assert edit == Edit(4, 5, ErrorKind.DETERMINER, Operation.MISSING, ("the",), ())


def test_drop_needs_an_article(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    assert drop_article(context(analyzer, "The dogs walk."), rng) is None


def test_add_puts_an_indefinite_before_plurals(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    edit = add_article(context(analyzer, "I bought apples."), rng)

    assert edit == Edit(2, 2, ErrorKind.DETERMINER, Operation.UNNECESSARY, (), ("an",))


def test_add_puts_the_before_singular_phrases(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    edit = add_article(context(analyzer, "I walk to big house."), rng)

    assert edit is not None
    assert edit.new == ("the",)
    assert edit.start == 3


def test_add_needs_a_bare_phrase_after_a_verb(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    assert add_article(context(analyzer, "The dog."), rng) is None


def test_swap_turns_the_into_the_right_indefinite(
    analyzer: FakeAnalyzer, rng: random.Random
) -> None:
    assert swap_article(context(analyzer, "I saw the apple."), rng) == Edit(
        2, 3, ErrorKind.DETERMINER, Operation.REPLACEMENT, ("the",), ("an",)
    )


def test_swap_of_an_indefinite_gives_the_or_the_other_form(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw a dog.")

    news = {e.new for seed in range(20) if (e := swap_article(ctx, random.Random(seed)))}

    assert news == {("the",), ("an",)}


def test_corruptor_falls_back_to_an_operation_with_a_site(analyzer: FakeAnalyzer) -> None:
    corruptor = DeterminerCorruptor()
    ctx = context(analyzer, "I bought apples.")

    edits = [corruptor.corrupt(ctx, random.Random(seed)) for seed in range(10)]

    assert all(e is not None and e.op is Operation.UNNECESSARY for e in edits)


def test_corruptor_without_any_site(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    assert DeterminerCorruptor().corrupt(context(analyzer, "Dogs."), rng) is None
