import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from contcheck.analysis.models import TenseHistogram, join_words
from contcheck.corpus.record import revert_edits
from contcheck.corrupt.apply import (
    apply_edits,
    choose_edits,
    clashes,
    corrupt_paragraph,
    match_case,
    touches,
    weighted_choice,
)
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
R, M, U = Operation.REPLACEMENT, Operation.MISSING, Operation.UNNECESSARY


def context(analyzer: FakeAnalyzer, text: str) -> CorruptionContext:
    return CorruptionContext(analyzer.analyze(text).tokens, NO_TENSE)


def test_replacement_keeps_case_and_space(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "Yesterday I went home.")
    edits = [
        Edit(0, 1, ErrorKind.SPELLING, R, ("Yesterday",), ("yestreday",)),
        Edit(2, 3, ErrorKind.VERB_TENSE, R, ("went",), ("go",)),
    ]

    result = apply_edits(ctx, edits)

    assert result.words == ("Yestreday", "I", "go", "home", ".")
    assert join_words(result.words, result.spaces) == "Yestreday I go home."
    assert result.labels == ("B-Spelling", "O", "B-VerbTense", "O", "O")


def test_capitalization_edits_keep_their_casing(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw London.")

    result = apply_edits(ctx, [Edit(2, 3, ErrorKind.CAPITALIZATION, R, ("London",), ("london",))])

    assert result.words[2] == "london"


def test_missing_word_labels_the_word_after_the_gap(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw the big dog.")

    result = apply_edits(ctx, [Edit(2, 3, ErrorKind.DETERMINER, M, ("the",), ())])

    assert join_words(result.words, result.spaces) == "I saw big dog."
    assert result.labels == ("O", "O", "B-Determiner", "O", "O")
    assert result.edits[0].start == result.edits[0].end == 2


def test_deleting_before_punctuation_hands_over_the_space(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw the dog.")

    result = apply_edits(ctx, [Edit(3, 4, ErrorKind.MISSING_WORD, M, ("dog",), ())])

    assert join_words(result.words, result.spaces) == "I saw the."
    assert result.labels[3] == "B-MissingWord"


def test_missing_at_the_end_labels_the_word_before(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "we walk home")

    result = apply_edits(ctx, [Edit(2, 3, ErrorKind.MISSING_WORD, M, ("home",), ())])

    assert result.words == ("we", "walk")
    assert result.labels == ("O", "B-MissingWord")


def test_insertion_labels_the_inserted_word(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw dogs and a cat.")

    result = apply_edits(
        ctx,
        [
            Edit(2, 2, ErrorKind.DETERMINER, U, (), ("the",)),
            Edit(5, 6, ErrorKind.SPELLING, R, ("cat",), ("czt",)),
        ],
    )

    assert join_words(result.words, result.spaces) == "I saw the dogs and a czt."
    assert result.labels == ("O", "O", "B-Determiner", "O", "O", "O", "B-Spelling", "O")
    assert [(e.start, e.end) for e in result.edits] == [(2, 3), (6, 7)]


def test_multi_word_replacement_spans_every_new_word(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw everyone there.")

    result = apply_edits(
        ctx, [Edit(2, 3, ErrorKind.WORD_BOUNDARY, R, ("everyone",), ("every", "one"))]
    )

    assert result.words == ("I", "saw", "every", "one", "there", ".")
    assert result.labels[2:4] == ("B-WordBoundary", "I-WordBoundary")


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((1, 2), (2, 3), True),
        ((1, 2), (3, 4), False),
        ((2, 2), (2, 3), True),
        ((2, 2), (1, 2), True),
        ((3, 3), (1, 2), False),
    ],
)
def test_clashes_need_an_untouched_word_between(
    first: tuple[int, int], second: tuple[int, int], expected: bool
) -> None:
    def edit(start: int, end: int) -> Edit:
        if start == end:
            return Edit(start, end, ErrorKind.EXTRA_WORD, U, (), ("x",))
        return Edit(start, end, ErrorKind.SPELLING, R, ("a",) * (end - start), ("b",))

    assert clashes(edit(*first), [edit(*second)]) is expected


def test_apply_rejects_overlapping_edits(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "a b c d")
    edits = [
        Edit(0, 2, ErrorKind.SPELLING, R, ("a", "b"), ("x",)),
        Edit(1, 2, ErrorKind.SPELLING, R, ("b",), ("y",)),
    ]

    with pytest.raises(ValueError, match="clashes"):
        apply_edits(ctx, edits)


def test_neighbouring_edits_get_separate_spans(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "the libary tomorrow is open")
    edits = [
        Edit(1, 2, ErrorKind.SPELLING, R, ("libary",), ("libbary",)),
        Edit(2, 3, ErrorKind.SPELLING, R, ("tomorrow",), ("tomorow",)),
    ]

    result = apply_edits(ctx, edits)

    assert result.labels == ("O", "B-Spelling", "B-Spelling", "O", "O")
    assert revert_edits(result.words, result.edits) == ctx.words


def test_missing_edits_never_touch(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw the dog")
    edits = [
        Edit(1, 2, ErrorKind.SPELLING, R, ("saw",), ("sav",)),
        Edit(2, 3, ErrorKind.DETERMINER, M, ("the",), ()),
    ]

    with pytest.raises(ValueError, match="clashes"):
        apply_edits(ctx, edits)


def test_touching_is_allowed_only_on_request() -> None:
    first = Edit(1, 2, ErrorKind.SPELLING, R, ("a",), ("x",))
    neighbour = Edit(2, 3, ErrorKind.SPELLING, R, ("b",), ("y",))
    apart = Edit(4, 5, ErrorKind.SPELLING, R, ("c",), ("z",))

    assert clashes(neighbour, [first])
    assert not clashes(neighbour, [first], allow_touching=True)
    assert touches(neighbour, [first])
    assert not touches(apart, [first])


def test_revert_restores_clean_words(analyzer: FakeAnalyzer) -> None:
    ctx = context(analyzer, "I saw the big dog and a cat.")
    edits = [
        Edit(0, 1, ErrorKind.SPELLING, R, ("I",), ("Ii",)),
        Edit(2, 3, ErrorKind.DETERMINER, M, ("the",), ()),
        Edit(5, 5, ErrorKind.EXTRA_WORD, U, (), ("very", "very")),
        Edit(7, 8, ErrorKind.SPELLING, R, ("cat",), ("czt",)),
    ]

    result = apply_edits(ctx, edits)

    assert revert_edits(result.words, result.edits) == ctx.words


def test_match_case() -> None:
    assert match_case("The", "a") == "A"
    assert match_case("NASA", "nasa") == "NASA"
    assert match_case("went", "Go") == "Go"
    assert match_case("I", "me") == "Me"


def test_weighted_choice_follows_weights(rng: random.Random) -> None:
    picks = [weighted_choice({"a": 1.0, "b": 3.0, "c": 0.0}, rng) for _ in range(4000)]

    assert picks.count("c") == 0
    assert picks.count("b") / len(picks) == pytest.approx(0.75, abs=0.03)
    with pytest.raises(ValueError):
        weighted_choice({"a": 0.0}, rng)


class AlmostOne:
    """An ``Rng`` whose ``random`` is as close to 1 as floats allow."""

    def random(self) -> float:
        return 1.0 - 2**-53

    def randrange(self, stop: int, /) -> int:
        return stop - 1

    def choice[T](self, seq: Sequence[T], /) -> T:
        return seq[-1]


def test_weighted_choice_never_returns_a_zero_weight_key() -> None:
    assert weighted_choice({"a": 0.1, "b": 0.2, "c": 0.0}, AlmostOne()) == "b"


@dataclass
class Scripted:
    """Proposes the next edit from a script, then ``None``."""

    kind: ErrorKind
    script: list[Edit | None]
    calls: int = 0

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        self.calls += 1
        return self.script.pop(0) if self.script else None


def config(count: int, weights: dict[str, float]) -> GenConfig:
    return GenConfig(
        seed=1, variants_per_paragraph=1, errors_per_paragraph={count: 1.0}, kind_weights=weights
    )


def test_choose_retries_another_kind_when_one_has_no_site(
    analyzer: FakeAnalyzer, rng: random.Random
) -> None:
    ctx = context(analyzer, "a b c d e")
    empty = Scripted(ErrorKind.REAL_WORD, [])
    useful = Scripted(ErrorKind.SPELLING, [Edit(0, 1, ErrorKind.SPELLING, R, ("a",), ("x",))])
    registry = CorruptorRegistry([empty, useful])

    edits = choose_edits(ctx, registry, config(1, {"default": 1}), rng)

    assert [e.kind for e in edits] == [ErrorKind.SPELLING]


def test_choose_discards_clashing_edits(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    ctx = context(analyzer, "a b c d e")
    first = Edit(0, 1, ErrorKind.SPELLING, R, ("a",), ("x",))
    clash = Edit(1, 2, ErrorKind.SPELLING, R, ("b",), ("y",))
    apart = Edit(3, 4, ErrorKind.SPELLING, R, ("d",), ("z",))
    registry = CorruptorRegistry([Scripted(ErrorKind.SPELLING, [first, clash, apart])])

    edits = choose_edits(ctx, registry, config(2, {"Spelling": 1}), rng)

    assert edits == [first, apart]


def test_choose_skips_kinds_without_weight(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    ctx = context(analyzer, "a b c")
    unweighted = Scripted(ErrorKind.REAL_WORD, [Edit(0, 1, ErrorKind.REAL_WORD, R, ("a",), ("x",))])

    edits = choose_edits(ctx, CorruptorRegistry([unweighted]), config(1, {"Spelling": 1}), rng)

    assert edits == []
    assert unweighted.calls == 0


def test_zero_errors_leaves_the_paragraph_clean(analyzer: FakeAnalyzer, rng: random.Random) -> None:
    ctx = context(analyzer, "I went home.")
    registry = CorruptorRegistry(
        [Scripted(ErrorKind.SPELLING, [Edit(0, 1, ErrorKind.SPELLING, R, ("I",), ("x",))])]
    )

    result = corrupt_paragraph(ctx, registry, config(0, {"default": 1}), rng)

    assert result.words == ctx.words
    assert set(result.labels) == {"O"}
    assert result.edits == ()


def test_adjacent_paragraphs_get_neighbouring_edits(
    analyzer: FakeAnalyzer, rng: random.Random
) -> None:
    ctx = context(analyzer, "a b c d e f g h")
    script: list[Edit | None] = [
        Edit(2, 3, ErrorKind.SPELLING, R, ("c",), ("x",)),
        Edit(6, 7, ErrorKind.SPELLING, R, ("g",), ("y",)),
        Edit(3, 4, ErrorKind.SPELLING, R, ("d",), ("z",)),
    ]
    registry = CorruptorRegistry([Scripted(ErrorKind.SPELLING, script)])
    adjacent = config(2, {"Spelling": 1}).model_copy(update={"adjacent_share": 1.0})

    edits = choose_edits(ctx, registry, adjacent, rng)

    assert [(e.start, e.end) for e in edits] == [(2, 3), (3, 4)]


def test_adjacent_mode_falls_back_to_separate_edits(
    analyzer: FakeAnalyzer, rng: random.Random
) -> None:
    ctx = context(analyzer, "a b c d e f g h")
    far: list[Edit | None] = [Edit(6, 7, ErrorKind.SPELLING, R, ("g",), ("y",))] * 30
    script = [Edit(2, 3, ErrorKind.SPELLING, R, ("c",), ("x",)), *far]
    registry = CorruptorRegistry([Scripted(ErrorKind.SPELLING, script)])
    adjacent = config(2, {"Spelling": 1}).model_copy(update={"adjacent_share": 1.0})

    edits = choose_edits(ctx, registry, adjacent, rng)

    assert [(e.start, e.end) for e in edits] == [(2, 3), (6, 7)]
