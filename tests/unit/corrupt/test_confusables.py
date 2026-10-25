from collections.abc import Mapping

import pytest

from contcheck.corrupt.confusables import (
    DEFAULT_SETTINGS,
    Confusion,
    ConfusionMap,
    ConfusionSettings,
    ConfusionSource,
    american_spelling,
    build_confusion_map,
    distinct_words,
    hand_written_map,
    homophone_groups,
    load_homophone_groups,
    look_alikes,
    most_frequent,
    parse_homophone_groups,
    same_stem,
)
from tests.conftest import FakeDictionary

FREQUENCIES: Mapping[str, int] = {
    "from": 10_000,
    "form": 1_000,
    "forms": 900,
    "farm": 100,
    "firm": 100,
    "color": 500,
    "colour": 400,
    "supported": 300,
    "supporter": 200,
    "john": 800,
    "to": 20_000,
    "too": 2_000,
    "two": 3_000,
    "replicate": 100,
    "duplicate": 100,
    "rare": 1,
}
LEMMAS: Mapping[str, frozenset[str]] = {
    word: frozenset({word}) for word in FREQUENCIES if word not in {"john", "forms"}
} | {"forms": frozenset({"form"})}


class FakeLemmatizer:
    """A ``Lemmatizer`` over a fixed table; words outside it are unknown."""

    def __init__(self, table: Mapping[str, frozenset[str]] = LEMMAS) -> None:
        self._table = table

    def lemmas(self, word: str) -> frozenset[str]:
        return self._table.get(word.lower(), frozenset[str]())


@pytest.fixture
def words() -> FakeDictionary:
    return FakeDictionary(FREQUENCIES)


def build(words: FakeDictionary, *, settings: ConfusionSettings = DEFAULT_SETTINGS) -> ConfusionMap:
    return build_confusion_map(
        frequencies=FREQUENCIES,
        dictionary=words,
        lemmatizer=FakeLemmatizer(),
        confusion_sets={"to": ("too",), "too": ("to",)},
        homophones=[["to", "too", "two"], ["i", "eye"]],
        settings=settings,
    )


def test_look_alikes_are_one_edit_apart_and_distinct(words: FakeDictionary) -> None:
    pairs = look_alikes(
        FREQUENCIES,
        dictionary=words,
        lemmatizer=FakeLemmatizer(),
        settings=ConfusionSettings(),
    )

    assert frozenset({"form", "from"}) not in pairs  # two edits under plain edit distance
    assert pairs[frozenset({"form", "farm"})] == 1.0
    assert frozenset({"form", "forms"}) not in pairs  # same lemma
    assert frozenset({"color", "colour"}) not in pairs  # spelling variant
    assert frozenset({"supported", "supporter"}) not in pairs  # same stem
    assert not any("john" in pair for pair in pairs)  # not a known word
    assert frozenset({"to", "two"}) not in pairs  # too short


def test_long_words_may_be_two_edits_apart_at_lower_weight(words: FakeDictionary) -> None:
    on = look_alikes(
        FREQUENCIES, dictionary=words, lemmatizer=FakeLemmatizer(), settings=ConfusionSettings()
    )
    off = look_alikes(
        FREQUENCIES,
        dictionary=words,
        lemmatizer=FakeLemmatizer(),
        settings=ConfusionSettings(long_word_length=None),
    )

    assert on[frozenset({"replicate", "duplicate"})] == 0.25
    assert frozenset({"replicate", "duplicate"}) not in off


def test_map_merges_sources_keeping_the_strongest(words: FakeDictionary) -> None:
    confusions = build(words)

    sources = {c.alternative: c.source for c in confusions.alternatives("to")}
    assert sources == {"too": ConfusionSource.HAND_WRITTEN, "two": ConfusionSource.HOMOPHONE}
    assert confusions.alternatives("eye") == ()  # one-letter homophones are skipped
    assert {c.alternative for c in confusions.alternatives("form")} == {"farm", "firm"}


def test_common_pairs_weigh_more_and_words_are_capped(words: FakeDictionary) -> None:
    confusions = build(words, settings=ConfusionSettings(word_cap=100.0))
    capped = build(words, settings=ConfusionSettings(word_cap=1.0, max_alternatives=1))

    weights = {c.alternative: c.weight for c in confusions.alternatives("two")}
    assert weights["to"] > weights["too"]  # the rarer word of to/two is more frequent
    assert confusions.alternatives("farm")[0].weight == pytest.approx(3.0)  # 1 + log10(100 / 1)
    assert len(capped.alternatives("two")) == 1
    assert capped.weight("two") == pytest.approx(1.0)
    assert capped.weight("farm") == pytest.approx(1.0)


def test_build_is_deterministic(words: FakeDictionary) -> None:
    assert build(words) == build(words)


def test_filter_drops_both_directions_of_a_pair(words: FakeDictionary) -> None:
    confusions = build(words)
    held_out = {frozenset({"to", "too"})}

    kept = confusions.filter(lambda c: c.pair not in held_out)

    assert frozenset({"to", "too"}) in confusions.pairs()
    assert frozenset({"to", "too"}) not in kept.pairs()
    assert len(kept) == len(confusions) - 2
    assert {c.alternative for c in kept.alternatives("To")} == {"two"}


def test_map_groups_heaviest_first() -> None:
    confusions = ConfusionMap.of(
        [
            Confusion("a", "b", 1.0, ConfusionSource.LOOK_ALIKE),
            Confusion("a", "c", 2.0, ConfusionSource.HOMOPHONE),
        ]
    )

    assert [c.alternative for c in confusions] == ["c", "b"]
    assert confusions.weight("A") == 3.0
    assert confusions.weight("z") == 0.0


def test_hand_written_map_weighs_alternatives_equally() -> None:
    confusions = hand_written_map({"cite": ("site", "sight")})

    assert [(c.alternative, c.weight) for c in confusions] == [("sight", 1.0), ("site", 1.0)]


def test_most_frequent_breaks_ties_alphabetically() -> None:
    assert list(most_frequent({"b": 2, "a": 2, "c": 3}, 2)) == ["c", "a"]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("optional", "optionally", True),
        ("successful", "unsuccessful", True),
        ("invented", "inventor", True),
        ("the", "they", False),
        ("grip", "rip", False),
        ("then", "than", False),
        ("banking", "ranking", False),
    ],
)
def test_same_stem(a: str, b: str, expected: bool) -> None:
    assert same_stem(a, b) is expected


@pytest.mark.parametrize(
    ("british", "american"),
    [
        ("colour", "color"),
        ("realised", "realized"),
        ("centre", "center"),
        ("fibre", "fiber"),
        ("travelled", "traveled"),
        ("defence", "defense"),
        ("catalogue", "catalog"),
        ("judgement", "judgment"),
        ("grey", "gray"),
        ("four", "four"),
    ],
)
def test_american_spelling(british: str, american: str) -> None:
    assert american_spelling(british) == american


def test_distinct_words_needs_two_known_unrelated_words() -> None:
    lemmatizer = FakeLemmatizer()

    assert distinct_words("form", "farm", lemmatizer)
    assert not distinct_words("form", "Form", lemmatizer)
    assert not distinct_words("form", "forms", lemmatizer)
    assert not distinct_words("form", "john", lemmatizer)
    assert not distinct_words("color", "colour", lemmatizer)
    assert not distinct_words("form", "f-rm", lemmatizer)


def test_homophone_groups_use_first_pronunciations_in_the_vocabulary() -> None:
    lines = [
        ";;; comment",
        "to T UW1",
        "too T UW1",
        "two T UW1 # the number",
        "or AO1 R",
        "or(2) ER0",
        "err ER0",
        "tu T UW1",
        "o'neil OW0 N IY1 L",
    ]

    groups = homophone_groups(lines, {"to", "too", "two", "or", "err", "o'neil"})

    assert groups == [["to", "too", "two"]]


def test_parse_homophone_groups_allows_repeats() -> None:
    groups = parse_homophone_groups(["# header", "read reed", "read red  # past", "solo"])

    assert groups == [["read", "reed"], ["read", "red"]]


def test_bundled_homophones() -> None:
    groups = load_homophone_groups()

    assert ["their", "there"] in groups
    assert len(groups) > 1000
