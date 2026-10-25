import random
from collections.abc import Mapping, Sequence

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.word_choice import DEFAULT_SETTINGS, WordChoiceCorruptor, WordChoiceSettings
from contcheck.lm.models import MaskedSlot, SlotPrediction, WordScore
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import DEFAULT_LEXICON, FakeAnalyzer

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
TAGS = {
    **DEFAULT_LEXICON,
    "in": ("IN", "in"),
    "garden": ("NN", "garden"),
    "park": ("NN", "park"),
    "owl": ("NN", "owl"),
    "saw": ("VBD", "see"),
    "gardens": ("NNS", "garden"),
    "grow": ("VBP", "grow"),
}
WORDS: Mapping[tuple[str, str], frozenset[str]] = {
    (word, tag): frozenset({lemma})
    for word, tag, lemma in [
        ("garden", "NN", "garden"),
        ("park", "NN", "park"),
        ("sky", "NN", "sky"),
        ("yard", "NN", "yard"),
        ("thing", "NN", "thing"),
        ("gardening", "NN", "garden"),
        ("paris", "NN", "paris"),
        ("sleep", "VB", "sleep"),
        ("owl", "NN", "owl"),
        ("eagle", "NN", "eagle"),
        ("hawk", "NN", "hawk"),
        ("gardens", "NNS", "garden"),
        ("trees", "NNS", "tree"),
    ]
}


class TableLexicon:
    def lemmas(self, word: str, tag: str) -> frozenset[str]:
        none = frozenset[str]()
        return WORDS.get((word, tag), none)


class FakeScorer:
    """Answers by the masked word: its own probability (``None``: not one token) and the top."""

    def __init__(self, table: Mapping[str, tuple[float | None, list[tuple[str, float]]]]) -> None:
        self._table = table
        self.batches: list[list[MaskedSlot]] = []

    def predict(self, slots: Sequence[MaskedSlot], *, top_k: int) -> list[SlotPrediction]:
        self.batches.append(list(slots))
        predictions = []
        for slot in slots:
            original, top = self._table.get(slot.words[slot.position], (0.0, []))
            scored = () if original is None else (WordScore(slot.words[slot.position], original),)
            predictions.append(
                SlotPrediction(tuple(WordScore(w, p) for w, p in top[:top_k]), scored)
            )
        return predictions


def context_of(text: str) -> CorruptionContext:
    return CorruptionContext(FakeAnalyzer(TAGS).analyze(text).tokens, NO_TENSE)


def replacements(
    text: str,
    scorer: FakeScorer,
    settings: WordChoiceSettings = DEFAULT_SETTINGS,
) -> set[tuple[str, str]]:
    corruptor = WordChoiceCorruptor(scorer, TableLexicon(), skip={"thing"}, settings=settings)
    context = context_of(text)
    edits = [corruptor.corrupt(context, random.Random(seed)) for seed in range(30)]
    for edit in edits:
        assert edit is None or (edit.kind, edit.op) == (ErrorKind.REAL_WORD, Operation.REPLACEMENT)
    return {(edit.orig[0], edit.new[0]) for edit in edits if edit is not None}


def test_keeps_only_candidates_the_context_clearly_rejects() -> None:
    scorer = FakeScorer(
        {
            "garden": (
                0.6,
                [
                    ("yard", 0.3),  # too probable: an acceptable alternative
                    ("park", 0.0005),
                    ("sky", 0.0004),
                    ("gardn", 0.0003),  # not a dictionary word
                    ("sleep", 0.0003),  # a verb, not a noun
                    ("gardening", 0.0002),  # same lemma
                    ("thing", 0.0002),  # skipped word
                    ("Paris", 0.0001),  # a capital inside a sentence is a name
                ],
            )
        }
    )

    assert replacements("In the garden.", scorer) == {("garden", "park"), ("garden", "sky")}


def test_ratio_threshold_rejects_candidates_close_to_the_original() -> None:
    scorer = FakeScorer({"garden": (0.3, [("park", 0.0009)])})

    assert replacements("In the garden.", scorer) == {("garden", "park")}
    assert replacements("In the garden.", scorer, WordChoiceSettings(min_ratio=1000)) == set()


def test_sites_the_model_does_not_expect_are_skipped() -> None:
    weak = FakeScorer({"garden": (0.1, [("park", 0.0001)])})
    unscorable = FakeScorer({"garden": (None, [("park", 0.0001)])})

    assert replacements("In the garden.", weak) == set()
    assert replacements("In the garden.", unscorable) == set()


def test_candidates_never_repeat_a_neighbour() -> None:
    scorer = FakeScorer({"garden": (0.6, [("park", 0.0001), ("sky", 0.0001)])})

    assert replacements("In the park garden.", scorer) == {("garden", "sky")}


def test_capitalized_candidates_are_kept_at_a_sentence_start() -> None:
    scorer = FakeScorer({"Gardens": (0.6, [("Trees", 0.0001)])})

    assert replacements("Gardens grow.", scorer) == {("Gardens", "trees")}


def test_a_and_an_stay_right() -> None:
    scorer = FakeScorer({"owl": (0.6, [("hawk", 0.0001), ("eagle", 0.0001)])})

    assert replacements("I saw an owl.", scorer) == {("owl", "eagle")}


def test_names_and_function_words_are_not_sites() -> None:
    scorer = FakeScorer({})
    corruptor = WordChoiceCorruptor(scorer, TableLexicon())

    assert corruptor.corrupt(context_of("In London."), random.Random(1)) is None
    assert scorer.batches == []


def test_sites_are_scored_in_one_batch_until_one_has_a_candidate() -> None:
    scorer = FakeScorer({"garden": (0.6, [("sky", 0.0001)]), "park": (0.6, [])})
    corruptor = WordChoiceCorruptor(
        scorer, TableLexicon(), settings=WordChoiceSettings(sites_per_call=2)
    )

    edit = corruptor.corrupt(context_of("In the park garden."), random.Random(3))

    assert edit is not None and edit.new == ("sky",)
    assert len(scorer.batches) == 1
    assert {slot.words[slot.position] for slot in scorer.batches[0]} == {"park", "garden"}


def test_slots_hold_a_window_of_context_and_the_original_as_candidate() -> None:
    scorer = FakeScorer({})
    corruptor = WordChoiceCorruptor(
        scorer, TableLexicon(), settings=WordChoiceSettings(context_words=2, sites_per_call=1)
    )
    context = context_of("I saw the park and the garden in the morning.")

    for seed in range(10):
        corruptor.corrupt(context, random.Random(seed))

    for (slot,) in scorer.batches:
        assert len(slot.words) <= 5
        assert slot.candidates == (slot.words[slot.position],)
        assert slot.words[slot.position] in context.words


@pytest.mark.parametrize("sites_per_call", [1, 3])
def test_no_candidate_at_any_site_gives_none(sites_per_call: int) -> None:
    scorer = FakeScorer({"garden": (0.6, []), "park": (0.6, [])})
    corruptor = WordChoiceCorruptor(
        scorer, TableLexicon(), settings=WordChoiceSettings(sites_per_call=sites_per_call)
    )

    assert corruptor.corrupt(context_of("In the park garden."), random.Random(1)) is None
