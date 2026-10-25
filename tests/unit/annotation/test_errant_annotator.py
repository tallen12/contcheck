from collections.abc import Mapping

import pytest
import spacy
from errant.edit import Edit
from spacy.tokens import Doc

from contcheck.annotation.errant_annotator import ErrantAnnotator
from contcheck.annotation.models import Annotated, Unmappable
from contcheck.corpus.record import EditRecord
from contcheck.taxonomy.kinds import ErrorKind, Operation

type Spans = tuple[int, int, int, int, str]


class FakeErrant:
    """Returns fixed ERRANT edits per noisy sentence, and records the docs it was given."""

    def __init__(self, edits: Mapping[str, list[Spans]]) -> None:
        self._edits = edits
        self.seen: list[tuple[str, str]] = []

    def annotate(self, orig: Doc, cor: Doc) -> list[Edit]:
        self.seen.append((orig.text, cor.text))
        return [
            Edit(orig, cor, [o_start, o_end, c_start, c_end], code)
            for o_start, o_end, c_start, c_end, code in self._edits.get(orig.text, [])
        ]


def annotator(
    edits: Mapping[str, list[Spans]], words: frozenset[str] = frozenset()
) -> ErrantAnnotator:
    # A blank pipeline builds docs without tagging, so this runs without the model.
    return ErrantAnnotator(spacy.blank("en"), FakeErrant(edits), words)


def test_errant_edits_map_onto_noisy_word_indices() -> None:
    subject = annotator({"I go home ": [(1, 2, 1, 2, "R:VERB:TENSE")]})

    result = subject.annotate(["I", "go", "home"], ["I", "went", "home"])

    assert result == Annotated(
        (
            EditRecord(
                start=1, end=2, kind=ErrorKind.VERB_TENSE, op=Operation.REPLACEMENT, orig=("went",)
            ),
        )
    )


def test_spelling_uses_the_word_list_to_split_real_words() -> None:
    edits = {"then ": [(0, 1, 0, 1, "R:SPELL")], "recieve ": [(0, 1, 0, 1, "R:SPELL")]}
    subject = annotator(edits, frozenset({"then", "than"}))

    real_word, misspelling = subject.annotate_many(
        [(["then"], ["than"]), (["recieve"], ["receive"])]
    )

    assert isinstance(real_word, Annotated)
    assert isinstance(misspelling, Annotated)
    assert real_word.edits[0].kind is ErrorKind.REAL_WORD
    assert misspelling.edits[0].kind is ErrorKind.SPELLING


def test_dropped_codes_reject_the_pair() -> None:
    subject = annotator({"a lot ": [(0, 2, 0, 1, "R:OTHER")]})

    result = subject.annotate(["a", "lot"], ["many"])

    assert isinstance(result, Unmappable)
    assert result.code == "R:OTHER"


def test_pairs_are_annotated_in_order() -> None:
    fake = FakeErrant({})
    subject = ErrantAnnotator(spacy.blank("en"), fake, frozenset())

    results = list(subject.annotate_many([(["a"], ["a"]), (["b", "c"], ["b", "c"])]))

    assert results == [Annotated(()), Annotated(())]
    assert fake.seen == [("a ", "a "), ("b c ", "b c ")]


@pytest.mark.parametrize("word", ["", "two words", "tab\there"])
def test_words_with_whitespace_are_rejected(word: str) -> None:
    with pytest.raises(ValueError, match="analyzer word"):
        annotator({}).annotate([word], ["fine"])
