import pytest

from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.annotation.errant_annotator import ErrantAnnotator, load_errant_annotator
from contcheck.annotation.models import Annotated, Unmappable
from contcheck.corpus.record import revert_edits
from contcheck.taxonomy.kinds import ErrorKind, Operation


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


@pytest.fixture(scope="module")
def annotator() -> ErrantAnnotator:
    return load_errant_annotator()


@pytest.mark.parametrize(
    ("noisy", "clean", "kind", "op"),
    [
        ("She recieved the letter.", "She received the letter.", ErrorKind.SPELLING, "R"),
        (
            "He is taller then his brother.",
            "He is taller than his brother.",
            ErrorKind.REAL_WORD,
            "R",
        ),
        (
            "Yesterday I go to the store.",
            "Yesterday I went to the store.",
            ErrorKind.VERB_TENSE,
            "R",
        ),
        (
            "She walk to school every day.",
            "She walks to school every day.",
            ErrorKind.SUBJECT_VERB_AGREEMENT,
            "R",
        ),
        ("I saw a elephant.", "I saw an elephant.", ErrorKind.DETERMINER, "R"),
        ("I saw elephant.", "I saw an elephant.", ErrorKind.DETERMINER, "M"),
        ("She has went home.", "She has gone home.", ErrorKind.VERB_FORM, "R"),
        ("I bought three apple.", "I bought three apples.", ErrorKind.NOUN_NUMBER, "R"),
        ("He runs very quick.", "He runs very quickly.", ErrorKind.ADJECTIVE_ADVERB, "R"),
        ("Him went home early.", "He went home early.", ErrorKind.PRONOUN_FORM, "R"),
        ("She is good in math.", "She is good at math.", ErrorKind.PREPOSITION, "R"),
        ("I walked the park.", "I walked to the park.", ErrorKind.PREPOSITION, "M"),
        (
            "Tom or Anna went to the party.",
            "Tom and Anna went to the party.",
            ErrorKind.CONJUNCTION,
            "R",
        ),
        (
            "I moved to paris last year.",
            "I moved to Paris last year.",
            ErrorKind.CAPITALIZATION,
            "R",
        ),
        ("I will see you to morrow.", "I will see you tomorrow.", ErrorKind.WORD_BOUNDARY, "R"),
    ],
)
def test_one_error_per_sentence_gets_its_kind(
    analyzer: SpacyAnalyzer,
    annotator: ErrantAnnotator,
    noisy: str,
    clean: str,
    kind: ErrorKind,
    op: str,
) -> None:
    noisy_words = analyzer.analyze(noisy).words
    clean_words = analyzer.analyze(clean).words

    result = annotator.annotate(noisy_words, clean_words)

    assert isinstance(result, Annotated)
    assert [(edit.kind, edit.op) for edit in result.edits] == [(kind, Operation(op))]
    assert revert_edits(noisy_words, result.edits) == clean_words


def test_several_errors_in_one_paragraph(
    analyzer: SpacyAnalyzer, annotator: ErrantAnnotator
) -> None:
    noisy = analyzer.analyze("Yesterday I go to the the store and buy a apple.").words
    clean = analyzer.analyze("Yesterday I went to the store and bought an apple.").words

    result = annotator.annotate(noisy, clean)

    assert isinstance(result, Annotated)
    assert [(e.start, e.end, e.kind, e.op) for e in result.edits] == [
        (2, 3, ErrorKind.VERB_TENSE, Operation.REPLACEMENT),
        (4, 5, ErrorKind.DETERMINER, Operation.UNNECESSARY),
        (8, 9, ErrorKind.VERB_TENSE, Operation.REPLACEMENT),
        (9, 10, ErrorKind.DETERMINER, Operation.REPLACEMENT),
    ]
    assert revert_edits(noisy, result.edits) == clean


def test_untypeable_rewrites_are_unmappable(
    analyzer: SpacyAnalyzer, annotator: ErrantAnnotator
) -> None:
    noisy = analyzer.analyze("There is a lot of reasons.").words
    clean = analyzer.analyze("There are many reasons.").words

    result = annotator.annotate(noisy, clean)

    assert isinstance(result, Unmappable)
