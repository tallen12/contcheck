"""Value types produced by text analysis."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    """One word or punctuation token with its Penn Treebank tag and lemma."""

    text: str
    tag: str
    lemma: str
    space_after: bool
    sentence_start: bool
    finite: bool


@dataclass(frozen=True, slots=True)
class AnalyzedText:
    """The tokens of one paragraph, in order."""

    tokens: tuple[Token, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(token.text for token in self.tokens)

    @property
    def text(self) -> str:
        """The text rebuilt from tokens and their trailing spaces."""
        return join_words(self.words, tuple(token.space_after for token in self.tokens))


@dataclass(frozen=True, slots=True)
class TenseHistogram:
    """Share of finite verbs per tense; the four shares sum to 1, or are all 0."""

    past: float
    present: float
    modal: float
    other: float


@dataclass(frozen=True, slots=True)
class DocStats:
    """Simple document statistics that separate dialogue, narration and exposition."""

    sentence_count: int
    mean_sentence_length: float
    first_person_share: float
    quoted_share: float


def join_words(words: tuple[str, ...], spaces: tuple[bool, ...]) -> str:
    """Join words, following each one with a space where ``spaces`` says so."""
    if len(words) != len(spaces):
        raise ValueError(f"{len(words)} words but {len(spaces)} space flags")
    return "".join(word + (" " if space else "") for word, space in zip(words, spaces, strict=True))
