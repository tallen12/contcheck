"""What a corruptor sees of the paragraph it corrupts."""

from dataclasses import dataclass, field

from contcheck.analysis.models import TenseHistogram, Token


@dataclass(frozen=True, slots=True)
class CorruptionContext:
    """One analyzed clean paragraph plus the tense profile of its whole document.

    ``words``, ``tags`` and ``lemmas`` are built once here: corruptors index them inside
    per-token loops, so rebuilding them on each access made a scan quadratic.
    """

    tokens: tuple[Token, ...]
    tense: TenseHistogram
    words: tuple[str, ...] = field(init=False, repr=False, compare=False)
    tags: tuple[str, ...] = field(init=False, repr=False, compare=False)
    lemmas: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Frozen dataclasses only allow setting derived fields through object.__setattr__.
        object.__setattr__(self, "words", tuple(token.text for token in self.tokens))
        object.__setattr__(self, "tags", tuple(token.tag for token in self.tokens))
        object.__setattr__(self, "lemmas", tuple(token.lemma for token in self.tokens))
