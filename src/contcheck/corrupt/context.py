"""What a corruptor sees of the paragraph it corrupts."""

from dataclasses import dataclass

from contcheck.analysis.models import TenseHistogram, Token


@dataclass(frozen=True, slots=True)
class CorruptionContext:
    """One analyzed clean paragraph plus the tense profile of its whole document."""

    tokens: tuple[Token, ...]
    tense: TenseHistogram

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(token.text for token in self.tokens)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(token.tag for token in self.tokens)

    @property
    def lemmas(self) -> tuple[str, ...]:
        return tuple(token.lemma for token in self.tokens)
