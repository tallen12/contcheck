"""Value types for masked-LM queries and predictions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MaskedSlot:
    """One word to mask in a run of words, plus words to score in its place.

    ``spaces`` says, like ``Token.space_after``, whether a space follows each word, so the
    text the model sees has the paragraph's real spacing.
    """

    words: tuple[str, ...]
    spaces: tuple[bool, ...]
    position: int
    candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.words) != len(self.spaces):
            raise ValueError(f"{len(self.words)} words but {len(self.spaces)} space flags")
        if not 0 <= self.position < len(self.words):
            raise ValueError(f"position {self.position} is outside {len(self.words)} words")


@dataclass(frozen=True, slots=True)
class WordScore:
    """A whole word and the model's probability for it at the masked position."""

    word: str
    probability: float


@dataclass(frozen=True, slots=True)
class SlotPrediction:
    """What the model expects at one masked position.

    ``top`` holds the most probable whole words, most probable first. ``scored`` holds the
    slot's candidates in order, leaving out any the model cannot score because they are
    not a single model token.
    """

    top: tuple[WordScore, ...]
    scored: tuple[WordScore, ...]

    def probability(self, word: str) -> float | None:
        """The scored probability of ``word``, or ``None`` if it was not scored."""
        return next((score.probability for score in self.scored if score.word == word), None)
