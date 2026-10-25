"""Value types for model input: subword encodings and fixed-size windows."""

from dataclasses import dataclass

IGNORE_INDEX = -100
"""Label of tokens the loss skips: special tokens and every subword after a word's first."""


@dataclass(frozen=True, slots=True)
class Encoding:
    """Subword ids for pre-split words, without special tokens.

    ``word_ids[i]`` is the index of the word token ``i`` belongs to.
    """

    input_ids: tuple[int, ...]
    word_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.word_ids):
            raise ValueError("one word id per token is needed")


@dataclass(frozen=True, slots=True)
class Window:
    """One model input: at most ``window`` tokens of a paragraph, with special tokens.

    ``word_ids`` is ``None`` for special tokens. ``labels`` holds a label id on each
    word's first subword and :data:`IGNORE_INDEX` elsewhere. ``start`` is the position of
    the first content token in the paragraph's subwords, so windows can be merged back.
    """

    doc_id: str
    input_ids: tuple[int, ...]
    word_ids: tuple[int | None, ...]
    labels: tuple[int, ...]
    start: int

    def __post_init__(self) -> None:
        if not len(self.input_ids) == len(self.word_ids) == len(self.labels):
            raise ValueError("input ids, word ids and labels must have one entry per token")
