"""Spelling: turn a known word into a typo that is not a word."""

from collections.abc import Callable

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.sampling import shuffled
from contcheck.spelling.protocols import Dictionary
from contcheck.taxonomy.kinds import ErrorKind, Operation

MIN_LENGTH = 4
PROPER_NOUN_TAGS = frozenset({"NNP", "NNPS"})

KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
PHONETIC_SWAPS = (
    ("ph", "f"),
    ("ie", "ei"),
    ("ei", "ie"),
    ("ance", "ence"),
    ("ence", "ance"),
    ("able", "ible"),
    ("ible", "able"),
    ("ary", "ery"),
    ("tion", "sion"),
    ("ou", "o"),
    ("ee", "ea"),
    ("ea", "ee"),
)


def keyboard_neighbours() -> dict[str, str]:
    """Keys next to each letter on a QWERTY keyboard, in the same row or the rows beside."""
    positions = {
        char: (row, col) for row, keys in enumerate(KEYBOARD_ROWS) for col, char in enumerate(keys)
    }
    return {
        char: "".join(
            other
            for other, (other_row, other_col) in positions.items()
            if other != char and abs(other_row - row) <= 1 and abs(other_col - col) <= 1
        )
        for char, (row, col) in positions.items()
    }


NEIGHBOURS = keyboard_neighbours()

type Typo = Callable[[str, Rng], str | None]


def _position(word: str, rng: Rng) -> int:
    # Typos rarely hit the first letter, and leaving it alone keeps the casing intact.
    return 1 + rng.randrange(len(word) - 1)


def substitute_adjacent_key(word: str, rng: Rng) -> str | None:
    """Replace one letter by a neighbouring key."""
    i = _position(word, rng)
    char = word[i]
    neighbours = NEIGHBOURS.get(char.lower(), "")
    if not neighbours:
        return None
    new = rng.choice(neighbours)
    return word[:i] + (new.upper() if char.isupper() else new) + word[i + 1 :]


def transpose(word: str, rng: Rng) -> str | None:
    """Swap two neighbouring letters."""
    i = _position(word, rng)
    if i + 1 >= len(word) or word[i] == word[i + 1]:
        return None
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def delete_letter(word: str, rng: Rng) -> str | None:
    """Drop one letter."""
    i = _position(word, rng)
    return word[:i] + word[i + 1 :]


def double_letter(word: str, rng: Rng) -> str | None:
    """Type one letter twice."""
    i = _position(word, rng)
    return word[: i + 1] + word[i] + word[i + 1 :]


def phonetic_swap(word: str, rng: Rng) -> str | None:
    """Swap a spelling for one that sounds alike, such as ph for f."""
    lower = word.lower()
    options = [(old, new) for old, new in PHONETIC_SWAPS if lower.find(old, 1) > 0]
    if not options:
        return None
    old, new = rng.choice(options)
    i = lower.find(old, 1)
    return word[:i] + new + word[i + len(old) :]


TYPOS: tuple[Typo, ...] = (
    substitute_adjacent_key,
    transpose,
    delete_letter,
    double_letter,
    phonetic_swap,
)


class SpellingCorruptor:
    """Misspells one known, alphabetic, non-name word of four or more letters.

    The result must not be a dictionary word; a typo that makes a real word is a
    RealWord error, not a Spelling error.
    """

    kind = ErrorKind.SPELLING

    def __init__(self, dictionary: Dictionary) -> None:
        self._dictionary = dictionary

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        sites = [
            i
            for i, token in enumerate(context.tokens)
            if len(token.text) >= MIN_LENGTH
            and token.text.isalpha()
            and token.tag not in PROPER_NOUN_TAGS
            and self._dictionary.contains(token.text)
        ]
        # Try every site and typo kind once, in random order, so ``None`` means no site
        # can be misspelled; the applier drops the kind for the paragraph on ``None``.
        for i, make_typo in shuffled([(i, typo) for i in sites for typo in TYPOS], rng):
            word = context.words[i]
            typo = make_typo(word, rng)
            if (
                typo is not None
                and typo.lower() != word.lower()
                and not self._dictionary.contains(typo)
            ):
                return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (typo,))
        return None
