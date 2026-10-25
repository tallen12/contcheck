"""Spelling: turn a known word into a typo that is not a word.

Two families of typo: slips of the keyboard (adjacent keys, transposed, dropped or
doubled letters), and misspellings that follow sound and word structure, which is how
people misspell words they do not know how to spell (``occured``, ``seperate``,
``truely``). The second family matters because such misspellings split into
normal-looking subwords, so a model trained only on keyboard slips misses them.
"""

from collections.abc import Callable

from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.spelling.protocols import Dictionary
from contcheck.taxonomy.kinds import ErrorKind, Operation

MIN_LENGTH = 4
ATTEMPTS = 8
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
    ("ent", "ant"),
    ("ant", "ent"),
    ("cly", "cally"),
)
VOWELS = frozenset("aeiou")
# Vowels in unstressed syllables sound alike, so writers swap them (separate, seperate).
UNSTRESSED_VOWEL_SWAPS = {"a": "ei", "e": "ai", "i": "ae"}
SILENT_E_SUFFIXES = ("ment", "ness", "ful", "ly")


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


def _consonant(char: str) -> bool:
    return char.isalpha() and char.lower() not in VOWELS


def single_for_double(word: str, rng: Rng) -> str | None:
    """Write a doubled consonant once (occurred, occured; tomorrow, tomorow)."""
    sites = [
        i
        for i in range(1, len(word) - 1)
        if _consonant(word[i]) and word[i].lower() == word[i + 1].lower()
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return word[:i] + word[i + 1 :]


def double_for_single(word: str, rng: Rng) -> str | None:
    """Double a single consonant between vowels (necessary, neccessary)."""
    lower = word.lower()
    sites = [
        i
        for i in range(1, len(word) - 1)
        if _consonant(lower[i]) and lower[i - 1] in VOWELS and lower[i + 1] in VOWELS
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return word[: i + 1] + word[i] + word[i + 1 :]


def swap_unstressed_vowel(word: str, rng: Rng) -> str | None:
    """Swap a lone inner vowel for one that sounds alike (definitely, definately).

    The word's first vowel is left alone: it is usually stressed, and stressed vowels
    are rarely misspelled.
    """
    lower = word.lower()
    first_vowel = next((i for i, char in enumerate(lower) if char in VOWELS), len(lower))
    sites = [
        i
        for i in range(first_vowel + 1, len(word) - 1)
        if lower[i] in UNSTRESSED_VOWEL_SWAPS
        and lower[i - 1] not in VOWELS
        and lower[i + 1] not in VOWELS
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return word[:i] + rng.choice(UNSTRESSED_VOWEL_SWAPS[lower[i]]) + word[i + 1 :]


def silent_e(word: str, rng: Rng) -> str | None:
    """Add or drop a silent e before a suffix (truly, truely; argument, arguement)."""
    lower = word.lower()
    options: list[str] = []
    for suffix in SILENT_E_SUFFIXES:
        stem = len(lower) - len(suffix)
        if stem < 3 or not lower.endswith(suffix):
            continue
        if lower[stem - 1] == "e":
            options.append(word[: stem - 1] + word[stem:])
        else:
            options.append(word[:stem] + "e" + word[stem:])
    return rng.choice(options) if options else None


def transpose_vowels(word: str, rng: Rng) -> str | None:
    """Swap two different neighbouring vowels (because, becuase)."""
    lower = word.lower()
    sites = [
        i
        for i in range(1, len(word) - 1)
        if lower[i] in VOWELS and lower[i + 1] in VOWELS and lower[i] != lower[i + 1]
    ]
    if not sites:
        return None
    i = rng.choice(sites)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


KEYBOARD_TYPOS: tuple[Typo, ...] = (
    substitute_adjacent_key,
    transpose,
    delete_letter,
    double_letter,
)
REALISTIC_TYPOS: tuple[Typo, ...] = (
    phonetic_swap,
    single_for_double,
    double_for_single,
    swap_unstressed_vowel,
    silent_e,
    transpose_vowels,
)
# Realistic misspellings are listed twice, so about three in four attempts use one;
# they also fail to apply more often, which brings the produced mix closer to even.
TYPOS: tuple[Typo, ...] = KEYBOARD_TYPOS + REALISTIC_TYPOS + REALISTIC_TYPOS


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
        if not sites:
            return None
        for _ in range(ATTEMPTS):
            i = rng.choice(sites)
            word = context.words[i]
            typo = rng.choice(TYPOS)(word, rng)
            if (
                typo is not None
                and typo.lower() != word.lower()
                and not self._dictionary.contains(typo)
            ):
                return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (word,), (typo,))
        return None
