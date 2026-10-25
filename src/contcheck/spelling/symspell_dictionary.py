"""``Dictionary`` over ``symspellpy`` and its bundled English frequency list."""

from collections.abc import Mapping
from importlib.resources import as_file, files

from symspellpy import SymSpell, Verbosity

from contcheck.spelling.models import Suggestion

BUNDLED_DICTIONARY = "frequency_dictionary_en_82_765.txt"
MAX_DISTANCE = 2


class SymSpellDictionary:
    """Looks words up in a loaded ``SymSpell`` index, ignoring case."""

    def __init__(self, symspell: SymSpell) -> None:
        self._symspell = symspell

    def contains(self, word: str) -> bool:
        return word.lower() in self._symspell.words

    def suggestions(self, word: str, *, max_distance: int = MAX_DISTANCE) -> list[Suggestion]:
        items = self._symspell.lookup(word.lower(), Verbosity.ALL, max_edit_distance=max_distance)
        return [Suggestion(item.term, item.distance, item.count) for item in items]

    def frequencies(self) -> Mapping[str, int]:
        """Every word in the index with its corpus count."""
        return self._symspell.words


def load_symspell_dictionary() -> SymSpellDictionary:
    """Build the index from symspellpy's bundled frequency list (about a second)."""
    symspell = SymSpell(max_dictionary_edit_distance=MAX_DISTANCE)
    with as_file(files("symspellpy") / BUNDLED_DICTIONARY) as path:
        if not symspell.load_dictionary(path, term_index=0, count_index=1):
            raise FileNotFoundError(f"could not load {path}")
    return SymSpellDictionary(symspell)
