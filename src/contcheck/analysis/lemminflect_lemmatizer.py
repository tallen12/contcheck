"""``Lemmatizer`` over ``lemminflect``, with spaCy's stop words as closed-class words."""

from lemminflect import getAllLemmas
from spacy.lang.en.stop_words import STOP_WORDS


class LemmInflectLemmatizer:
    """Looks words up in lemminflect's tables, without guessing for unknown words.

    lemminflect covers open-class words only, so stop words (the, of, than) count as
    their own lemma. Names, abbreviations and other unknown forms have no lemma.
    """

    def lemmas(self, word: str) -> frozenset[str]:
        lower = word.lower()
        table: object = getAllLemmas(lower)
        found: set[str] = set()
        if isinstance(table, dict):
            for forms in table.values():
                if isinstance(forms, tuple):
                    found.update(lemma for lemma in forms if isinstance(lemma, str))
        if not found and lower in STOP_WORDS:
            found.add(lower)
        return frozenset(found)
