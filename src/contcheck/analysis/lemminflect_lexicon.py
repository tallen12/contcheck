"""Lexicons over ``lemminflect``'s dictionary tables, never its out-of-vocabulary rules.

``LemmLexicon`` answers whether a word is a listed adjective or adverb (the
``WordClassLexicon`` protocol, for AdjectiveAdverb); ``LemmInflectLexicon`` finds a
tagged word's lemmas (for word-choice candidates).
"""

from functools import cache

from lemminflect import getAllLemmas, getInflection

UPOS_BY_TAG_PREFIX = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ", "RB": "ADV"}
"""Universal part of speech of the Penn tags lemminflect inflects, by their first letters."""


class LemmLexicon:
    """Looks words up in lemminflect's dictionary only, never its out-of-vocabulary rules,
    so made-up derivations such as *greenly* are not words."""

    def is_adjective(self, word: str) -> bool:
        return _listed(word, "ADJ")

    def is_adverb(self, word: str) -> bool:
        return _listed(word, "ADV")


def _listed(word: str, upos: str) -> bool:
    lemmas: object = getAllLemmas(word.lower(), upos=upos)
    return isinstance(lemmas, dict) and bool(lemmas)


class LemmInflectLexicon:
    """Finds lemmas in lemminflect's dictionary only, so unknown words have none."""

    def lemmas(self, word: str, tag: str) -> frozenset[str]:
        return _lemmas(word, tag)


@cache
def _lemmas(word: str, tag: str) -> frozenset[str]:
    found: set[str] = set()
    upos = UPOS_BY_TAG_PREFIX.get(tag[:2])
    candidates: object = None if upos is None else getAllLemmas(word, upos=upos)
    if not isinstance(candidates, dict):
        return frozenset(found)
    # getInflection inflects any lemma for any tag, so the part of speech is checked
    # above; unlike getAllInflections it also gives VBN forms equal to VBD.
    for lemma in candidates.get(upos, ()):
        forms: object = getInflection(lemma, tag=tag, inflect_oov=False)
        if isinstance(lemma, str) and isinstance(forms, tuple) and word in forms:
            found.add(lemma)
    return frozenset(found)
