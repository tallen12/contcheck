"""``WordClassLexicon`` over ``lemminflect``'s lemma tables."""

from lemminflect import getAllLemmas


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
