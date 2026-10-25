"""``Inflector`` over ``lemminflect``."""

from lemminflect import getInflection


class LemmInflector:
    """Inflects lemmas with lemminflect's tables, guessing forms for unknown words."""

    def inflect(self, lemma: str, tag: str) -> tuple[str, ...]:
        forms: object = getInflection(lemma, tag=tag)
        if not isinstance(forms, tuple):
            return ()
        return tuple(form for form in forms if isinstance(form, str))
