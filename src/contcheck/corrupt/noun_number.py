"""NounNumber: make one common noun singular or plural against what its context marks."""

from enum import Enum, auto

from contcheck.analysis.protocols import Inflector
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.protocols import Rng
from contcheck.taxonomy.kinds import ErrorKind, Operation

SINGULAR_MARKERS = frozenset(
    {"a", "an", "this", "that", "every", "each", "another", "one", "either", "neither"}
)
PLURAL_MARKERS = frozenset(
    {"these", "those", "many", "several", "few", "both", "various", "numerous"}
    | {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"}
    | {"twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million"}
)
MARKERS = SINGULAR_MARKERS | PLURAL_MARKERS
MARKER_TAGS = frozenset({"DT", "PDT", "CD", "JJ"})
# Determiners that leave the number open; then only the verb right after the noun marks it.
NEUTRAL_DETERMINERS = frozenset({"the", "my", "your", "his", "her", "its", "our", "their"})
MODIFIER_TAGS = frozenset({"JJ", "JJR", "JJS"})
# A noun followed by one of these is not the head of its phrase (``book store``,
# ``children's``), so its number is not what the marker constrains.
NON_HEAD_NEXT_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", "POS", "HYPH"})
SINGULAR_VERBS = frozenset({"VBZ"})
PLURAL_VERBS = frozenset({"VBP"})
# Plural-only nouns whose singular is another word or no word (``clothe``, ``glass``).
PLURAL_ONLY = frozenset(
    {"clothes", "scissors", "trousers", "pants", "jeans", "glasses", "thanks", "goods"}
    | {"arms", "remains", "savings", "belongings", "surroundings", "outskirts", "cattle"}
    | {"police", "people", "headquarters", "premises", "manners", "earnings", "odds"}
)


class Number(Enum):
    """The grammatical number a noun's context demands."""

    SINGULAR = auto()
    PLURAL = auto()


def _is_plural_numeral(word: str) -> bool:
    # Four digits are usually a year (``1990 elections``), which marks no number.
    return word.isdigit() and word != "1" and len(word) != 4


def marked_number(context: CorruptionContext, noun: int) -> Number | None:
    """The number the words around ``noun`` demand, or ``None`` if they leave it open.

    A marker determiner or numeral may be separated from the noun by adjectives. With a
    number-neutral determiner (``the``, ``my``), a finite present verb or ``was``/``were``
    right after the noun marks it instead, unless the phrase follows ``of``, where the
    noun is unlikely to be the subject (``the list of the items is``).
    """
    start = noun - 1
    # spaCy tags ``many``, ``several`` and ``few`` JJ, so markers stop the walk too.
    while (
        start >= 0
        and context.tags[start] in MODIFIER_TAGS
        and context.words[start].lower() not in MARKERS
    ):
        start -= 1
    if start < 0:
        return None
    word = context.words[start].lower()
    if context.tags[start] in MARKER_TAGS:
        if word in SINGULAR_MARKERS:
            return Number.SINGULAR
        if word in PLURAL_MARKERS or _is_plural_numeral(word):
            return Number.PLURAL
    if word not in NEUTRAL_DETERMINERS or noun + 1 >= len(context.tokens):
        return None
    if start > 0 and context.words[start - 1].lower() == "of":
        return None
    following = context.tokens[noun + 1]
    if not following.finite:
        return None
    if following.tag in SINGULAR_VERBS or following.text.lower() == "was":
        return Number.SINGULAR
    if following.tag in PLURAL_VERBS or following.text.lower() == "were":
        return Number.PLURAL
    return None


class NounNumberCorruptor:
    """Flips a common noun between singular and plural where a determiner, a numeral or
    the verb after it marks the number: ``this book`` to ``this books``, ``three
    children`` to ``three child``, ``the dog barks`` to ``the dogs barks``.

    Proper nouns, capitalized words and non-head nouns are skipped, as are nouns whose
    two forms coincide (``sheep``) and plural-only nouns. Mass nouns are kept: after a
    marker their plural is wrong too (``this informations``).
    """

    kind = ErrorKind.NOUN_NUMBER

    def __init__(self, inflector: Inflector) -> None:
        self._inflector = inflector

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        edits = [edit for i in range(len(context.tokens)) if (edit := self._flip(context, i))]
        return rng.choice(edits) if edits else None

    def _flip(self, context: CorruptionContext, i: int) -> Edit | None:
        token = context.tokens[i]
        if token.tag not in {"NN", "NNS"} or not (token.text.isalpha() and token.text.islower()):
            return None
        if i + 1 < len(context.tokens) and context.tags[i + 1] in NON_HEAD_NEXT_TAGS:
            return None
        if token.text in PLURAL_ONLY:
            return None
        match token.tag, marked_number(context, i):
            case "NN", Number.SINGULAR:
                new = self._other_form(token.lemma, token.text, to="NNS", keep="NN")
            case "NNS", Number.PLURAL:
                new = self._other_form(token.lemma, token.text, to="NN", keep="NNS")
            case _:
                new = None
        if new is None:
            return None
        return Edit(i, i + 1, self.kind, Operation.REPLACEMENT, (token.text,), (new,))

    def _other_form(self, lemma: str, word: str, *, to: str, keep: str) -> str | None:
        """The most common ``to`` form of ``lemma``, unless it is the most common ``keep`` form
        or the word itself, which means the noun is invariant (``sheep``) and the swap is no error.

        Only first forms count: lemminflect also lists rare forms, such as ``dog`` as a
        plural of *dog*.
        """
        forms = self._inflector.inflect(lemma, to)
        kept = self._inflector.inflect(lemma, keep)
        if not forms:
            return None
        new = forms[0].lower()
        return None if new == word or (kept and new == kept[0].lower()) else new
