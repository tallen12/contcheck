"""Screening out paragraphs that are not clean prose before they become training text.

An error already in the source becomes a correct-looking training example, so a
paragraph with misspellings, markup, list syntax, shouting or run-on sentences is
dropped rather than kept.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import assert_never

from contcheck.corpus.models import Clean, Rejected, RejectReason, Verdict
from contcheck.corpus.protocols import ParagraphScreen
from contcheck.spelling.protocols import Dictionary

MARKUP = re.compile(
    r"</?[a-zA-Z][^>]*>|\{\{|\}\}|\[\[|\]\]|https?://|www\.|&#?\w+;|[|\\{}<>]|={2,}"
    r"|(?<!\w)_\w|\w_(?!\w)|\*"
)
LIST_ITEM = re.compile(r"^(?:[-*•·▪◦‣\u2013]|\d{1,3}[.)]|[a-z][.)]|\(\w{1,3}\))\s|\s[•·▪]\s")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”\u2019)\]]*\s+")
MIN_CHECKED_LETTERS = 3
"""Shorter words are mostly abbreviations and units (``km``, ``hr``), not misspellings."""

AMERICAN_TO_BRITISH = (
    (re.compile(r"or(s|ed|ing|ite|ites|hood|hoods|ful|less|ly)?$"), r"our\1"),
    (re.compile(r"([^aeiou])er(s)?$"), r"\1re\2"),
    (re.compile(r"([aeiou])l(ed|ing|er|ers)$"), r"\1ll\2"),
    (re.compile(r"ll(ful|ment|ments)?$"), r"l\1"),
    (re.compile(r"ic(ing|ed)$"), r"is\1"),
    (re.compile(r"num$"), r"nium"),
    (re.compile(r"sulf"), r"sulph"),
    (re.compile(r"ense(s)?$"), r"ence\1"),
    (re.compile(r"og(s)?$"), r"ogue\1"),
    (re.compile(r"^sk(ep)"), r"sc\1"),
)
"""American spellings missing from the bundled list, rewritten to the British form it has."""

EDGE_PUNCTUATION = "\"'“”\u2018\u2019()[]{}.,;:!?…—\u2013-"


@dataclass(frozen=True, slots=True)
class CleanlinessConfig:
    """Thresholds of the cleanliness checks.

    ``max_non_word_share`` is the largest share of lowercase alphabetic words that may
    be missing from the dictionary. Capitalized words are not checked, so names and
    sentence starts never count against a paragraph.
    """

    max_non_word_share: float = 0.03
    max_caps_share: float = 0.25
    max_sentence_words: int = 70

    def __post_init__(self) -> None:
        if not 0 <= self.max_non_word_share <= 1 or not 0 <= self.max_caps_share <= 1:
            raise ValueError("shares must be between 0 and 1")
        if self.max_sentence_words < 1:
            raise ValueError("max_sentence_words must be positive")


class CleanlinessFilter:
    """Checks markup, list syntax, capitals, sentence length and dictionary coverage."""

    def __init__(self, dictionary: Dictionary, config: CleanlinessConfig) -> None:
        self._dictionary = dictionary
        self._config = config

    def screen(self, paragraph: str) -> Verdict:
        if LIST_ITEM.search(paragraph):
            return Rejected(RejectReason.LIST)
        if MARKUP.search(paragraph):
            return Rejected(RejectReason.MARKUP)
        words = [word.strip(EDGE_PUNCTUATION) for word in paragraph.split()]
        if caps_share(words) > self._config.max_caps_share:
            return Rejected(RejectReason.ALL_CAPS)
        if longest_sentence(paragraph) > self._config.max_sentence_words:
            return Rejected(RejectReason.LONG_SENTENCE)
        if self.non_word_share(words) > self._config.max_non_word_share:
            return Rejected(RejectReason.NON_WORDS)
        return Clean()

    def non_word_share(self, words: list[str]) -> float:
        """Share of lowercase alphabetic ``words`` the dictionary does not know.

        Words shorter than ``MIN_CHECKED_LETTERS`` are not checked, and an American
        spelling counts as known when its British form is.
        """
        checked = [word for word in words if is_checked(word)]
        if not checked:
            return 0.0
        unknown = sum(1 for word in checked if not self.knows(word))
        return unknown / len(checked)

    def knows(self, word: str) -> bool:
        """Whether the dictionary has ``word`` or the British spelling of it."""
        return any(self._dictionary.contains(form) for form in spelling_forms(word))


def is_checked(word: str) -> bool:
    """Whether ``word`` is a lowercase ASCII word long enough to look up."""
    return len(word) >= MIN_CHECKED_LETTERS and word.isascii() and word.isalpha() and word.islower()


def spelling_forms(word: str) -> list[str]:
    """``word`` followed by its British forms under each ``AMERICAN_TO_BRITISH`` rule."""
    forms = [word]
    for pattern, replacement in AMERICAN_TO_BRITISH:
        form = pattern.sub(replacement, word)
        if form != word:
            forms.append(form)
    return forms


def caps_share(words: list[str]) -> float:
    """Share of alphabetic words of two or more letters written in capitals."""
    lettered = [word for word in words if len(word) > 1 and word.isalpha()]
    if not lettered:
        return 0.0
    return sum(1 for word in lettered if word.isupper()) / len(lettered)


def longest_sentence(paragraph: str) -> int:
    """Word count of the longest sentence, splitting after ``.``, ``!`` or ``?``."""
    return max(len(sentence.split()) for sentence in SENTENCE_END.split(paragraph))


class AcceptAll:
    """A screen that keeps every paragraph, for sources that need no cleanliness check."""

    def screen(self, paragraph: str) -> Verdict:
        return Clean()


ACCEPT_ALL = AcceptAll()


class CountingScreen:
    """Wraps a screen and counts its verdicts, for reporting what a fetch dropped."""

    def __init__(self, screen: ParagraphScreen) -> None:
        self._screen = screen
        self.screened = 0
        self.rejected: Counter[RejectReason] = Counter()

    def screen(self, paragraph: str) -> Verdict:
        verdict = self._screen.screen(paragraph)
        self.screened += 1
        match verdict:
            case Clean():
                pass
            case Rejected(reason=reason):
                self.rejected[reason] += 1
            case _:
                assert_never(verdict)
        return verdict

    def summary(self) -> str:
        """``screened N, rejected M (P%)`` and the count for each reason."""
        total = sum(self.rejected.values())
        share = total / self.screened if self.screened else 0.0
        reasons = ", ".join(f"{reason.value} {self.rejected[reason]}" for reason in RejectReason)
        return f"screened {self.screened}, rejected {total} ({share:.1%}): {reasons}"
