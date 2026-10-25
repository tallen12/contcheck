"""``TextAnalyzer`` over a spaCy pipeline such as ``en_core_web_sm``."""

from collections.abc import Iterable, Iterator

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from contcheck.analysis.models import AnalyzedText, Token

DEFAULT_MODEL = "en_core_web_sm"
BATCH_SIZE = 64


class SpacyAnalyzer:
    """Tags, lemmatizes and sentence-splits text with a loaded spaCy pipeline."""

    def __init__(self, nlp: Language) -> None:
        self._nlp = nlp

    def analyze(self, text: str) -> AnalyzedText:
        return _convert(self._nlp(text))

    def analyze_many(self, texts: Iterable[str]) -> Iterator[AnalyzedText]:
        for doc in self._nlp.pipe(texts, batch_size=BATCH_SIZE):
            yield _convert(doc)


def load_spacy_analyzer(model: str = DEFAULT_MODEL) -> SpacyAnalyzer:
    """Load ``model`` with only the components the analyzer needs.

    The dependency parser and NER are left out for speed; the statistical sentence
    splitter (``senter``) replaces the parser's sentence boundaries.
    """
    nlp = spacy.load(model, exclude=["parser", "ner"])
    nlp.enable_pipe("senter")
    return SpacyAnalyzer(nlp)


def _convert(doc: Doc) -> AnalyzedText:
    return AnalyzedText(
        tuple(
            Token(
                text=token.text,
                tag=token.tag_,
                lemma=token.lemma_,
                space_after=bool(token.whitespace_),
                sentence_start=token.i == 0 or bool(token.is_sent_start),
                finite=token.tag_ == "MD" or "Fin" in token.morph.get("VerbForm", []),
            )
            for token in doc
            if not token.is_space
        )
    )
