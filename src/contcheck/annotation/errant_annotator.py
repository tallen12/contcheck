"""``Annotator`` over ERRANT with a full spaCy pipeline.

Alignment with record words is by construction: ERRANT never tokenizes. Each side is
built as a spaCy ``Doc`` from exactly the words given (which come from the analyzer that
split the record), then tagged and parsed, so ERRANT's token indices are our word
indices. :func:`~contcheck.annotation.convert.to_annotation` then checks that the mapped
edits revert the noisy words to the clean ones and raises otherwise.
"""

from collections.abc import Collection, Iterable, Iterator, Sequence
from itertools import batched

import errant
import spacy
from errant.edit import Edit
from errant.en import classifier
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from contcheck.annotation.convert import to_annotation
from contcheck.annotation.models import Annotation, TypedEdit
from contcheck.annotation.protocols import EditExtractor
from contcheck.taxonomy.errant_map import EditEvidence

DEFAULT_MODEL = "en_core_web_sm"
BATCH_SIZE = 64


class ErrantAnnotator:
    """Runs ERRANT on noisy to clean words and maps its edits onto our taxonomy."""

    def __init__(self, nlp: Language, pipeline: EditExtractor, word_list: Collection[str]) -> None:
        """``nlp`` tags and parses for ``pipeline``; ``word_list`` is its spelling dictionary."""
        self._nlp = nlp
        self._pipeline = pipeline
        self._word_list = word_list

    def annotate(self, noisy: Sequence[str], clean: Sequence[str]) -> Annotation:
        return next(self.annotate_many([(noisy, clean)]))

    def annotate_many(
        self, pairs: Iterable[tuple[Sequence[str], Sequence[str]]]
    ) -> Iterator[Annotation]:
        for edits, noisy, clean in self.typed_edits_many(pairs):
            yield to_annotation(edits, noisy, clean)

    def typed_edits_many(
        self, pairs: Iterable[tuple[Sequence[str], Sequence[str]]]
    ) -> Iterator[tuple[list[TypedEdit], tuple[str, ...], tuple[str, ...]]]:
        """ERRANT's edits per pair before mapping, with the pair's noisy and clean words."""
        # One stream of noisy, clean, noisy, clean... docs lets spaCy batch across pairs.
        docs = (_doc(self._nlp.vocab, words) for pair in pairs for words in pair)
        stream = self._nlp.pipe(docs, batch_size=BATCH_SIZE)
        for noisy_doc, clean_doc in batched(stream, 2, strict=True):
            edits = self._pipeline.annotate(noisy_doc, clean_doc)
            yield [self._typed(edit) for edit in edits], _words(noisy_doc), _words(clean_doc)

    def _typed(self, edit: Edit) -> TypedEdit:
        noisy_tokens = _span(edit.o_toks)
        clean_tokens = _span(edit.c_toks)
        noisy = tuple(token.text for token in noisy_tokens)
        return TypedEdit(
            noisy_start=int(edit.o_start),
            noisy_end=int(edit.o_end),
            clean_start=int(edit.c_start),
            clean_end=int(edit.c_end),
            code=str(edit.type),
            evidence=EditEvidence(
                noisy=noisy,
                clean=tuple(token.text for token in clean_tokens),
                noisy_tags=tuple(token.tag_ for token in noisy_tokens),
                clean_tags=tuple(token.tag_ for token in clean_tokens),
                noisy_deps=tuple(token.dep_ for token in noisy_tokens),
                clean_deps=tuple(token.dep_ for token in clean_tokens),
                noisy_is_word=len(noisy) == 1
                and (noisy[0] in self._word_list or noisy[0].lower() in self._word_list),
            ),
        )


def load_errant_annotator(model: str = DEFAULT_MODEL) -> ErrantAnnotator:
    """Load ``model`` with the tagger and parser ERRANT's classifier needs.

    Tokenization is not used (see the module docstring), and tagging is the same component
    the analyzer runs, so tags match record analysis. ``errant.load`` also points its
    classifier module at this pipeline, a global ERRANT keeps; load one annotator per
    process.
    """
    nlp = spacy.load(model, exclude=["ner"])
    return ErrantAnnotator(nlp, errant.load("en", nlp), classifier.spell)


def _doc(vocab: Vocab, words: Sequence[str]) -> Doc:
    for word in words:
        if not word or any(character.isspace() for character in word):
            raise ValueError(f"{word!r} cannot be an analyzer word")
    return Doc(vocab, words=list(words))


def _words(doc: Doc) -> tuple[str, ...]:
    return tuple(token.text for token in doc)


def _span(tokens: object) -> Span:
    # ERRANT is untyped; its edit sides are slices of the parsed docs.
    if not isinstance(tokens, Span):
        raise TypeError(f"expected an ERRANT edit side as a spaCy Span, got {type(tokens)}")
    return tokens
