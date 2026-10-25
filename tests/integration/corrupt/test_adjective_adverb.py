import random

import pytest

from contcheck.analysis.lemminflect_lexicon import LemmLexicon
from contcheck.analysis.models import TenseHistogram
from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.corrupt.adjective_adverb import AdjectiveAdverbCorruptor
from contcheck.corrupt.context import CorruptionContext

TEXT = (
    "She spoke carefully to a careful driver. He did well on the test, though it was "
    "really cold. The room was quiet. She ran fast and hardly noticed the kind man."
)


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


def test_real_sentences_get_adjective_adverb_swaps(analyzer: SpacyAnalyzer) -> None:
    context = CorruptionContext(analyzer.analyze(TEXT).tokens, TenseHistogram(0.5, 0.5, 0, 0))
    corruptor = AdjectiveAdverbCorruptor(LemmLexicon())
    found = {
        (edit.orig[0], edit.new[0])
        for seed in range(200)
        if (edit := corruptor.corrupt(context, random.Random(seed)))
    }

    assert {("carefully", "careful"), ("careful", "carefully"), ("well", "good")} <= found
    assert {("really", "real"), ("quiet", "quietly")} <= found
    assert not any(orig in {"fast", "hardly", "kind"} for orig, _ in found)
