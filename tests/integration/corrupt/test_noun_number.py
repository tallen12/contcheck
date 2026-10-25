import random

import pytest

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.models import TenseHistogram
from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.noun_number import NounNumberCorruptor

TEXT = (
    "This book is heavy, and three children found several mice in the garden. "
    "The committee meets every morning. Two sheep and London stayed behind."
)


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


def test_real_sentences_get_wrong_noun_number(analyzer: SpacyAnalyzer) -> None:
    context = CorruptionContext(analyzer.analyze(TEXT).tokens, TenseHistogram(0.5, 0.5, 0, 0))
    corruptor = NounNumberCorruptor(LemmInflector())
    found = {
        (edit.orig[0], edit.new[0])
        for seed in range(200)
        if (edit := corruptor.corrupt(context, random.Random(seed)))
    }

    assert {("book", "books"), ("children", "child"), ("mice", "mouse")} <= found
    assert {("committee", "committees"), ("morning", "mornings")} <= found
    assert not any(orig in {"sheep", "London", "garden"} for orig, _ in found)
