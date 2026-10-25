import random

import pytest

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.models import TenseHistogram
from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.verb_form import VerbFormCorruptor

TEXT = (
    "She has gone home, and the boy was taken to school. They are going to visit us, "
    "but I can swim. He did not know that she saw the ship."
)


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


def test_real_sentences_get_wrong_verb_forms(analyzer: SpacyAnalyzer) -> None:
    context = CorruptionContext(analyzer.analyze(TEXT).tokens, TenseHistogram(0.5, 0.5, 0, 0))
    corruptor = VerbFormCorruptor(LemmInflector())
    found = {
        (edit.orig[0], edit.new[0])
        for seed in range(200)
        if (edit := corruptor.corrupt(context, random.Random(seed)))
    }

    assert {("gone", "went"), ("taken", "took"), ("going", "go"), ("visit", "visiting")} <= found
    assert {("swim", "swimming"), ("know", "knew"), ("saw", "seen")} <= found
    assert not any(orig in {"has", "was", "are", "can", "did"} for orig, _ in found)
