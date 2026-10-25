import random

import pytest

from contcheck.analysis.models import TenseHistogram
from contcheck.analysis.spacy_analyzer import SpacyAnalyzer, load_spacy_analyzer
from contcheck.corrupt.conjunction import ConjunctionCorruptor
from contcheck.corrupt.context import CorruptionContext
from contcheck.taxonomy.kinds import ErrorKind

NO_TENSE = TenseHistogram(0.0, 0.0, 0.0, 0.0)
PARAGRAPH = (
    "Although he waited for her at the station, she never came, so he listened to the radio. "
    "The man who lives next door gave him her keys, and we discussed the plan with them."
)


@pytest.fixture(scope="module")
def analyzer() -> SpacyAnalyzer:
    return load_spacy_analyzer()


def test_finds_sites_in_a_realistic_paragraph(analyzer: SpacyAnalyzer) -> None:
    context = CorruptionContext(analyzer.analyze(PARAGRAPH).tokens, NO_TENSE)
    corruptor = ConjunctionCorruptor()

    edits = [corruptor.corrupt(context, random.Random(seed)) for seed in range(40)]

    found = {(e.orig, e.new) for e in edits if e is not None}
    assert all(e is not None and e.kind is ErrorKind.CONJUNCTION for e in edits)
    assert {orig for orig, _ in found} >= {("Although",), ("so",), ("and",)}
