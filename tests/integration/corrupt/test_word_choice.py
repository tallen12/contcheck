import random

import pytest
from spacy.lang.en.stop_words import STOP_WORDS

from contcheck.analysis.lemminflect_lexicon import LemmInflectLexicon
from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.analysis.tense import tense_histogram
from contcheck.cli.device import choose_device, visible_gpus
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.word_choice import CONTENT_TAGS, WordChoiceCorruptor
from contcheck.lm.cache import CachedScorer
from contcheck.lm.hf_masked_lm import load_masked_scorer

PARAGRAPH = (
    "The old fisherman walked to the harbour every morning before sunrise. He repaired his "
    "nets, counted the gulls and waited for the tide to turn. In winter the boats stayed "
    "in port, and the village grew quiet until the spring storms had passed."
)


@pytest.mark.models
def test_real_masked_lm_swaps_a_content_word_for_a_same_tag_word() -> None:
    analyzer = load_spacy_analyzer()
    tokens = analyzer.analyze(PARAGRAPH).tokens
    context = CorruptionContext(tokens, tense_histogram(tokens))
    scorer = CachedScorer(load_masked_scorer(device=choose_device(None, visible_gpus())))
    corruptor = WordChoiceCorruptor(scorer, LemmInflectLexicon(), skip=STOP_WORDS)

    edits = [corruptor.corrupt(context, random.Random(seed)) for seed in range(8)]
    found = [edit for edit in edits if edit is not None]

    assert found
    for edit in found:
        (orig,), (new,) = edit.orig, edit.new
        assert context.tags[edit.start] in CONTENT_TAGS
        assert new.lower() != orig.lower()
        assert LemmInflectLexicon().lemmas(new, context.tags[edit.start])
