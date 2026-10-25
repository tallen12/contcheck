"""``contcheck corrupt``: generate rule-based training data."""

from itertools import islice
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.lemminflect_lemmatizer import LemmInflectLemmatizer
from contcheck.analysis.lemminflect_lexicon import LemmInflectLexicon, LemmLexicon
from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.corpus.documents import SplitShares, group_documents
from contcheck.corpus.jsonl import write_jsonl
from contcheck.corpus.readers import read_paragraphs
from contcheck.corrupt.adjective_adverb import AdjectiveAdverbCorruptor
from contcheck.corrupt.agreement import AgreementCorruptor
from contcheck.corrupt.composite import CompositeCorruptor
from contcheck.corrupt.config import GenConfig, load_gen_config
from contcheck.corrupt.confusables import ConfusionMap, build_confusion_map, load_homophone_groups
from contcheck.corrupt.conjunction import ConjunctionCorruptor
from contcheck.corrupt.determiner import DeterminerCorruptor
from contcheck.corrupt.generate import generate_records
from contcheck.corrupt.holdout import ConfusionSplit, split_confusions
from contcheck.corrupt.noun_number import NounNumberCorruptor
from contcheck.corrupt.preposition import PrepositionCorruptor
from contcheck.corrupt.pronoun import PronounCorruptor
from contcheck.corrupt.protocols import Corruptor
from contcheck.corrupt.real_word import RealWordCorruptor, load_confusion_sets
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.corrupt.spelling import SpellingCorruptor
from contcheck.corrupt.verb_form import VerbFormCorruptor
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
from contcheck.corrupt.word_choice import WordChoiceCorruptor
from contcheck.spelling.symspell_dictionary import SymSpellDictionary, load_symspell_dictionary
from contcheck.taxonomy.kinds import ErrorKind

RULE_KINDS: tuple[ErrorKind, ...] = tuple(
    corruptor.kind
    for corruptor in (
        SpellingCorruptor,
        RealWordCorruptor,
        DeterminerCorruptor,
        VerbTenseCorruptor,
        AgreementCorruptor,
        PrepositionCorruptor,
        ConjunctionCorruptor,
        PronounCorruptor,
        VerbFormCorruptor,
        NounNumberCorruptor,
        AdjectiveAdverbCorruptor,
    )
)
"""The kinds :func:`build_registry` has corruptors for, known without loading adapters."""


def load_confusions(dictionary: SymSpellDictionary) -> ConfusionMap:
    """Hand-written, homophone and look-alike confusions over the SymSpell word list."""
    return build_confusion_map(
        frequencies=dictionary.frequencies(),
        dictionary=dictionary,
        lemmatizer=LemmInflectLemmatizer(),
        confusion_sets=load_confusion_sets(),
        homophones=load_homophone_groups(),
    )


def confusion_split(config: GenConfig, confusions: ConfusionMap) -> ConfusionSplit:
    """The confusion pairs for training and the ones withheld for the held-out check.

    ``corrupt`` and ``evaluate`` both derive the split here, from the same config and the
    same confusion map.
    """
    return split_confusions(
        confusions.as_sets(), share=config.held_out_confusion_share, seed=config.seed
    )


def build_registry(
    confusions: ConfusionMap,
    *,
    dictionary: SymSpellDictionary,
    word_choice_share: float = 0.0,
    word_choice_model: str = "distilroberta-base",
    device: str | None = None,
) -> CorruptorRegistry:
    """Every corruptor there is, wired to the real adapters.

    RealWord draws from ``confusions`` (already reduced to the training split) and, when
    ``word_choice_share`` is positive, from a masked LM on ``device`` or the largest
    discrete GPU.
    """
    inflector = LemmInflector()
    return CorruptorRegistry(
        [
            SpellingCorruptor(dictionary),
            _real_word(confusions, word_choice_share, word_choice_model, device),
            DeterminerCorruptor(),
            VerbTenseCorruptor(inflector),
            AgreementCorruptor(inflector),
            PrepositionCorruptor(),
            ConjunctionCorruptor(),
            PronounCorruptor(),
            VerbFormCorruptor(inflector),
            NounNumberCorruptor(inflector),
            AdjectiveAdverbCorruptor(LemmLexicon()),
        ]
    )


def _real_word(
    confusions: ConfusionMap, word_choice_share: float, model: str, device: str | None
) -> Corruptor:
    pairs = RealWordCorruptor(confusions)
    if word_choice_share <= 0:
        return pairs
    # Imported here so generation without the masked LM never loads torch.
    from contcheck.cli.device import choose_device, describe, visible_gpus
    from contcheck.lm.cache import CachedScorer
    from contcheck.lm.hf_masked_lm import load_masked_scorer

    gpus = visible_gpus()
    chosen = choose_device(device, gpus)
    print(f"word choice masked LM {model} on {describe(chosen, gpus)}")
    scorer = CachedScorer(load_masked_scorer(model, device=chosen))
    word_choice = WordChoiceCorruptor(scorer, LemmInflectLexicon(), skip=STOP_WORDS)
    return CompositeCorruptor(
        ErrorKind.REAL_WORD, [(pairs, 1 - word_choice_share), (word_choice, word_choice_share)]
    )


def corrupt(
    *,
    config: Path = Path("configs/gen.yaml"),
    in_: Path,
    out: Path,
    limit: int | None = None,
    device: str | None = None,
) -> None:
    """Corrupt clean paragraphs into labeled noisy records.

    Parameters
    ----------
    config
        Generation config (YAML).
    in_
        A ``.txt`` or ``.jsonl`` file of clean paragraphs, or a directory of them.
    out
        JSONL file of records to write.
    limit
        Use only the first ``limit`` documents, for quick runs.
    device
        PyTorch device for the word-choice masked LM; default: the largest discrete GPU.
    """
    gen_config = load_gen_config(config)
    dictionary = load_symspell_dictionary()
    confusions = load_confusions(dictionary)
    training = confusions.restrict(confusion_split(gen_config, confusions).train)
    documents = group_documents(read_paragraphs(in_), seed=gen_config.seed, shares=SplitShares())
    records = generate_records(
        islice(documents, limit),
        analyzer=load_spacy_analyzer(),
        registry=build_registry(
            training,
            dictionary=dictionary,
            word_choice_share=gen_config.word_choice_share,
            word_choice_model=gen_config.word_choice_model,
            device=device,
        ),
        config=gen_config,
    )
    count = write_jsonl(out, records)
    print(f"wrote {count} records to {out}")
