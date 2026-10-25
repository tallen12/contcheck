"""``contcheck corrupt``: generate rule-based training data."""

from itertools import islice
from pathlib import Path

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.lemminflect_lemmatizer import LemmInflectLemmatizer
from contcheck.analysis.lemminflect_lexicon import LemmLexicon
from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.corpus.documents import SplitShares, group_documents
from contcheck.corpus.jsonl import write_jsonl
from contcheck.corpus.readers import read_paragraphs
from contcheck.corrupt.adjective_adverb import AdjectiveAdverbCorruptor
from contcheck.corrupt.agreement import AgreementCorruptor
from contcheck.corrupt.config import GenConfig, load_gen_config
from contcheck.corrupt.confusables import ConfusionMap, build_confusion_map, load_homophone_groups
from contcheck.corrupt.conjunction import ConjunctionCorruptor
from contcheck.corrupt.determiner import DeterminerCorruptor
from contcheck.corrupt.generate import generate_records
from contcheck.corrupt.holdout import ConfusionSplit, split_confusions
from contcheck.corrupt.noun_number import NounNumberCorruptor
from contcheck.corrupt.preposition import PrepositionCorruptor
from contcheck.corrupt.pronoun import PronounCorruptor
from contcheck.corrupt.real_word import RealWordCorruptor, load_confusion_sets
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.corrupt.spelling import SpellingCorruptor
from contcheck.corrupt.verb_form import VerbFormCorruptor
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
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
    confusions: ConfusionMap, *, dictionary: SymSpellDictionary
) -> CorruptorRegistry:
    """Every corruptor there is, wired to the real adapters, with RealWord drawing from
    ``confusions`` (already reduced to the training split)."""
    inflector = LemmInflector()
    return CorruptorRegistry(
        [
            SpellingCorruptor(dictionary),
            RealWordCorruptor(confusions),
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


def corrupt(
    *,
    config: Path = Path("configs/gen.yaml"),
    in_: Path,
    out: Path,
    limit: int | None = None,
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
    """
    gen_config = load_gen_config(config)
    dictionary = load_symspell_dictionary()
    confusions = load_confusions(dictionary)
    training = confusions.restrict(confusion_split(gen_config, confusions).train)
    documents = group_documents(read_paragraphs(in_), seed=gen_config.seed, shares=SplitShares())
    records = generate_records(
        islice(documents, limit),
        analyzer=load_spacy_analyzer(),
        registry=build_registry(training, dictionary=dictionary),
        config=gen_config,
    )
    count = write_jsonl(out, records)
    print(f"wrote {count} records to {out}")
