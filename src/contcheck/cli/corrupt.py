"""``contcheck corrupt``: generate rule-based training data."""

from collections.abc import Callable
from itertools import islice
from pathlib import Path

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.lemminflect_lemmatizer import LemmInflectLemmatizer
from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.corpus.documents import SplitShares, group_documents
from contcheck.corpus.jsonl import write_jsonl
from contcheck.corpus.readers import read_paragraphs
from contcheck.corrupt.agreement import AgreementCorruptor
from contcheck.corrupt.config import load_gen_config
from contcheck.corrupt.confusables import (
    Confusion,
    ConfusionMap,
    build_confusion_map,
    load_homophone_groups,
)
from contcheck.corrupt.determiner import DeterminerCorruptor
from contcheck.corrupt.generate import generate_records
from contcheck.corrupt.real_word import RealWordCorruptor, load_confusion_sets
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.corrupt.spelling import SpellingCorruptor
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
from contcheck.spelling.symspell_dictionary import SymSpellDictionary, load_symspell_dictionary


def load_confusions(dictionary: SymSpellDictionary) -> ConfusionMap:
    """Hand-written, homophone and look-alike confusions over the SymSpell word list."""
    return build_confusion_map(
        frequencies=dictionary.frequencies(),
        dictionary=dictionary,
        lemmatizer=LemmInflectLemmatizer(),
        confusion_sets=load_confusion_sets(),
        homophones=load_homophone_groups(),
    )


def keep_every_confusion(confusion: Confusion) -> bool:
    """The default confusion filter: use them all."""
    del confusion
    return True


def build_registry(
    *, keep_confusion: Callable[[Confusion], bool] = keep_every_confusion
) -> CorruptorRegistry:
    """Every corruptor the proof of concept has, wired to the real adapters.

    ``keep_confusion`` drops RealWord pairs, for example those held out for evaluation.
    """
    inflector = LemmInflector()
    dictionary = load_symspell_dictionary()
    return CorruptorRegistry(
        [
            SpellingCorruptor(dictionary),
            RealWordCorruptor(load_confusions(dictionary).filter(keep_confusion)),
            DeterminerCorruptor(),
            VerbTenseCorruptor(inflector),
            AgreementCorruptor(inflector),
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
    documents = group_documents(read_paragraphs(in_), seed=gen_config.seed, shares=SplitShares())
    records = generate_records(
        islice(documents, limit),
        analyzer=load_spacy_analyzer(),
        registry=build_registry(),
        config=gen_config,
    )
    count = write_jsonl(out, records)
    print(f"wrote {count} records to {out}")
