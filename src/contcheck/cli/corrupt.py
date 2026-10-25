"""``contcheck corrupt``: generate rule-based training data."""

from itertools import islice
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

from contcheck.analysis.lemminflect_inflector import LemmInflector
from contcheck.analysis.lemminflect_lexicon import LemmInflectLexicon
from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.corpus.documents import SplitShares, group_documents
from contcheck.corpus.jsonl import write_jsonl
from contcheck.corpus.readers import read_paragraphs
from contcheck.corrupt.agreement import AgreementCorruptor
from contcheck.corrupt.composite import CompositeCorruptor
from contcheck.corrupt.config import load_gen_config
from contcheck.corrupt.determiner import DeterminerCorruptor
from contcheck.corrupt.generate import generate_records
from contcheck.corrupt.protocols import Corruptor
from contcheck.corrupt.real_word import RealWordCorruptor, load_confusion_sets
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.corrupt.spelling import SpellingCorruptor
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
from contcheck.corrupt.word_choice import WordChoiceCorruptor
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary
from contcheck.taxonomy.kinds import ErrorKind


def build_registry(
    *,
    word_choice_share: float = 0.0,
    word_choice_model: str = "distilroberta-base",
    device: str | None = None,
) -> CorruptorRegistry:
    """Every corruptor the proof of concept has, wired to the real adapters.

    The masked LM behind word-choice RealWord errors is loaded only when
    ``word_choice_share`` is positive, on ``device`` or the largest discrete GPU.
    """
    inflector = LemmInflector()
    return CorruptorRegistry(
        [
            SpellingCorruptor(load_symspell_dictionary()),
            _real_word(word_choice_share, word_choice_model, device),
            DeterminerCorruptor(),
            VerbTenseCorruptor(inflector),
            AgreementCorruptor(inflector),
        ]
    )


def _real_word(word_choice_share: float, model: str, device: str | None) -> Corruptor:
    confusions = RealWordCorruptor(load_confusion_sets())
    if word_choice_share <= 0:
        return confusions
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
        ErrorKind.REAL_WORD, [(confusions, 1 - word_choice_share), (word_choice, word_choice_share)]
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
    documents = group_documents(read_paragraphs(in_), seed=gen_config.seed, shares=SplitShares())
    records = generate_records(
        islice(documents, limit),
        analyzer=load_spacy_analyzer(),
        registry=build_registry(
            word_choice_share=gen_config.word_choice_share,
            word_choice_model=gen_config.word_choice_model,
            device=device,
        ),
        config=gen_config,
    )
    count = write_jsonl(out, records)
    print(f"wrote {count} records to {out}")
