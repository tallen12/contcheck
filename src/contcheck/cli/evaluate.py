"""``contcheck evaluate``: score a checkpoint and the SymSpell baseline."""

from collections.abc import Sequence
from itertools import islice
from pathlib import Path

from contcheck.cli.corrupt import confusion_split, load_confusions
from contcheck.cli.device import choose_device, describe, visible_gpus
from contcheck.corpus.jsonl import read_jsonl
from contcheck.corpus.record import Record
from contcheck.corrupt.config import GenConfig, load_gen_config
from contcheck.corrupt.confusables import ConfusionMap
from contcheck.model.checkpoint import load_checkpoint
from contcheck.model.tagger import TorchTagger
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.evaluate import (
    append_result,
    evaluate_tagger,
    format_result,
    symspell_baseline,
)
from contcheck.training.misspellings import load_misspellings
from contcheck.training.probes import CheckName, ProbeSet, substitution_probes

BATCH_SIZE = 64


def evaluate(
    *,
    checkpoint: Path = Path("runs/baseline"),
    data: Path = Path("data/synthetic/val.jsonl"),
    results: Path = Path("runs/results.jsonl"),
    device: str | None = None,
    baseline: bool = True,
    limit: int | None = None,
    gen_config: Path = Path("configs/gen.yaml"),
) -> None:
    """Print per-kind scores for a checkpoint on a record file and log them.

    Also runs the probe checks: words in the records are swapped for common real
    misspellings, for confusion partners withheld from training, and for partners seen in
    training, and the share flagged is shown for each.

    Parameters
    ----------
    checkpoint
        Checkpoint directory written by ``contcheck train``.
    data
        Record JSONL file to score on.
    results
        Results log; one JSON line per scored model is appended.
    device
        PyTorch device; default: the largest discrete GPU.
    baseline
        Also score the SymSpell-only baseline.
    limit
        Use only the first ``limit`` records.
    gen_config
        Generation config the training data was made with; it defines which confusion
        pairs were withheld for the held-out check.
    """
    records = list(islice(read_jsonl(data, Record), limit))
    gpus = visible_gpus()
    chosen = choose_device(device, gpus)
    print(describe(chosen, gpus))
    dictionary = load_symspell_dictionary()
    checks = build_checks(
        records, gen_config=load_gen_config(gen_config), confusions=load_confusions(dictionary)
    )
    loaded = load_checkpoint(checkpoint, device=chosen)
    result = evaluate_tagger(
        records,
        tagger=TorchTagger(loaded.model, chosen),
        tokenizer=loaded.tokenizer,
        window=loaded.info.window,
        stride=loaded.info.stride,
        batch_size=BATCH_SIZE,
        name=str(checkpoint),
        dataset=str(data),
        checks=checks,
    )
    print(format_result(result))
    append_result(results, result)
    if baseline:
        reference = symspell_baseline(
            records, dictionary=dictionary, dataset=str(data), checks=checks
        )
        print()
        print(format_result(reference))
        append_result(results, reference)


def build_checks(
    records: Sequence[Record], *, gen_config: GenConfig, confusions: ConfusionMap
) -> dict[CheckName, ProbeSet]:
    """The probe sets for every check, built from ``records``."""
    split = confusion_split(gen_config, confusions)
    checks = {
        CheckName.REAL_MISSPELLINGS: ProbeSet(
            ErrorKind.SPELLING, substitution_probes(records, load_misspellings())
        ),
        CheckName.SEEN_CONFUSIONS: ProbeSet(
            ErrorKind.REAL_WORD, substitution_probes(records, split.train)
        ),
    }
    if split.held_out:
        checks[CheckName.HELD_OUT_CONFUSIONS] = ProbeSet(
            ErrorKind.REAL_WORD, substitution_probes(records, split.held_out)
        )
    return checks
