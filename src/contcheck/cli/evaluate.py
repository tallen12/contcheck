"""``contcheck evaluate``: score a checkpoint and the SymSpell baseline."""

from itertools import islice
from pathlib import Path

from contcheck.cli.device import choose_device, describe, visible_gpus
from contcheck.corpus.jsonl import read_jsonl
from contcheck.corpus.record import Record
from contcheck.model.checkpoint import load_checkpoint
from contcheck.model.tagger import TorchTagger
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary
from contcheck.training.evaluate import (
    append_result,
    evaluate_tagger,
    format_result,
    symspell_baseline,
)
from contcheck.training.misspellings import build_probes, load_misspellings

BATCH_SIZE = 64


def evaluate(
    *,
    checkpoint: Path = Path("runs/baseline"),
    data: Path = Path("data/synthetic/val.jsonl"),
    results: Path = Path("runs/results.jsonl"),
    device: str | None = None,
    baseline: bool = True,
    limit: int | None = None,
) -> None:
    """Print per-kind scores for a checkpoint on a record file and log them.

    Also runs the real-misspelling check: common misspellings from a bundled list are
    swapped into the records' correct words, and the share flagged as Spelling is shown.

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
    """
    records = list(islice(read_jsonl(data, Record), limit))
    gpus = visible_gpus()
    chosen = choose_device(device, gpus)
    print(describe(chosen, gpus))
    probes = build_probes(records, load_misspellings())
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
        probes=probes,
    )
    print(format_result(result))
    append_result(results, result)
    if baseline:
        reference = symspell_baseline(
            records, dictionary=load_symspell_dictionary(), dataset=str(data), probes=probes
        )
        print()
        print(format_result(reference))
        append_result(results, reference)
