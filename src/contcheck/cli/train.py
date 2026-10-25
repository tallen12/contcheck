"""``contcheck train``: fine-tune the baseline tagger."""

import json
from itertools import islice
from pathlib import Path

import torch

from contcheck.cli.device import choose_device, describe, visible_gpus
from contcheck.corpus.jsonl import read_jsonl
from contcheck.corpus.record import Record
from contcheck.model.checkpoint import CheckpointInfo, save_checkpoint
from contcheck.model.hf_tokenizer import load_tokenizer
from contcheck.model.tagger import ContextTagger, pretrained_encoder
from contcheck.taxonomy.labels import LABELS
from contcheck.training.config import load_train_config
from contcheck.training.train import train as run_training

HISTORY = "history.jsonl"


def train(
    *,
    config: Path = Path("configs/train.yaml"),
    device: str | None = None,
    limit: int | None = None,
) -> None:
    """Train the tagger and save the best epoch as a checkpoint directory.

    Parameters
    ----------
    config
        Training config (YAML).
    device
        PyTorch device such as ``cuda:0`` or ``cpu``; default: the largest discrete GPU.
    limit
        Use only the first ``limit`` train and val records, for smoke runs.
    """
    settings = load_train_config(config)
    gpus = visible_gpus()
    chosen = choose_device(device, gpus)
    print(describe(chosen, gpus))
    if settings.use_doc:
        raise SystemExit("use_doc needs document vectors, which arrive in M4")

    tokenizer = load_tokenizer(settings.encoder)
    model = ContextTagger(
        pretrained_encoder(settings.encoder),
        use_doc=False,
        fuse_dim=settings.fuse_dim,
        outside_weight=settings.outside_weight,
    )
    train_records = list(islice(read_jsonl(settings.train, Record), limit))
    val_records = list(islice(read_jsonl(settings.val, Record), limit))
    result = run_training(
        model,
        tokenizer=tokenizer,
        train_records=train_records,
        val_records=val_records,
        config=settings,
        device=torch.device(chosen),
        log=print,
    )

    best = result.history[result.best_epoch - 1]
    save_checkpoint(
        settings.out,
        model=model,
        state=result.best_state,
        tokenizer=tokenizer,
        info=CheckpointInfo(
            encoder=settings.encoder,
            labels=LABELS,
            use_doc=False,
            fuse_dim=settings.fuse_dim,
            window=settings.window,
            stride=settings.stride,
            epoch=result.best_epoch,
            val_f05=best.val.overall.f05,
        ),
    )
    with (settings.out / HISTORY).open("w", encoding="utf-8") as file:
        for report in result.history:
            line = {
                "epoch": report.epoch,
                "train_loss": report.train_loss,
                "val": report.val.model_dump(mode="json"),
            }
            file.write(json.dumps(line) + "\n")
    print(
        f"saved epoch {result.best_epoch} (val F0.5 {best.val.overall.f05:.4f}) to {settings.out}"
    )
