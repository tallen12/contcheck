"""The training loop: AdamW with warmup and linear decay, early stop on val F0.5."""

import copy
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

from contcheck.corpus.record import Record
from contcheck.infer.decode import decode
from contcheck.infer.predict import predict_paragraphs
from contcheck.model.align import record_windows
from contcheck.model.collate import Batch, collate
from contcheck.model.models import Window
from contcheck.model.protocols import Tokenizer
from contcheck.model.tagger import ContextTagger, TorchTagger
from contcheck.taxonomy.bio import WordSpan, labels_to_spans
from contcheck.training.config import TrainConfig
from contcheck.training.metrics import ParagraphSpans, score
from contcheck.training.models import EvalResult

type Log = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class EpochReport:
    """Mean training loss and validation scores after one epoch."""

    epoch: int
    train_loss: float
    val: EvalResult


@dataclass(frozen=True, slots=True)
class TrainResult:
    """The best epoch's weights and every epoch's report."""

    best_epoch: int
    best_state: dict[str, torch.Tensor]
    history: tuple[EpochReport, ...]


def warmup_then_linear_decay(step: int, *, warmup: int, total: int) -> float:
    """Learning-rate factor: linear from 0 to 1 over ``warmup`` steps, then down to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def training_windows(
    records: Sequence[Record], tokenizer: Tokenizer, config: TrainConfig
) -> list[Window]:
    """Every window of every record."""
    return [
        window
        for record in records
        for window in record_windows(record, tokenizer, window=config.window, stride=config.stride)
    ]


def validate(
    model: ContextTagger,
    records: Sequence[Record],
    *,
    tokenizer: Tokenizer,
    config: TrainConfig,
    device: torch.device,
    name: str,
) -> EvalResult:
    """Score ``model`` on ``records`` with span-level metrics."""
    tagger = TorchTagger(model, device)
    probabilities = predict_paragraphs(
        [record.words for record in records],
        tagger=tagger,
        tokenizer=tokenizer,
        window=config.window,
        stride=config.stride,
        batch_size=config.batch_size * 2,
    )
    paragraphs = (
        ParagraphSpans(
            words=len(record.words),
            gold=tuple(labels_to_spans(record.labels)),
            predicted=tuple(WordSpan(e.start, e.end, e.kind) for e in decode(probs)),
        )
        for record, probs in zip(records, probabilities, strict=True)
    )
    return score(paragraphs, name=name, dataset="val")


def shuffled_batches(
    windows: Sequence[Window], *, batch_size: int, pad_id: int, rng: random.Random
) -> Iterator[Batch]:
    """One epoch of padded batches in a fresh random order."""
    order = list(range(len(windows)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield collate([windows[i] for i in order[start : start + batch_size]], pad_id=pad_id)


def train(
    model: ContextTagger,
    *,
    tokenizer: Tokenizer,
    train_records: Sequence[Record],
    val_records: Sequence[Record],
    config: TrainConfig,
    device: torch.device,
    log: Log,
) -> TrainResult:
    """Train ``model`` and keep the weights of the epoch with the best val F0.5.

    Stops early after ``config.patience`` epochs without improvement. On a GPU the
    forward pass runs in bfloat16 autocast; the loss is computed in float32.
    """
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    windows = training_windows(train_records, tokenizer, config)
    steps_per_epoch = -(-len(windows) // config.batch_size)
    log(f"{len(train_records)} train records, {len(windows)} windows")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(encoder_lr=config.encoder_lr, head_lr=config.head_lr)
    )
    total = config.epochs * steps_per_epoch
    warmup = max(1, round(config.warmup_ratio * total))
    scheduler = LambdaLR(optimizer, partial(warmup_then_linear_decay, warmup=warmup, total=total))
    autocast = device.type == "cuda"

    history: list[EpochReport] = []
    best_epoch = 0
    best_f05 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        batches = shuffled_batches(
            windows, batch_size=config.batch_size, pad_id=tokenizer.pad_id, rng=rng
        )
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                output = model.forward(batch.input_ids, batch.attention_mask, labels=batch.labels)
            if output.loss is None:
                raise RuntimeError("the model returned no loss for a labeled batch")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += output.loss.item()
            steps += 1
            if steps % config.log_every == 0:
                log(f"epoch {epoch} step {steps}/{steps_per_epoch} loss {running / steps:.4f}")

        val = validate(
            model,
            val_records,
            tokenizer=tokenizer,
            config=config,
            device=device,
            name=f"epoch-{epoch}",
        )
        history.append(EpochReport(epoch, running / max(steps, 1), val))
        log(
            f"epoch {epoch}: loss {running / max(steps, 1):.4f}, val F0.5 {val.overall.f05:.4f}, "
            f"detection F0.5 {val.detection.f05:.4f}"
        )
        if val.overall.f05 > best_f05:
            best_f05 = val.overall.f05
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            log(f"stopping: no improvement since epoch {best_epoch}")
            break
    return TrainResult(best_epoch, best_state, tuple(history))
