"""The training loop: AdamW with warmup and linear decay, early stop on val F0.5."""

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR

from contcheck.corpus.record import Record
from contcheck.model.align import record_windows
from contcheck.model.collate import Batch, collate
from contcheck.model.models import Window
from contcheck.model.protocols import Tokenizer
from contcheck.model.tagger import ContextTagger, TorchTagger
from contcheck.training.config import TrainConfig
from contcheck.training.evaluate import evaluate_tagger
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


def _snapshot(model: ContextTagger) -> dict[str, torch.Tensor]:
    """A copy of the weights that later training steps do not change."""
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {name} is not a tensor")
        state[name] = value.detach().clone()
    return state


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
    return evaluate_tagger(
        records,
        tagger=TorchTagger(model, device),
        tokenizer=tokenizer,
        window=config.window,
        stride=config.stride,
        batch_size=config.batch_size * 2,
        name=name,
        dataset="val",
    )


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
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        # Summed on the device: calling .item() every step would wait for the GPU.
        running = torch.zeros((), device=device)
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
            running += output.loss.detach()
            steps += 1
            if steps % config.log_every == 0:
                mean = running.item() / steps
                log(f"epoch {epoch} step {steps}/{steps_per_epoch} loss {mean:.4f}")
        train_loss = running.item() / max(steps, 1)

        val = validate(
            model,
            val_records,
            tokenizer=tokenizer,
            config=config,
            device=device,
            name=f"epoch-{epoch}",
        )
        history.append(EpochReport(epoch, train_loss, val))
        log(
            f"epoch {epoch}: loss {train_loss:.4f}, val F0.5 {val.overall.f05:.4f}, "
            f"detection F0.5 {val.detection.f05:.4f}"
        )
        if val.overall.f05 > best_f05:
            best_f05 = val.overall.f05
            best_epoch = epoch
            best_state = _snapshot(model)
        elif epoch - best_epoch >= config.patience:
            log(f"stopping: no improvement since epoch {best_epoch}")
            break
    if best_state is None:
        # Epoch 1 always beats the starting score of -1, and epochs is at least 1.
        raise RuntimeError("training ran no epochs")
    return TrainResult(best_epoch, best_state, tuple(history))
