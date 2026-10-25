import random

import pytest
import torch
from transformers import ElectraConfig

from contcheck.corpus.record import Record
from contcheck.model.tagger import ContextTagger, encoder_from_config
from contcheck.training.config import TrainConfig
from contcheck.training.train import (
    shuffled_batches,
    train,
    training_windows,
    validate,
    warmup_then_linear_decay,
)
from tests.conftest import FakeTokenizer


def test_schedule_warms_up_then_decays() -> None:
    factors = [warmup_then_linear_decay(step, warmup=2, total=6) for step in range(6)]

    assert factors == pytest.approx([0.5, 1.0, 1.0, 0.75, 0.5, 0.25])


def test_batches_cover_every_window_once(records: list[Record], tokenizer: FakeTokenizer) -> None:
    windows = training_windows(records, tokenizer, TrainConfig(window=16, stride=8))

    batches = list(shuffled_batches(windows, batch_size=5, pad_id=0, rng=random.Random(0)))

    assert [b.input_ids.shape[0] for b in batches] == [5, 5, 5, 1]


def test_smoke_training_learns_and_keeps_the_best_epoch(
    records: list[Record], tokenizer: FakeTokenizer, tiny_encoder_config: ElectraConfig
) -> None:
    torch.manual_seed(0)
    model = ContextTagger(encoder_from_config(tiny_encoder_config), outside_weight=0.5)
    config = TrainConfig(
        window=16, stride=8, batch_size=4, epochs=3, patience=3, head_lr=5e-3, encoder_lr=5e-3
    )
    messages: list[str] = []

    result = train(
        model,
        tokenizer=tokenizer,
        train_records=records,
        val_records=records,
        config=config,
        device=torch.device("cpu"),
        log=messages.append,
    )

    assert [report.epoch for report in result.history] == [1, 2, 3]
    assert result.history[-1].train_loss < result.history[0].train_loss
    assert result.best_state.keys() == model.state_dict().keys()
    assert result.best_epoch in {1, 2, 3}
    assert any("val F0.5" in message for message in messages)


def test_early_stop_after_patience(
    records: list[Record], tokenizer: FakeTokenizer, tiny_encoder_config: ElectraConfig
) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config))
    # A learning rate this small changes nothing, so no epoch improves on the first.
    config = TrainConfig(
        window=16, stride=8, batch_size=8, epochs=5, patience=1, head_lr=1e-12, encoder_lr=1e-12
    )

    result = train(
        model,
        tokenizer=tokenizer,
        train_records=records,
        val_records=records,
        config=config,
        device=torch.device("cpu"),
        log=lambda _: None,
    )

    assert len(result.history) == 2
    assert result.best_epoch == 1


def test_validate_scores_every_record(
    records: list[Record], tokenizer: FakeTokenizer, tiny_encoder_config: ElectraConfig
) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config))

    result = validate(
        model,
        records,
        tokenizer=tokenizer,
        config=TrainConfig(window=16, stride=8),
        device=torch.device("cpu"),
        name="x",
    )

    gold = result.overall.true_positives + result.overall.false_negatives
    assert gold == sum(len(r.edits) for r in records)
    assert result.clean_words == 8 * 7
