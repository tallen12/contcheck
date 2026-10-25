"""Padding windows into tensor batches."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from contcheck.model.models import IGNORE_INDEX, Window


@dataclass(frozen=True, slots=True)
class Batch:
    """Padded ``(batch, tokens)`` tensors; padding has mask 0 and label ignored."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.labels.to(device),
        )


def collate(windows: Sequence[Window], *, pad_id: int) -> Batch:
    """Pad ``windows`` to the longest one in the batch."""
    if not windows:
        raise ValueError("cannot collate an empty batch")
    length = max(len(window.input_ids) for window in windows)
    input_ids = torch.full((len(windows), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(windows), length), dtype=torch.long)
    labels = torch.full((len(windows), length), IGNORE_INDEX, dtype=torch.long)
    for row, window in enumerate(windows):
        size = len(window.input_ids)
        input_ids[row, :size] = torch.tensor(window.input_ids, dtype=torch.long)
        attention_mask[row, :size] = 1
        labels[row, :size] = torch.tensor(window.labels, dtype=torch.long)
    return Batch(input_ids, attention_mask, labels)
