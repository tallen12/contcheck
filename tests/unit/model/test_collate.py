import pytest
import torch

from contcheck.model.collate import collate
from contcheck.model.models import IGNORE_INDEX, Window


def test_collate_pads_to_the_longest_window() -> None:
    windows = [
        Window("a", (1, 5, 2), (None, 0, None), (IGNORE_INDEX, 3, IGNORE_INDEX), start=0),
        Window("b", (1, 2), (None, None), (IGNORE_INDEX, IGNORE_INDEX), start=0),
    ]

    batch = collate(windows, pad_id=0)

    assert batch.input_ids.tolist() == [[1, 5, 2], [1, 2, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, 3, IGNORE_INDEX], [IGNORE_INDEX] * 3]
    assert batch.input_ids.dtype == torch.long
    assert batch.to("cpu").labels.tolist() == batch.labels.tolist()


def test_empty_batches_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate([], pad_id=0)
