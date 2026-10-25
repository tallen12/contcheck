"""Withholding some confusion pairs from training, to test generalization.

If the tagger only memorizes the confusion pairs it was trained on, it will flag those
and miss every other wrong word. Withholding a share of pairs from the RealWord
corruptor, and later swapping them into evaluation text, measures the difference.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfusionSplit:
    """Confusions for training, and the withheld ones, both as word to alternatives."""

    train: dict[str, tuple[str, ...]]
    held_out: dict[str, tuple[str, ...]]


def pair_position(first: str, second: str, *, seed: int) -> float:
    """A stable position in [0, 1) for an unordered pair of words."""
    low, high = sorted((first, second))
    digest = hashlib.sha256(f"{seed}:{low}:{high}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_confusions(
    confusions: Mapping[str, Sequence[str]], *, share: float, seed: int
) -> ConfusionSplit:
    """Move about ``share`` of the pairs, in both directions, into the held-out map.

    The split is per unordered pair and depends only on the two words and ``seed``, so
    ``corrupt`` and ``evaluate`` derive the same split from the same config. Words left
    with no alternatives drop out of the training map; they still appear, correctly
    used, in the training text.
    """
    if not 0 <= share < 1:
        raise ValueError("share must be in [0, 1)")
    train: dict[str, list[str]] = {}
    held_out: dict[str, list[str]] = {}
    for word, alternatives in confusions.items():
        for alternative in alternatives:
            target = held_out if pair_position(word, alternative, seed=seed) < share else train
            target.setdefault(word, []).append(alternative)
    return ConfusionSplit(
        train={word: tuple(alts) for word, alts in train.items()},
        held_out={word: tuple(alts) for word, alts in held_out.items()},
    )
