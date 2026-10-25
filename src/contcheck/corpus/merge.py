"""Merging record files: dedupe, shuffle, and keep each document in its split."""

import random
from collections.abc import Iterable, Mapping

from contcheck.corpus.record import Record, Split


def merge_records(records: Iterable[Record], *, seed: int) -> Mapping[Split, list[Record]]:
    """Group records by split, drop repeated noisy texts, and shuffle each split.

    Two variants of a paragraph that came out the same (often both left clean) count
    as one. The split is taken from each record, which got it from its document, so no
    document crosses splits.
    """
    seen: set[tuple[str, int, str]] = set()
    by_split: dict[Split, list[Record]] = {split: [] for split in Split}
    for record in records:
        key = (record.doc_id, record.paragraph, record.noisy)
        if key in seen:
            continue
        seen.add(key)
        by_split[record.split].append(record)
    rng = random.Random(seed)
    for split_records in by_split.values():
        rng.shuffle(split_records)
    return by_split
