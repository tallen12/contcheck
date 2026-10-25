"""Streaming rows of a Hugging Face Hub dataset, described by a query value."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HubQuery:
    """Which rows of a Hub dataset to stream, and how far to shuffle them.

    ``shuffle_buffer`` is a number of rows held in memory, so sources with whole books
    as rows need a much smaller buffer than sources of short posts.
    """

    dataset: str
    config: str | None = None
    data_files: str | None = None
    data_dir: str | None = None
    split: str = "train"
    shuffle_buffer: int = 10_000


type HubLoader = Callable[[HubQuery, int], Iterable[object]]
"""Opens ``(query, seed)`` as a stream of unvalidated rows."""


def stream_query(query: HubQuery, seed: int) -> Iterator[object]:
    """Stream the rows ``query`` names without downloading the whole dataset.

    Shards and a buffer of rows are shuffled, so a prefix of the stream samples the
    dataset instead of reading its first shard.
    """
    from datasets import load_dataset  # Imported here: datasets is slow to import.

    stream = load_dataset(
        query.dataset,
        query.config,
        data_files=query.data_files,
        data_dir=query.data_dir,
        split=query.split,
        streaming=True,
    )
    yield from stream.shuffle(seed=seed, buffer_size=query.shuffle_buffer)
