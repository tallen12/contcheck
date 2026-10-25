"""``ArticleSource`` streaming English Wikipedia through Hugging Face ``datasets``."""

from collections.abc import Callable, Iterable, Iterator

from contcheck.corpus.models import Article

DATASET = "wikimedia/wikipedia"
SNAPSHOT = "20231101.en"
SHUFFLE_BUFFER = 10_000

type StreamLoader = Callable[[str, str, int], Iterable[object]]
"""Opens ``(dataset, snapshot, seed)`` as a stream of unvalidated article rows."""


class WikipediaArticles:
    """Streams a shuffled order of articles from one Wikipedia snapshot.

    The dataset is CC BY-SA 4.0. Rows are validated as :class:`Article`.
    """

    def __init__(
        self, load: StreamLoader, *, seed: int, dataset: str = DATASET, snapshot: str = SNAPSHOT
    ) -> None:
        self._load = load
        self._seed = seed
        self._dataset = dataset
        self._snapshot = snapshot

    def articles(self) -> Iterator[Article]:
        for row in self._load(self._dataset, self._snapshot, self._seed):
            yield Article.model_validate(row)


def stream_from_hub(dataset: str, snapshot: str, seed: int) -> Iterator[object]:
    """Stream ``dataset`` from the Hugging Face Hub without downloading all of it.

    Shards and a buffer of rows are shuffled, so a prefix of the stream samples many
    articles instead of the alphabetically first ones.
    """
    from datasets import load_dataset  # Imported here: datasets is slow to import.

    stream = load_dataset(dataset, snapshot, split="train", streaming=True)
    yield from stream.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
