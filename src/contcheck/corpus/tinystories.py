"""``ArticleSource`` streaming TinyStories, short children's stories written by GPT-4.

The rows come from ``noanabeshima/TinyStoriesV2``, the GPT-4-only V2 stories of
``roneneldan/TinyStories`` (CDLA-Sharing 1.0) as JSONL, one story per row. The original
dataset's parquet files hold V1, which mixes in GPT-3.5 stories with more errors, and its
V2 text files separate stories with ``<|endoftext|>`` lines, which a row stream cannot
group. Each story keeps one paragraph per line and becomes one article.
"""

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from contcheck.corpus.cleanup import lines_to_paragraphs
from contcheck.corpus.hub import HubLoader, HubQuery
from contcheck.corpus.ids import text_id
from contcheck.corpus.models import Article

QUERY = HubQuery(dataset="noanabeshima/TinyStoriesV2", data_files="TinyStoriesV2-GPT4-train.jsonl")


class TinyStoryRow(BaseModel):
    """One story of the dataset."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str


class TinyStoriesArticles:
    """Streams a shuffled order of TinyStories stories as articles."""

    def __init__(self, load: HubLoader, *, seed: int, query: HubQuery = QUERY) -> None:
        self._load = load
        self._seed = seed
        self._query = query

    def articles(self) -> Iterator[Article]:
        for raw in self._load(self._query, self._seed):
            row = TinyStoryRow.model_validate(raw)
            yield Article(id=text_id(row.text), title="", text=lines_to_paragraphs(row.text))
