from itertools import islice

import pytest

from contcheck.corpus.wikipedia import SIMPLE_SNAPSHOT, WikipediaArticles, stream_from_hub


@pytest.mark.datasets
def test_streams_simple_english_wikipedia() -> None:
    source = WikipediaArticles(stream_from_hub, seed=0, snapshot=SIMPLE_SNAPSHOT)

    articles = list(islice(source.articles(), 2))

    assert len(articles) == 2
