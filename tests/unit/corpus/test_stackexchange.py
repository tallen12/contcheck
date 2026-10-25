from contcheck.corpus.hub import HubQuery
from contcheck.corpus.stackexchange import (
    DATASET,
    StackExchangeArticles,
    html_paragraphs,
    round_robin,
    site_query,
)


def question(qid: int, site: str, scores: list[int]) -> dict[str, object]:
    return {
        "qid": qid,
        "question": "<p>Question?</p>",
        "answers": [
            {
                "answer_id": qid * 10 + index,
                "pm_score": score,
                "selected": False,
                "text": f"<p>Answer {index}.</p><p>More text.</p>",
                "author": "a",
            }
            for index, score in enumerate(scores)
        ],
        "metadata": [f"https://{site}.stackexchange.com/questions/{qid}", "site", "user"],
    }


def test_site_query_names_the_site_directory() -> None:
    query = site_query("travel")

    assert (query.dataset, query.data_dir) == (DATASET, "data/travel.stackexchange.com")


def test_answers_above_the_threshold_become_articles_site_by_site() -> None:
    rows = {
        "data/travel.stackexchange.com": [
            question(1, "travel", [5, 2]),
            question(2, "travel", [4]),
        ],
        "data/cooking.stackexchange.com": [question(1, "cooking", [6])],
    }

    def load(query: HubQuery, seed: int) -> list[object]:
        assert query.data_dir is not None
        return list(rows[query.data_dir])

    source = StackExchangeArticles(load, seed=0, sites=("travel", "cooking"), min_pm_score=4)
    articles = list(source.articles())

    assert [a.id for a in articles] == ["travel-10", "cooking-10", "travel-20"]
    assert articles[0].text == "Answer 0.\n\nMore text."
    assert articles[0].title == "https://travel.stackexchange.com/questions/1"


def test_round_robin_takes_one_item_from_each_stream_in_turn() -> None:
    assert list(round_robin([[1, 2, 3], [], [10, 20]])) == [1, 10, 2, 20, 3]


def test_html_keeps_only_prose_paragraphs() -> None:
    post = (
        "<h1>Title</h1>"
        "<p>Use the <a href='x'>night train</a> &amp; save a hotel night.</p>"
        "<p>Run <code>ls</code> first.</p>"
        "<blockquote><p>Quoted text.</p></blockquote>"
        "<ul><li><p>A list item.</p></li></ul>"
        "<pre><code>print(1)</code></pre>"
        "<p>Line one<br>line two.</p>"
        "<p>It <em>really</em> works.</p>"
    )

    assert html_paragraphs(post) == [
        "Use the night train & save a hotel night.",
        "It really works.",
    ]
