from contcheck.corpus.hub import HubQuery
from contcheck.corpus.wikimedia import DATASET, Wiki, WikimediaArticles, page_text, wiki_query


def page(page_id: str, namespace: str, text: str) -> dict[str, object]:
    return {
        "id": page_id,
        "text": text,
        "metadata": {"namespace": namespace, "title": "Aachen", "wiki": "wikivoyage.com"},
    }


def test_query_reads_only_the_files_of_one_wiki() -> None:
    query = wiki_query(Wiki.WIKIVOYAGE)

    assert query.dataset == DATASET
    assert query.data_files == "*_wikivoyage.com.jsonl.gz"


def test_only_main_namespace_pages_become_articles() -> None:
    def load(query: HubQuery, seed: int) -> list[object]:
        assert query == wiki_query(Wiki.WIKIBOOKS)
        return [page("0-1", "0", "Aachen\n\nA city."), page("2-4", "2", "User page.")]

    articles = list(WikimediaArticles(load, seed=1, wiki=Wiki.WIKIBOOKS).articles())

    assert [(a.id, a.text) for a in articles] == [("0-1", "A city.")]


def test_page_text_splits_lines_unescapes_and_drops_gaps() -> None:
    text = (
        "Aachen\n\nAachen is a city &mdash; a spa town.\n"
        "Understand\nThe  of the city met,, today.\n"
    )

    assert page_text(text, "Aachen") == "Aachen is a city — a spa town.\n\nUnderstand"
