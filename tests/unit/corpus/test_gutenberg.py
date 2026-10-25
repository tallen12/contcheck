import json

import pytest

from contcheck.corpus.gutenberg import (
    QUERY,
    BookSelection,
    GutenbergArticles,
    GutenbergMetadata,
    author_birth_years,
    book_paragraphs,
    clean_paragraph,
    strip_boilerplate,
)
from contcheck.corpus.hub import HubQuery

FICTION = "Detective and mystery stories; Scientists -- Juvenile fiction"


def metadata(**fields: object) -> GutenbergMetadata:
    values: dict[str, object] = {
        "text_id": 1,
        "title": "A Book",
        "authors": "Goodwin, Harold L. (Harold Leland), 1914-1990",
        "subjects": FICTION,
    }
    values.update(fields)
    return GutenbergMetadata.model_validate(values)


def test_author_birth_years_skip_contributors_with_roles() -> None:
    authors = (
        "Bone, Jesse F. (Jesse Franklin), 1916-2006; Wood, Wallace, 1927-1981 [Illustrator]; "
        "Anonymous; Doyle, Arthur Conan, 1859?-1930"
    )

    assert author_birth_years(authors) == [1916, None, 1859]


@pytest.mark.parametrize(
    ("fields", "kept"),
    [
        ({}, True),
        ({"authors": "Dahn, Felix, 1834-1912; Wolffsohn, Lily [Translator]"}, False),
        ({"authors": "Anonymous"}, False),
        ({"authors": ""}, False),
        ({"subjects": "United States. Navy -- Biography"}, False),
        ({"subjects": "Science fiction; Drama"}, False),
        ({"subjects": "", "bookshelves": "Science Fiction"}, True),
        ({"language": "fr"}, False),
    ],
)
def test_selection_keeps_modern_english_fiction(fields: dict[str, object], kept: bool) -> None:
    assert BookSelection().accepts(metadata(**fields)) is kept


def test_hard_wraps_are_joined_and_single_breaks_end_paragraphs() -> None:
    text = (
        "CHAPTER I\r\nThe night passed without incident, and the party was awake at\r\n\r\n"
        "dawn.[12] Over breakfast they talked.\r\n"
        '"I\'m _hungry_," he said.\r\n\r\n\r\nII\r\nThe end came [Illustration: A ship] soon.'
    )

    assert book_paragraphs(text) == [
        "The night passed without incident, and the party was awake at dawn. "
        "Over breakfast they talked.",
        '"I\'m hungry," he said.',
        "The end came soon.",
    ]


def test_license_header_and_footer_are_cut() -> None:
    text = (
        "The Project Gutenberg eBook of X\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "Body.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicense text."
    )

    assert strip_boilerplate(text) == "\nBody.\n"


def test_clean_paragraph_drops_headings_only() -> None:
    assert clean_paragraph("Chapter XII. The Return") == ""
    assert clean_paragraph("XIV.") == ""
    assert clean_paragraph("BOOK THE FIRST") == ""
    assert clean_paragraph("Part of the crowd stayed behind.") == "Part of the crowd stayed behind."


def test_selected_books_become_articles_without_front_matter() -> None:
    body = "\r\n".join(f"Paragraph {n}." for n in range(10))
    rows = [
        {"TEXT": body, "METADATA": json.dumps(metadata(text_id=7).model_dump())},
        {"TEXT": body, "METADATA": json.dumps(metadata(text_id=8, subjects="").model_dump())},
    ]

    def load(query: HubQuery, seed: int) -> list[object]:
        assert (query, seed) == (QUERY, 4)
        return list(rows)

    source = GutenbergArticles(load, seed=4, selection=BookSelection(), start_share=0.3)
    articles = list(source.articles())

    assert [a.id for a in articles] == ["7"]
    assert articles[0].text.split("\n\n")[0] == "Paragraph 3."
