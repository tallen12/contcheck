"""Text cleanup shared by several sources."""

import re
from collections.abc import Iterable

DROPPED_TEXT = re.compile(r"\S {2,}\S|\s[,.;:!?)]|\(\s|,,")
"""Gaps left where a converter dropped a link or template: ``the  of Brazil,,``."""


def join_paragraphs(paragraphs: Iterable[str]) -> str:
    """Join paragraphs with blank lines, the ``Article`` text convention."""
    return "\n\n".join(paragraphs)


def lines_to_paragraphs(text: str) -> str:
    """Treat every non-blank line as its own paragraph.

    Sources whose paragraphs are single lines put headings directly above the paragraph
    they introduce, so splitting only on blank lines would glue the two together.
    """
    return join_paragraphs(line.strip() for line in text.splitlines() if line.strip())


def has_dropped_text(line: str) -> bool:
    """Whether ``line`` shows the gaps a lossy wikitext conversion leaves behind."""
    return DROPPED_TEXT.search(line) is not None
