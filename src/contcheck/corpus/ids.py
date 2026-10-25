"""Article ids for sources whose rows carry none."""

import hashlib

TEXT_ID_LENGTH = 16
"""Hex digits of the text hash kept as an id."""


def text_id(text: str) -> str:
    """A stable id from an article's text, so the same row keeps its id across fetches."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:TEXT_ID_LENGTH]
