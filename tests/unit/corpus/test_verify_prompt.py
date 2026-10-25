from contcheck.corpus.annotated import Change
from contcheck.corpus.verify_prompt import (
    SYSTEM_PROMPT,
    SentencePair,
    build_messages,
    format_changes,
    sentence_around,
)
from contcheck.llm.models import Role
from contcheck.taxonomy.kinds import Formality


def change(text: str, original: str, replacement: str) -> Change:
    start = text.index(original)
    return Change(
        start=start,
        end=start + len(original),
        original=original,
        replacement=replacement,
        type=Formality.ERROR,
    )


def test_sentence_around_shows_only_the_touched_sentence() -> None:
    text = "One is here. The legs end in Valença. Three “quoted.” Four."

    assert sentence_around(text, change(text, "legs", "ledge")) == SentencePair(
        "The legs end in Valença.", "The ledge end in Valença."
    )
    assert sentence_around(text, change(text, "Four", "4")) == SentencePair("Four.", "4.")
    assert sentence_around(text, change(text, "One is", "One")) == SentencePair(
        "One is here.", "One here."
    )


def test_a_change_across_sentences_shows_them_all() -> None:
    text = "It ends here. And starts again. Next."

    pair = sentence_around(text, change(text, "here. And", "here, and"))

    assert pair == SentencePair(
        "It ends here. And starts again.", "It ends here, and starts again."
    )


def test_deletions_leave_single_spaces() -> None:
    text = "Open a browser web now."

    assert sentence_around(text, change(text, "web", "")).after == "Open a browser now."


def test_changes_are_numbered_from_one() -> None:
    text = "Their dog ran. It were fast."
    listed = format_changes(text, [change(text, "Their", "The"), change(text, "were", "was")])

    assert listed.splitlines() == [
        '1. "Their" -> "The"',
        "   before: Their dog ran.",
        "   after: The dog ran.",
        '2. "were" -> "was"',
        "   before: It were fast.",
        "   after: It was fast.",
    ]


def test_messages_hold_the_instructions_paragraph_and_changes() -> None:
    text = "Their dog ran."
    system, user = build_messages(text, [change(text, "Their", "The")])

    assert (system.role, system.content) == (Role.SYSTEM, SYSTEM_PROMPT)
    assert user.role is Role.USER
    assert user.content.startswith("Paragraph:\nTheir dog ran.\n\nChanges:\n1. ")
