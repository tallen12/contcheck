import pytest

from contcheck.corpus.annotated import (
    AnnotatedParagraph,
    Change,
    ProposedChange,
    ProposedChanges,
    RejectedChange,
    paragraph_hash,
)
from contcheck.taxonomy.kinds import Formality


def test_paragraph_hash_is_stable_and_short() -> None:
    assert paragraph_hash("Hello.") == paragraph_hash("Hello.")
    assert paragraph_hash("Hello.") != paragraph_hash("Hello!")
    assert len(paragraph_hash("Hello.")) == 16


def test_proposed_changes_ignore_extra_fields_and_need_a_known_type() -> None:
    reply = ProposedChanges.model_validate_json(
        '{"changes": [{"original": "u", "replacement": "you", "type": "informal",'
        ' "reason": "texting"}]}'
    )

    assert reply.changes == (
        ProposedChange(original="u", replacement="you", type=Formality.INFORMAL),
    )
    with pytest.raises(ValueError):
        ProposedChanges.model_validate_json(
            '{"changes": [{"original": "u", "replacement": "you", "type": "style"}]}'
        )


def test_annotated_paragraph_round_trips_through_json() -> None:
    paragraph = AnnotatedParagraph(
        source_id="s",
        hash=paragraph_hash("u ok"),
        text="u ok",
        changes=(Change(start=0, end=1, original="u", replacement="you", type=Formality.INFORMAL),),
        corrected="you ok",
        model="m",
        attempts=1,
    )

    assert AnnotatedParagraph.model_validate_json(paragraph.model_dump_json()) == paragraph


def test_lines_from_before_verification_load_unverified() -> None:
    line = (
        '{"source_id": "s", "hash": "h", "text": "ok", "changes": [], "corrected": "ok",'
        ' "model": "m", "attempts": 1}'
    )

    paragraph = AnnotatedParagraph.model_validate_json(line)

    assert paragraph.verifier is None
    assert paragraph.rejected == ()


def test_verified_paragraph_keeps_its_rejected_changes() -> None:
    hull = Change(start=4, end=8, original="hull", replacement="hall", type=Formality.ERROR)
    paragraph = AnnotatedParagraph(
        source_id="s",
        hash=paragraph_hash("The hull held."),
        text="The hull held.",
        changes=(),
        corrected="The hull held.",
        model="m",
        attempts=1,
        verifier="v",
        rejected=(RejectedChange(change=hull, reason="a ship has a hull"),),
    )

    assert AnnotatedParagraph.model_validate_json(paragraph.model_dump_json()) == paragraph
