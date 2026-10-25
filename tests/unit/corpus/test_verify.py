import json
from pathlib import Path

import pytest

from contcheck.corpus.annotate import Located, RunSettings, annotate_files, locate_changes
from contcheck.corpus.annotated import (
    Annotated,
    AnnotatedParagraph,
    Failed,
    FailureReason,
    ProposedChange,
    paragraph_hash,
)
from contcheck.corpus.correct import corrected_text
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.verify import ChangeVerifier, Judged, Unusable, parse_verdicts
from contcheck.llm.models import LLMError
from contcheck.taxonomy.kinds import Formality
from tests.conftest import FakeLLMClient

TEXT = "Their going to the park. The ship's hull was gonna sink."


def annotated(text: str = TEXT) -> AnnotatedParagraph:
    located = locate_changes(
        text,
        [
            ProposedChange(original="Their", replacement="They're", type=Formality.ERROR),
            ProposedChange(original="hull", replacement="hall", type=Formality.ERROR),
            ProposedChange(original="gonna", replacement="going to", type=Formality.INFORMAL),
        ],
    )
    assert isinstance(located, Located)
    return AnnotatedParagraph(
        source_id="s",
        hash=paragraph_hash(text),
        text=text,
        changes=located.changes,
        corrected=text,
        model="annotator",
        attempts=1,
    )


def verdict(number: int, *, confirmed: bool, reason: str = "because") -> dict[str, object]:
    return {
        "id": number,
        "reason": reason,
        "original_is_wrong": True,
        "result_is_correct": confirmed,
        "keeps_meaning": True,
    }


def reply(*verdicts: dict[str, object]) -> str:
    return json.dumps({"verdicts": list(verdicts)})


def Clock() -> float:
    return 0.0


def test_verdicts_are_matched_to_changes_in_order() -> None:
    result = parse_verdicts(reply(verdict(2, confirmed=False), verdict(1, confirmed=True)), count=2)

    assert isinstance(result, Judged)
    assert [v.id for v in result.verdicts] == [1, 2]
    assert [v.confirmed for v in result.verdicts] == [True, False]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("nonsense", FailureReason.INVALID_JSON),
        ('{"verdicts": [{"id": 1}]}', FailureReason.INVALID_JSON),
        (reply(verdict(1, confirmed=True)), FailureReason.INCOMPLETE),
        (reply(verdict(1, confirmed=True), verdict(1, confirmed=True)), FailureReason.INCOMPLETE),
        (reply(verdict(1, confirmed=True), verdict(3, confirmed=True)), FailureReason.INCOMPLETE),
    ],
)
def test_unusable_verdicts(text: str, reason: FailureReason) -> None:
    result = parse_verdicts(text, count=2)

    assert isinstance(result, Unusable)
    assert result.reason is reason


def test_a_bare_list_counts_as_the_verdicts() -> None:
    result = parse_verdicts(json.dumps([verdict(1, confirmed=True)]), count=1)

    assert isinstance(result, Judged)


def test_unconfirmed_changes_move_to_rejected() -> None:
    client = FakeLLMClient(
        [reply(verdict(1, confirmed=True), verdict(2, confirmed=False, reason="hull is right"))]
    )

    outcome = ChangeVerifier(client, model="checker").verify(annotated())

    assert isinstance(outcome, Annotated)
    paragraph = outcome.paragraph
    assert [c.original for c in paragraph.changes] == ["Their", "gonna"]
    assert [(r.change.original, r.reason) for r in paragraph.rejected] == [
        ("hull", "hull is right")
    ]
    assert paragraph.verifier == "checker"
    assert paragraph.corrected == "They're going to the park. The ship's hull was going to sink."
    assert corrected_text(paragraph) == "They're going to the park. The ship's hull was gonna sink."


def test_only_error_changes_are_sent_with_their_sentences() -> None:
    client = FakeLLMClient([reply(verdict(1, confirmed=True), verdict(2, confirmed=True))])

    ChangeVerifier(client, model="checker").verify(annotated())

    messages, options = client.requests[0]
    prompt = messages[-1].content
    assert '1. "Their" -> "They\'re"' in prompt
    assert '2. "hull" -> "hall"' in prompt
    assert "after: The ship's hall was gonna sink." in prompt
    assert 'gonna" ->' not in prompt
    assert options.json_schema is not None


def test_paragraph_without_error_changes_needs_no_request() -> None:
    client = FakeLLMClient(["unused"])
    clean = AnnotatedParagraph(
        source_id="s",
        hash=paragraph_hash("Fine."),
        text="Fine.",
        changes=(),
        corrected="Fine.",
        model="annotator",
        attempts=1,
    )

    outcome = ChangeVerifier(client, model="checker").verify(clean)

    assert outcome == Annotated(clean.model_copy(update={"verifier": "checker"}))
    assert client.requests == []


def test_bad_replies_are_retried_at_rising_temperature() -> None:
    client = FakeLLMClient(
        [
            "nonsense",
            reply(verdict(1, confirmed=True)),
            reply(verdict(1, confirmed=True), verdict(2, confirmed=True)),
        ]
    )

    outcome = ChangeVerifier(client, model="checker", attempts=3).verify(annotated())

    assert isinstance(outcome, Annotated)
    assert [o.temperature for _, o in client.requests] == pytest.approx([0, 0.3, 0.6])


def test_the_last_failure_is_reported() -> None:
    client = FakeLLMClient(["nonsense", reply(verdict(1, confirmed=True))])

    outcome = ChangeVerifier(client, model="checker", attempts=2).verify(annotated())

    assert isinstance(outcome, Failed)
    assert outcome.failure.reason is FailureReason.INCOMPLETE
    assert outcome.failure.attempts == 2
    assert outcome.failure.model == "checker"


def test_server_errors_fail_at_once() -> None:
    client = FakeLLMClient([LLMError("down")])

    outcome = ChangeVerifier(client, model="checker").verify(annotated())

    assert isinstance(outcome, Failed)
    assert outcome.failure.reason is FailureReason.REQUEST
    assert len(client.requests) == 1


def test_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attempts"):
        ChangeVerifier(FakeLLMClient(["x"]), model="m", attempts=0)


def test_verified_files_resume_like_annotated_ones(tmp_path: Path) -> None:
    source = tmp_path / "annotated"
    write_jsonl(
        source / "a.jsonl", [annotated(), annotated("Their dog ran. The hull was gonna hold.")]
    )
    out = tmp_path / "verified"
    client = FakeLLMClient([reply(verdict(1, confirmed=True), verdict(2, confirmed=False))])
    verifier = ChangeVerifier(client, model="checker")

    def run() -> None:
        annotate_files(
            source,
            out,
            verifier.verify,
            read=lambda path: read_jsonl(path, AnnotatedParagraph),
            settings=RunSettings(),
            clock=Clock,
            log=lambda _: None,
        )

    run()
    run()

    verified = list(read_jsonl(out / "a.jsonl", AnnotatedParagraph))
    assert [len(p.rejected) for p in verified] == [1, 1]
    assert len(client.requests) == 2


def test_a_second_verifier_checks_only_the_kept_changes() -> None:
    first = FakeLLMClient([reply(verdict(1, confirmed=True), verdict(2, confirmed=False))])
    second = FakeLLMClient([reply(verdict(1, confirmed=False, reason="second says no"))])
    once = ChangeVerifier(first, model="a").verify(annotated())
    assert isinstance(once, Annotated)

    twice = ChangeVerifier(second, model="b").verify(once.paragraph)

    assert isinstance(twice, Annotated)
    assert '1. "Their"' in second.requests[0][0][-1].content
    assert '2. "hull"' not in second.requests[0][0][-1].content
    assert [c.original for c in twice.paragraph.changes] == ["gonna"]
    assert [r.change.original for r in twice.paragraph.rejected] == ["hull", "Their"]
    assert twice.paragraph.verifier == "a, b"
