import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from contcheck.corpus.annotate import (
    Invalid,
    Located,
    ParagraphAnnotator,
    Parsed,
    RunSettings,
    Tally,
    annotate_concurrently,
    annotate_files,
    apply_changes,
    done_keys,
    extract_json,
    format_progress,
    input_files,
    locate_changes,
    parse_reply,
    prune_failures,
)
from contcheck.corpus.annotated import (
    Annotated,
    AnnotatedParagraph,
    Change,
    Failed,
    FailedParagraph,
    FailureReason,
    Outcome,
    ProposedChange,
    paragraph_hash,
)
from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.llm.models import LLMError
from contcheck.taxonomy.kinds import Formality
from tests.conftest import FakeLLMClient

ERROR = Formality.ERROR

TEXT = "Their going to the park, and the park are big."
REPLY = json.dumps(
    {
        "changes": [
            {"original": "Their", "replacement": "They're", "type": "error"},
            {"original": "are", "replacement": "is", "type": "error"},
        ]
    }
)


def proposed(original: str, replacement: str, kind: str = "error") -> ProposedChange:
    return ProposedChange(original=original, replacement=replacement, type=Formality(kind))


@pytest.mark.parametrize(
    "reply",
    [
        '{"changes": []}',
        '```json\n{"changes": []}\n```',
        '<think>hmm {"x": 1}</think>\n{"changes": []}',
        'Here you go: {"changes": []} Hope that helps.',
    ],
)
def test_extract_json_drops_fences_reasoning_and_chatter(reply: str) -> None:
    assert extract_json(reply) == '{"changes": []}'


def test_parse_reply_accepts_an_object_or_a_bare_list() -> None:
    item = '{"original": "a", "replacement": "b", "type": "informal"}'

    assert parse_reply('{"changes": [' + item + "]}") == Parsed((proposed("a", "b", "informal"),))
    assert parse_reply("[" + item + "]") == Parsed((proposed("a", "b", "informal"),))


@pytest.mark.parametrize(
    "reply",
    [
        "no JSON here",
        '{"changes": [{"original": "a", "replacement": "b", "type": "style"}]}',
        '{"changes": [{"original": "", "replacement": "b", "type": "error"}]}',
        '{"edits": []}',
    ],
)
def test_parse_reply_rejects_invalid_replies(reply: str) -> None:
    result = parse_reply(reply)

    assert isinstance(result, Invalid)
    assert result.reason is FailureReason.INVALID_JSON


def test_locate_changes_finds_spans_in_reading_order() -> None:
    text = "the cat and the dog"

    result = locate_changes(text, [proposed("the", "a"), proposed("the", "one")])

    assert isinstance(result, Located)
    assert [(c.start, c.end, c.replacement) for c in result.changes] == [
        (0, 3, "a"),
        (12, 15, "one"),
    ]


def test_locate_changes_handles_out_of_order_changes_and_drops_no_ops() -> None:
    text = "one two three"

    result = locate_changes(
        text, [proposed("three", "3"), proposed("two", "two"), proposed("one", "1")]
    )

    assert isinstance(result, Located)
    assert [c.original for c in result.changes] == ["one", "three"]


def test_locate_changes_matches_whole_words_only() -> None:
    text = "They disapprove of floor traders, as is usual."

    inside = locate_changes(text, [proposed("is", "was")])
    partial = locate_changes(text, [proposed("floor trader", "floor traders")])
    punctuated = locate_changes(text, [proposed(", as", " as")])

    assert isinstance(inside, Located)
    assert [c.start for c in inside.changes] == [text.index(" is ") + 1]
    assert isinstance(partial, Invalid)
    assert isinstance(punctuated, Located)


def test_locate_changes_rejects_missing_or_overlapping_text() -> None:
    missing = locate_changes("the cat", [proposed("dog", "cat")])
    overlapping = locate_changes("the cat", [proposed("the cat", "a cat"), proposed("cat", "dog")])

    assert missing == Invalid(FailureReason.NOT_FOUND, "not in the paragraph: 'dog'")
    assert isinstance(overlapping, Invalid)


def test_apply_changes_rebuilds_the_text() -> None:
    located = locate_changes(TEXT, [proposed("Their", "They're"), proposed("are", "is")])
    assert isinstance(located, Located)

    assert apply_changes(TEXT, located.changes) == "They're going to the park, and the park is big."
    assert apply_changes(TEXT, ()) == TEXT


@pytest.mark.parametrize(
    ("text", "start", "end", "expected"),
    [
        ("the the cat", 4, 7, "the cat"),  # between words: one space goes
        ("It is is.", 6, 8, "It is."),  # before punctuation: no space is stranded
        ("Well, so it goes.", 0, 5, "so it goes."),  # at the start: the next space goes
    ],
)
def test_deleting_text_leaves_no_doubled_or_stranded_space(
    text: str, start: int, end: int, expected: str
) -> None:
    change = Change(start=start, end=end, original=text[start:end], replacement="", type=ERROR)

    assert apply_changes(text, [change]) == expected


def test_a_quote_that_occurs_more_often_than_it_is_changed_is_ambiguous() -> None:
    text = "their dog and their cat"

    result = locate_changes(text, [proposed("their", "there"), proposed("dog", "dogs")])

    assert isinstance(result, Invalid)
    assert result.reason is FailureReason.AMBIGUOUS


def test_annotator_returns_the_changes_and_corrected_text() -> None:
    client = FakeLLMClient([REPLY])
    annotator = ParagraphAnnotator(client, model="m")

    outcome = annotator.annotate(RawParagraph(source_id="s", text=TEXT))

    assert isinstance(outcome, Annotated)
    paragraph = outcome.paragraph
    assert paragraph.corrected == "They're going to the park, and the park is big."
    assert paragraph.hash == paragraph_hash(TEXT)
    assert paragraph.model == "m"
    assert paragraph.attempts == 1
    options = client.requests[0][1]
    assert options.json_schema is not None
    assert options.think is False
    assert TEXT in client.requests[0][0][-1].content


def test_annotator_retries_bad_replies_at_rising_temperature() -> None:
    client = FakeLLMClient(
        [
            "nonsense",
            '{"changes": [{"original": "zebra", "replacement": "x", "type": "error"}]}',
            REPLY,
        ]
    )
    annotator = ParagraphAnnotator(client, model="m", attempts=3)

    outcome = annotator.annotate(RawParagraph(source_id="s", text=TEXT))

    assert isinstance(outcome, Annotated)
    assert outcome.paragraph.attempts == 3
    assert [options.temperature for _, options in client.requests] == pytest.approx([0, 0.3, 0.6])


def test_annotator_reports_the_last_failure() -> None:
    client = FakeLLMClient(
        ["nonsense", '{"changes": [{"original": "zebra", "replacement": "x", "type": "error"}]}']
    )

    outcome = ParagraphAnnotator(client, model="m", attempts=2).annotate(
        RawParagraph(source_id="s", text=TEXT)
    )

    assert isinstance(outcome, Failed)
    assert outcome.failure.reason is FailureReason.NOT_FOUND
    assert "zebra" in outcome.failure.reply
    assert outcome.failure.attempts == 2


def test_annotator_does_not_retry_when_the_server_is_down() -> None:
    client = FakeLLMClient([LLMError("gave up")])

    outcome = ParagraphAnnotator(client, model="m").annotate(RawParagraph(source_id="s", text=TEXT))

    assert isinstance(outcome, Failed)
    assert outcome.failure.reason is FailureReason.REQUEST
    assert outcome.failure.detail == "gave up"
    assert len(client.requests) == 1


def test_annotator_needs_an_attempt() -> None:
    with pytest.raises(ValueError, match="attempts"):
        ParagraphAnnotator(FakeLLMClient(["x"]), model="m", attempts=0)


def clean(paragraph: RawParagraph) -> Outcome:
    return Annotated(
        AnnotatedParagraph(
            source_id=paragraph.source_id,
            hash=paragraph_hash(paragraph.text),
            text=paragraph.text,
            changes=(),
            corrected=paragraph.text,
            model="m",
            attempts=1,
        )
    )


def test_annotate_concurrently_runs_in_parallel_and_keeps_order() -> None:
    barrier = threading.Barrier(3, timeout=5)

    def slow(paragraph: RawParagraph) -> Outcome:
        barrier.wait()
        time.sleep(0.01 * (3 - int(paragraph.text)))
        return clean(paragraph)

    paragraphs = [RawParagraph(source_id="s", text=str(i)) for i in range(3)]

    outcomes = list(annotate_concurrently(paragraphs, slow, concurrency=3))

    texts = [o.paragraph.text for o in outcomes if isinstance(o, Annotated)]
    assert texts == ["0", "1", "2"]


def test_annotate_concurrently_needs_a_worker() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        list(annotate_concurrently([], clean, concurrency=0))


def test_format_progress_reports_rate_share_and_time_left() -> None:
    tally = Tally(total=110, skipped=10, planned=100, annotated=20, flagged=5, failed=5)

    line = format_progress("a.jsonl", tally, elapsed=60)

    assert line == (
        "a.jsonl: 25/100 new (10 skipped as done or repeated) | 25.0 paragraphs/min"
        " | flagged 25% | failed 5 | left 0h03m"
    )


def test_input_files_takes_a_file_or_the_jsonl_files_of_a_folder(tmp_path: Path) -> None:
    for name in ("b.jsonl", "a.jsonl", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert [p.name for p in input_files(tmp_path)] == ["a.jsonl", "b.jsonl"]
    assert input_files(tmp_path / "a.jsonl") == [tmp_path / "a.jsonl"]


def test_format_progress_counts_only_what_this_run_sends() -> None:
    # With --limit 10 on a 1,000-paragraph file, only 10 are planned.
    tally = Tally(total=1000, planned=10, annotated=10)

    line = format_progress("a.jsonl", tally, elapsed=60)

    assert line.startswith("a.jsonl: 10/10 new")
    assert line.endswith("left 0h00m")


def test_done_keys_drops_a_cut_off_last_line(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    outcome = clean(RawParagraph(source_id="s", text="Done."))
    assert isinstance(outcome, Annotated)
    out.write_text(
        outcome.paragraph.model_dump_json() + '\n{"source_id": "s", "ha', encoding="utf-8"
    )

    keys = done_keys(out)

    assert keys == {("s", paragraph_hash("Done."))}
    assert out.read_text(encoding="utf-8").endswith("}\n")
    assert done_keys(tmp_path / "missing.jsonl") == set()


def test_done_keys_rejects_a_corrupt_complete_line(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    out.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"a.jsonl:1"):
        done_keys(out)


class Clock:
    def __init__(self) -> None:
        self._ticks: Iterator[float] = (float(i) for i in range(1000))

    def __call__(self) -> float:
        return next(self._ticks)


def write_raw(path: Path, texts: list[str]) -> None:
    write_jsonl(path, [RawParagraph(source_id=path.stem, text=text) for text in texts])


def test_annotate_files_writes_skips_resumes_and_logs_failures(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw / "a.jsonl", ["One.", "Two.", "One.", "Bad."])
    write_raw(raw / "b.jsonl", ["Three."])
    out = tmp_path / "out"
    lines: list[str] = []

    def annotate(paragraph: RawParagraph) -> Outcome:
        if paragraph.text == "Bad.":
            failure = FailedParagraph(
                source_id="a",
                hash=paragraph_hash("Bad."),
                reason=FailureReason.INVALID_JSON,
                detail="x",
                reply="",
                model="m",
                attempts=3,
            )
            return Failed(failure)
        return clean(paragraph)

    settings = RunSettings(concurrency=2, limit=None, log_every=2)
    first = annotate_files(raw, out, annotate, settings=settings, clock=Clock(), log=lines.append)
    second = annotate_files(raw, out, annotate, settings=settings, clock=Clock(), log=lines.append)

    annotated = [p.text for p in read_jsonl(out / "a.jsonl", AnnotatedParagraph)]
    assert annotated == ["One.", "Two."]
    assert [p.text for p in read_jsonl(out / "b.jsonl", AnnotatedParagraph)] == ["Three."]
    failed = list(read_jsonl(out / "failed" / "a.jsonl", FailedParagraph))
    assert [f.hash for f in failed] == [paragraph_hash("Bad.")]
    assert [(t.annotated, t.failed, t.skipped) for t in first] == [(2, 1, 1), (1, 0, 0)]
    assert [(t.annotated, t.failed, t.skipped) for t in second] == [(0, 1, 3), (0, 0, 1)]
    assert any(line.startswith("a.jsonl: 2/3 new") for line in lines)


def test_annotate_files_stops_at_the_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw(raw / "a.jsonl", ["One.", "Two."])
    write_raw(raw / "b.jsonl", ["Three."])
    out = tmp_path / "out"

    tallies = annotate_files(
        raw, out, clean, settings=RunSettings(limit=1), clock=Clock(), log=lambda _: None
    )

    assert [t.annotated for t in tallies] == [1]
    assert not (out / "b.jsonl").exists()


def test_the_same_text_under_two_sources_is_annotated_for_each(tmp_path: Path) -> None:
    raw = tmp_path / "a.jsonl"
    write_jsonl(
        raw,
        [
            RawParagraph(source_id="doc1", text="Same."),
            RawParagraph(source_id="doc2", text="Same."),
        ],
    )
    out = tmp_path / "out"

    annotate_files(raw, out, clean, settings=RunSettings(), clock=Clock(), log=lambda _: None)

    sources = [p.source_id for p in read_jsonl(out / "a.jsonl", AnnotatedParagraph)]
    assert sources == ["doc1", "doc2"]


def test_prune_failures_keeps_only_paragraphs_still_failing(tmp_path: Path) -> None:
    path = tmp_path / "failed.jsonl"

    def failure(text: str) -> FailedParagraph:
        return FailedParagraph(
            source_id="s",
            hash=paragraph_hash(text),
            reason=FailureReason.INVALID_JSON,
            detail="x",
            reply="",
            model="m",
            attempts=1,
        )

    write_jsonl(path, [failure("Fixed."), failure("Stuck."), failure("Fixed.")])

    prune_failures(path, drop={("s", paragraph_hash("Fixed."))})

    assert [f.hash for f in read_jsonl(path, FailedParagraph)] == [paragraph_hash("Stuck.")]
    prune_failures(tmp_path / "missing.jsonl", drop=set())
