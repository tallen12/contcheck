"""Annotate clean paragraphs with the changes an LLM would make, typed error or informal.

The model lists changes, not a rewrite; each must quote an exact span of the paragraph.
Replies that are not valid JSON, or quote text the paragraph lacks, are retried at a
higher temperature and logged as failures when every attempt fails. Output files are
appended to and skip paragraphs already there, so an interrupted run resumes.
"""

import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, assert_never

from pydantic import ValidationError

from contcheck.corpus.annotate_prompt import RESPONSE_SCHEMA, SCHEMA_NAME, build_messages
from contcheck.corpus.annotated import (
    Annotated,
    AnnotatedParagraph,
    Change,
    Failed,
    FailedParagraph,
    FailureReason,
    Outcome,
    ProposedChange,
    ProposedChanges,
    paragraph_hash,
)
from contcheck.corpus.documents import RawParagraph
from contcheck.corpus.readers import read_paragraphs
from contcheck.llm.models import ChatOptions, LLMError
from contcheck.llm.protocols import LLMClient

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
FAILED_DIR = "failed"
DEFAULT_ATTEMPTS = 3
RETRY_TEMPERATURE_STEP = 0.3
REPLY_EXCERPT = 2000


@dataclass(frozen=True, slots=True)
class Parsed:
    """The reply held a valid list of proposed changes."""

    changes: tuple[ProposedChange, ...]


@dataclass(frozen=True, slots=True)
class Located:
    """Every proposed change was found in the paragraph."""

    changes: tuple[Change, ...]


@dataclass(frozen=True, slots=True)
class Invalid:
    """The reply cannot be used."""

    reason: FailureReason
    detail: str


def extract_json(reply: str) -> str:
    """The JSON part of a reply: without reasoning blocks, code fences or chatter around it."""
    text = THINK_BLOCK.sub("", reply).strip()
    fenced = CODE_FENCE.search(text)
    if fenced is not None:
        text = str(fenced.group(1)).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1] if end > start else text[start:]


def parse_reply(reply: str) -> Parsed | Invalid:
    """Validate a reply as ``ProposedChanges``; a bare JSON list counts as the changes."""
    try:
        value: object = json.loads(extract_json(reply))
    except json.JSONDecodeError as error:
        return Invalid(FailureReason.INVALID_JSON, f"not JSON: {error}")
    if isinstance(value, list):
        value = {"changes": value}
    try:
        return Parsed(ProposedChanges.model_validate(value).changes)
    except ValidationError as error:
        return Invalid(FailureReason.INVALID_JSON, f"schema: {error.errors()[0]['msg']}")


def locate_changes(text: str, proposed: Sequence[ProposedChange]) -> Located | Invalid:
    """Find each change's ``original`` in ``text``, without overlaps, in text order.

    Each change takes the first free occurrence after the previous change, else the
    first free one anywhere, so repeated words resolve in reading order. An occurrence
    that starts or ends inside a word does not count. Changes that leave their span as
    it is are dropped.
    """
    taken: list[Change] = []
    cursor = 0
    for change in proposed:
        if change.original == change.replacement:
            continue
        start = _free_occurrence(text, change.original, cursor, taken)
        if start is None:
            start = _free_occurrence(text, change.original, 0, taken)
        if start is None:
            return Invalid(FailureReason.NOT_FOUND, f"not in the paragraph: {change.original!r}")
        end = start + len(change.original)
        taken.append(
            Change(
                start=start,
                end=end,
                original=change.original,
                replacement=change.replacement,
                type=change.type,
            )
        )
        cursor = end
    return Located(tuple(sorted(taken, key=lambda c: c.start)))


def _free_occurrence(text: str, needle: str, cursor: int, taken: Sequence[Change]) -> int | None:
    start = text.find(needle, cursor)
    while start >= 0:
        end = start + len(needle)
        free = all(end <= other.start or start >= other.end for other in taken)
        if free and not _splits_a_word(text, needle, start, end):
            return start
        start = text.find(needle, start + 1)
    return None


def _splits_a_word(text: str, needle: str, start: int, end: int) -> bool:
    # Models quote whole words, so "is" must not match inside "disapprove", nor
    # "floor trader" inside "floor traders".
    cut_left = needle[0].isalnum() and start > 0 and text[start - 1].isalnum()
    cut_right = needle[-1].isalnum() and end < len(text) and text[end].isalnum()
    return cut_left or cut_right


def apply_changes(text: str, changes: Sequence[Change]) -> str:
    """``text`` with non-overlapping ``changes`` applied."""
    parts: list[str] = []
    position = 0
    for change in sorted(changes, key=lambda c: c.start):
        parts.append(text[position : change.start])
        parts.append(change.replacement)
        position = change.end
    parts.append(text[position:])
    return "".join(parts)


class ParagraphAnnotator:
    """Asks the model for one paragraph's changes, retrying unusable replies.

    Attempt ``n`` (from 0) samples at temperature ``n * 0.3``, so a retry does not repeat
    a deterministic mistake. ``model`` is only recorded in the output.
    """

    def __init__(self, client: LLMClient, *, model: str, attempts: int = DEFAULT_ATTEMPTS) -> None:
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        self._client = client
        self._model = model
        self._attempts = attempts

    def annotate(self, paragraph: RawParagraph) -> Outcome:
        """Annotate one paragraph; never raises for a bad reply or an unreachable server."""
        digest = paragraph_hash(paragraph.text)
        messages = build_messages(paragraph.text)
        problem = Invalid(FailureReason.REQUEST, "no attempt made")
        reply = ""
        for attempt in range(self._attempts):
            options = ChatOptions(
                temperature=attempt * RETRY_TEMPERATURE_STEP,
                json_schema=RESPONSE_SCHEMA,
                schema_name=SCHEMA_NAME,
            )
            try:
                reply = self._client.complete(messages, options=options)
            except LLMError as error:
                problem = Invalid(FailureReason.REQUEST, str(error))
                return Failed(self._failure(paragraph, digest, problem, attempt + 1, ""))
            result = _check(paragraph.text, reply)
            match result:
                case Located(changes):
                    return Annotated(
                        AnnotatedParagraph(
                            source_id=paragraph.source_id,
                            hash=digest,
                            text=paragraph.text,
                            changes=changes,
                            corrected=apply_changes(paragraph.text, changes),
                            model=self._model,
                            attempts=attempt + 1,
                        )
                    )
                case Invalid():
                    problem = result
                case _:
                    assert_never(result)
        return Failed(self._failure(paragraph, digest, problem, self._attempts, reply))

    def _failure(
        self, paragraph: RawParagraph, digest: str, problem: Invalid, attempts: int, reply: str
    ) -> FailedParagraph:
        return FailedParagraph(
            source_id=paragraph.source_id,
            hash=digest,
            reason=problem.reason,
            detail=problem.detail,
            reply=reply[:REPLY_EXCERPT],
            model=self._model,
            attempts=attempts,
        )


def _check(text: str, reply: str) -> Located | Invalid:
    parsed = parse_reply(reply)
    match parsed:
        case Parsed(changes):
            return locate_changes(text, changes)
        case Invalid():
            return parsed
        case _:
            assert_never(parsed)


def annotate_concurrently(
    paragraphs: Iterable[RawParagraph],
    annotate: Callable[[RawParagraph], Outcome],
    *,
    concurrency: int,
) -> Iterator[Outcome]:
    """Run ``annotate`` on up to ``concurrency`` paragraphs at once; yields in input order."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: deque[Future[Outcome]] = deque()
        for paragraph in paragraphs:
            pending.append(pool.submit(annotate, paragraph))
            if len(pending) >= 2 * concurrency:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


@dataclass(slots=True)
class Tally:
    """Running counts for one input file."""

    total: int
    skipped: int = 0
    annotated: int = 0
    flagged: int = 0
    failed: int = 0

    def add(self, outcome: Outcome) -> None:
        match outcome:
            case Annotated(paragraph):
                self.annotated += 1
                self.flagged += bool(paragraph.changes)
            case Failed():
                self.failed += 1
            case _:
                assert_never(outcome)

    @property
    def done(self) -> int:
        return self.annotated + self.failed


def format_progress(name: str, tally: Tally, *, elapsed: float) -> str:
    """One progress line: counts, throughput, flagged share and time left."""
    rate = tally.done / elapsed * 60 if elapsed > 0 else 0.0
    remaining = tally.total - tally.skipped - tally.done
    flagged = tally.flagged / tally.annotated if tally.annotated else 0.0
    eta = _duration(remaining / rate * 60) if rate > 0 else "?"
    return (
        f"{name}: {tally.done}/{tally.total - tally.skipped} new"
        f" ({tally.skipped} skipped as done or repeated) | {rate:.1f} paragraphs/min"
        f" | flagged {flagged:.0%} | failed {tally.failed} | left {eta}"
    )


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def input_files(path: Path) -> list[Path]:
    """``path`` itself, or the ``.jsonl`` files directly in it, in name order."""
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".jsonl")
    return [path]


def done_hashes(path: Path) -> set[str]:
    """Hashes already in an output file; drops a final line cut off by an interrupted run."""
    if not path.exists():
        return set()
    content = path.read_bytes()
    complete = content[: content.rfind(b"\n") + 1]
    if len(complete) != len(content):
        path.write_bytes(complete)
    hashes: set[str] = set()
    for number, line in enumerate(complete.decode("utf-8").splitlines(), start=1):
        if line.strip():
            try:
                hashes.add(AnnotatedParagraph.model_validate_json(line).hash)
            except ValidationError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return hashes


@dataclass(frozen=True, slots=True)
class RunSettings:
    """How ``annotate_files`` runs: parallel requests, a cap on new paragraphs, logging."""

    concurrency: int = 4
    limit: int | None = None
    log_every: int = 25


def annotate_files(
    source: Path,
    out_dir: Path,
    annotate: Callable[[RawParagraph], Outcome],
    *,
    settings: RunSettings,
    clock: Callable[[], float],
    log: Callable[[str], None],
) -> list[Tally]:
    """Annotate every paragraph of ``source`` (a JSONL file or a folder of them).

    Each input file appends to ``out_dir/<name>``; failures go to
    ``out_dir/failed/<name>`` and are retried on the next run. Paragraphs whose hash is
    already in the output (or earlier in this run) are skipped. ``settings.limit`` caps
    new paragraphs across all files.
    """
    budget = settings.limit
    tallies: list[Tally] = []
    for file in input_files(source):
        if budget is not None and budget <= 0:
            break
        out = out_dir / file.name
        out.parent.mkdir(parents=True, exist_ok=True)
        seen = done_hashes(out)
        paragraphs = list(read_paragraphs(file))
        tally = Tally(total=len(paragraphs))
        todo = _new_paragraphs(paragraphs, seen, tally)
        if budget is not None:
            todo = todo[:budget]
            budget -= len(todo)
        tallies.append(tally)
        started = clock()
        failed_path = out_dir / FAILED_DIR / file.name
        with out.open("a", encoding="utf-8", newline="\n") as sink:
            for outcome in annotate_concurrently(todo, annotate, concurrency=settings.concurrency):
                tally.add(outcome)
                _write(outcome, sink, failed_path)
                if tally.done % settings.log_every == 0 and tally.done < len(todo):
                    log(format_progress(file.name, tally, elapsed=clock() - started))
        log(format_progress(file.name, tally, elapsed=clock() - started))
    return tallies


def _new_paragraphs(
    paragraphs: Sequence[RawParagraph], seen: set[str], tally: Tally
) -> list[RawParagraph]:
    todo: list[RawParagraph] = []
    for paragraph in paragraphs:
        digest = paragraph_hash(paragraph.text)
        if digest in seen:
            tally.skipped += 1
            continue
        seen.add(digest)
        todo.append(paragraph)
    return todo


def _write(outcome: Outcome, sink: TextIO, failed_path: Path) -> None:
    match outcome:
        case Annotated(paragraph):
            sink.write(paragraph.model_dump_json() + "\n")
            sink.flush()
        case Failed(failure):
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            with failed_path.open("a", encoding="utf-8", newline="\n") as log_file:
                log_file.write(failure.model_dump_json() + "\n")
        case _:
            assert_never(outcome)
