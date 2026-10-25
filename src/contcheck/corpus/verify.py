"""Check each error change ``corpus annotate`` proposed with a second LLM pass.

On a hand-reviewed sample about one applied change in nine made correct text wrong
("legs" -> "ledge", "there was a leak" -> "were"). The checker sees each change's
sentence before and after it and keeps a change only when the original is a real
mistake, the result is correct and the meaning is kept. Informal changes are passed
through unchecked, since ``corpus correct`` does not apply them.

The output has the annotate format, so ``corpus correct`` reads it unchanged: only the
confirmed changes stay in ``changes``, the others move to ``rejected`` with the reason.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from pydantic import BaseModel, ConfigDict, ValidationError

from contcheck.corpus.annotate import (
    DEFAULT_ATTEMPTS,
    REPLY_EXCERPT,
    RETRY_TEMPERATURE_STEP,
    apply_changes,
    extract_json,
)
from contcheck.corpus.annotated import (
    Annotated,
    AnnotatedParagraph,
    Change,
    Failed,
    FailedParagraph,
    FailureReason,
    Outcome,
    RejectedChange,
)
from contcheck.corpus.verify_prompt import RESPONSE_SCHEMA, SCHEMA_NAME, build_messages
from contcheck.llm.models import ChatOptions, LLMError
from contcheck.llm.protocols import LLMClient
from contcheck.taxonomy.kinds import Formality


class Verdict(BaseModel):
    """The checker's answer for one change, numbered from 1 in prompt order."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: int
    reason: str
    original_is_wrong: bool
    result_is_correct: bool
    keeps_meaning: bool

    @property
    def confirmed(self) -> bool:
        """Whether the change fixes a real mistake without breaking anything."""
        return self.original_is_wrong and self.result_is_correct and self.keeps_meaning


class Verdicts(BaseModel):
    """The checker's whole reply."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True, slots=True)
class Judged:
    """Exactly one verdict per change, in change order."""

    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True, slots=True)
class Unusable:
    """The reply cannot be used."""

    reason: FailureReason
    detail: str


def parse_verdicts(reply: str, *, count: int) -> Judged | Unusable:
    """Validate a reply and match its verdicts to changes ``1..count``."""
    try:
        value: object = json.loads(extract_json(reply))
    except json.JSONDecodeError as error:
        return Unusable(FailureReason.INVALID_JSON, f"not JSON: {error}")
    if isinstance(value, list):
        value = {"verdicts": value}
    try:
        verdicts = Verdicts.model_validate(value).verdicts
    except ValidationError as error:
        return Unusable(FailureReason.INVALID_JSON, f"schema: {error.errors()[0]['msg']}")
    by_id = {verdict.id: verdict for verdict in verdicts}
    expected = set(range(1, count + 1))
    if len(by_id) != len(verdicts) or set(by_id) != expected:
        ids = sorted(verdict.id for verdict in verdicts)
        return Unusable(FailureReason.INCOMPLETE, f"ids {ids}, expected 1..{count}")
    return Judged(tuple(by_id[i] for i in range(1, count + 1)))


def apply_verdicts(
    paragraph: AnnotatedParagraph,
    checked: Sequence[Change],
    verdicts: Sequence[Verdict],
    *,
    verifier: str,
) -> AnnotatedParagraph:
    """``paragraph`` with the unconfirmed ``checked`` changes moved to ``rejected``.

    Verifying an already verified paragraph checks only the changes still kept, adds to
    ``rejected`` and lists both models in ``verifier``, so running ``corpus verify`` a
    second time with another model keeps only changes both confirmed.
    """
    rejected = [
        RejectedChange(change=change, reason=verdict.reason)
        for change, verdict in zip(checked, verdicts, strict=True)
        if not verdict.confirmed
    ]
    dropped = {item.change for item in rejected}
    kept = tuple(change for change in paragraph.changes if change not in dropped)
    return paragraph.model_copy(
        update={
            "changes": kept,
            "corrected": apply_changes(paragraph.text, kept),
            "verifier": verifier
            if paragraph.verifier is None
            else f"{paragraph.verifier}, {verifier}",
            "rejected": (*paragraph.rejected, *rejected),
        }
    )


class ChangeVerifier:
    """Asks the model for a verdict on each error change of a paragraph.

    Attempt ``n`` (from 0) samples at temperature ``n * 0.3``, as annotation does.
    Paragraphs without error changes are passed through without a request.
    """

    def __init__(self, client: LLMClient, *, model: str, attempts: int = DEFAULT_ATTEMPTS) -> None:
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        self._client = client
        self._model = model
        self._attempts = attempts

    def verify(self, paragraph: AnnotatedParagraph) -> Outcome:
        """Verify one paragraph; never raises for a bad reply or an unreachable server."""
        checked = tuple(c for c in paragraph.changes if c.type is Formality.ERROR)
        if not checked:
            return Annotated(apply_verdicts(paragraph, (), (), verifier=self._model))
        messages = build_messages(paragraph.text, checked)
        problem = Unusable(FailureReason.REQUEST, "no attempt made")
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
                problem = Unusable(FailureReason.REQUEST, str(error))
                return Failed(self._failure(paragraph, problem, attempt + 1, ""))
            result = parse_verdicts(reply, count=len(checked))
            match result:
                case Judged(verdicts):
                    return Annotated(
                        apply_verdicts(paragraph, checked, verdicts, verifier=self._model)
                    )
                case Unusable():
                    problem = result
                case _:
                    assert_never(result)
        return Failed(self._failure(paragraph, problem, self._attempts, reply))

    def _failure(
        self, paragraph: AnnotatedParagraph, problem: Unusable, attempts: int, reply: str
    ) -> FailedParagraph:
        return FailedParagraph(
            source_id=paragraph.source_id,
            hash=paragraph.hash,
            reason=problem.reason,
            detail=problem.detail,
            reply=reply[:REPLY_EXCERPT],
            model=self._model,
            attempts=attempts,
        )
