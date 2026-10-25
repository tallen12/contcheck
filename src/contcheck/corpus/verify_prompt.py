"""The prompt and reply schema that ask an LLM to check proposed error changes.

The checker sees the paragraph and, for each change, its sentence before and after the
change, so it judges the result in context rather than the edit in isolation.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from contcheck.corpus.annotated import Change
from contcheck.llm.models import ChatMessage, JsonSchema, Role

SYSTEM_PROMPT = """\
You review corrections another editor proposed for published English text. Some \
proposals fix real mistakes. Others change text that was already correct, introduce a \
new mistake, or change what the writer meant. Approve only the first kind.

For each numbered change, answer three questions about it, in context:
- "original_is_wrong": is the original a real mistake in standard written English \
here? A style preference, a correct alternative wording, British or American spelling, \
an optional comma, a name, a technical term, a foreign word or a quotation is not a \
mistake.
- "result_is_correct": is the sentence after the change correct English? Check \
agreement with the real subject (not the nearest noun), articles, punctuation (no \
doubled marks), capitalisation at the start of a sentence, and that no word or phrase \
is now repeated or missing.
- "keeps_meaning": does the sentence after the change say what the writer meant? Facts, \
numbers, names and quotations must not change.

Write "reason" first: one short sentence on what the original and the change do. Then \
the three answers. Judge each change on its own.

Reply with JSON only:
{"verdicts": [{"id": 1, "reason": "...", "original_is_wrong": true, \
"result_is_correct": true, "keeps_meaning": true}]}

Example paragraph:
The list of winners were announced on Friday. Each of the three teams in the northern \
division plays twice. The ship's hull was repaired in the dry dock at Leith.
Example changes:
1. "were announced" -> "was announced"
   before: The list of winners were announced on Friday.
   after: The list of winners was announced on Friday.
2. "plays" -> "play"
   before: Each of the three teams in the northern division plays twice.
   after: Each of the three teams in the northern division play twice.
3. "hull" -> "hall"
   before: The ship's hull was repaired in the dry dock at Leith.
   after: The ship's hall was repaired in the dry dock at Leith.
Example reply:
{"verdicts": [\
{"id": 1, "reason": "The subject is 'list', singular, so 'were' is wrong.", \
"original_is_wrong": true, "result_is_correct": true, "keeps_meaning": true}, \
{"id": 2, "reason": "The subject is 'each', singular, so 'plays' was right.", \
"original_is_wrong": false, "result_is_correct": false, "keeps_meaning": true}, \
{"id": 3, "reason": "A ship has a hull; 'hall' is a different word.", \
"original_is_wrong": false, "result_is_correct": false, "keeps_meaning": false}]}
"""

USER_TEMPLATE = "Paragraph:\n{text}\n\nChanges:\n{changes}"

RESPONSE_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reason": {"type": "string"},
                    "original_is_wrong": {"type": "boolean"},
                    "result_is_correct": {"type": "boolean"},
                    "keeps_meaning": {"type": "boolean"},
                },
                "required": [
                    "id",
                    "reason",
                    "original_is_wrong",
                    "result_is_correct",
                    "keeps_meaning",
                ],
            },
        }
    },
    "required": ["verdicts"],
}
"""Mirrors ``Verdicts`` in ``verify``, written out flat like the annotate schema."""

SCHEMA_NAME = "verdicts"

SENTENCE_BREAK = re.compile("[.!?][\"'\u201d\u2019)\\]]*\\s+")


@dataclass(frozen=True, slots=True)
class SentencePair:
    """The sentence (or sentences) a change touches, before and after it."""

    before: str
    after: str


def sentence_around(text: str, change: Change) -> SentencePair:
    """The sentences of ``text`` that ``change`` touches, as written and as changed."""
    start = 0
    for match in SENTENCE_BREAK.finditer(text, 0, change.start):
        start = match.end()
    following = SENTENCE_BREAK.search(text, change.end)
    end = following.end() if following is not None else len(text)
    before = text[start:end].strip()
    after = (text[start : change.start] + change.replacement + text[change.end : end]).strip()
    return SentencePair(before, " ".join(after.split()))


def format_changes(text: str, changes: Sequence[Change]) -> str:
    """The numbered change list of the prompt, ids from 1."""
    blocks: list[str] = []
    for number, change in enumerate(changes, start=1):
        pair = sentence_around(text, change)
        blocks.append(
            f'{number}. "{change.original}" -> "{change.replacement}"\n'
            f"   before: {pair.before}\n"
            f"   after: {pair.after}"
        )
    return "\n".join(blocks)


def build_messages(text: str, changes: Sequence[Change]) -> list[ChatMessage]:
    """The chat that asks for a verdict on each of ``changes`` to ``text``."""
    return [
        ChatMessage(Role.SYSTEM, SYSTEM_PROMPT),
        ChatMessage(
            Role.USER, USER_TEMPLATE.format(text=text, changes=format_changes(text, changes))
        ),
    ]
