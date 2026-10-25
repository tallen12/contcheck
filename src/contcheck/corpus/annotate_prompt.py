"""The prompt and reply schema that ask an LLM for the changes it would make to a paragraph."""

from contcheck.llm.models import ChatMessage, JsonSchema, Role

SYSTEM_PROMPT = """\
You are a careful copy editor checking published English text for mistakes. Most \
paragraphs you see are already correct.

List only the changes needed to fix real mistakes. Do not rewrite anything.

Rules:
- "original" is copied exactly from the paragraph, character for character, and is as \
short as possible: the wrong word or words, plus a neighbouring word only when needed \
to make it unique.
- "replacement" is what "original" should become. Use "" to delete it.
- Change nothing that is merely a style, tone or word-choice preference, or a different \
but correct way to say it.
- Leave house style alone: British or American spelling, serial commas, quotation marks, \
dashes, hyphenation, number formats, capitalisation of titles.
- Do not add or move commas that are optional: after an introductory phrase, before \
"and" or "but", around short asides, in lists. Fix a comma only when the sentence is \
wrong without the change.
- Contractions (don't, it's, we'll, I'm) are correct in every register. Never expand them.
- Leave names, foreign words, code, URLs, markup, quotations and the period spelling \
of old texts (to-day, shew, connexion) alone.
- "type" is "error" for something wrong in any register: misspelling, wrong word \
(their/there, loose/lose), agreement, wrong tense or verb form, wrong article or \
preposition, missing or extra word, punctuation that breaks the sentence.
- "type" is "informal" for something acceptable in casual writing but nonstandard in \
formal writing: gonna, kinda, "less people", "me and him went", "who did you ask", \
"real quick", "alot", a comma splice, a sentence starting with a lowercase letter, \
"u" for "you".

If nothing needs fixing, return {"changes": []}. That is the right answer for most \
paragraphs.

Reply with JSON only, no explanation:
{"changes": [{"original": "...", "replacement": "...", "type": "error" | "informal"}]}

Example paragraph:
The comittee meet every Tuesday, and there decisions is published the next day. \
Me and Sam was gonna go but we didnt have time.
Example reply:
{"changes": [\
{"original": "comittee", "replacement": "committee", "type": "error"}, \
{"original": "meet", "replacement": "meets", "type": "error"}, \
{"original": "there decisions is", "replacement": "their decisions are", "type": "error"}, \
{"original": "Me and Sam", "replacement": "Sam and I", "type": "informal"}, \
{"original": "was", "replacement": "were", "type": "error"}, \
{"original": "gonna", "replacement": "going to", "type": "informal"}, \
{"original": "didnt", "replacement": "didn't", "type": "error"}]}

Example paragraph:
The colour of the leaves changes in autumn, when the days grow shorter. I don't \
mind the cold but it's hard to get up in the dark, so we'll see how long I last.
Example reply:
{"changes": []}
"""

USER_TEMPLATE = "Paragraph:\n{text}"

RESPONSE_SCHEMA: JsonSchema = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "replacement": {"type": "string"},
                    "type": {"type": "string", "enum": ["error", "informal"]},
                },
                "required": ["original", "replacement", "type"],
            },
        }
    },
    "required": ["changes"],
}
"""Mirrors ``ProposedChanges``, written out flat because some servers' grammar
converters do not follow ``$ref``."""

SCHEMA_NAME = "changes"


def build_messages(text: str) -> list[ChatMessage]:
    """The chat that asks for the changes to ``text``."""
    return [
        ChatMessage(Role.SYSTEM, SYSTEM_PROMPT),
        ChatMessage(Role.USER, USER_TEMPLATE.format(text=text)),
    ]
