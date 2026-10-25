import json

from contcheck.corpus.annotate import Parsed, parse_reply
from contcheck.corpus.annotate_prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_messages
from contcheck.corpus.annotated import ProposedChange
from contcheck.llm.models import Role
from contcheck.taxonomy.kinds import Formality


def test_messages_hold_the_instructions_then_the_paragraph() -> None:
    messages = build_messages("A paragraph.")

    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].content.endswith("A paragraph.")


def test_schema_requires_the_fields_of_a_proposed_change() -> None:
    changes = RESPONSE_SCHEMA["properties"]
    assert isinstance(changes, dict)
    items = changes["changes"]["items"]

    assert items["required"] == sorted(ProposedChange.model_fields)
    assert items["properties"]["type"]["enum"] == [f.value for f in Formality]


def test_example_replies_in_the_prompt_validate() -> None:
    replies = [
        line.removeprefix("Example reply:").strip()
        for line in SYSTEM_PROMPT.splitlines()
        if line.startswith("{")
    ]

    assert len(replies) >= 2
    for reply in replies[1:]:
        assert isinstance(parse_reply(reply), Parsed), reply
        json.loads(reply)
