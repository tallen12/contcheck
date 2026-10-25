from collections.abc import Sequence

from contcheck.llm.models import ChatMessage, ChatOptions, Role
from contcheck.llm.protocols import LLMClient


class EchoClient:
    """Replies with the last message's text."""

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        return messages[-1].content


def test_client_contract() -> None:
    client: LLMClient = EchoClient()

    assert client.complete([ChatMessage(Role.USER, "hi")], options=ChatOptions()) == "hi"
