"""Interfaces for chat-completion LLMs."""

from collections.abc import Sequence
from typing import Protocol

from contcheck.llm.models import ChatMessage, ChatOptions


class LLMClient(Protocol):
    """Answers a chat with one reply.

    Implementations are safe to call from several threads at once and limit how many
    requests run in parallel themselves.
    """

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``.

        Raises ``LLMError`` when the server gives no usable reply after retries.
        """
        ...
