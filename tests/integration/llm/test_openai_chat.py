"""Talks to a real OpenAI-compatible server named by ``CONTCHECK_LLM_BASE_URL``.

Run with ``CONTCHECK_LLM_BASE_URL=http://localhost:11434/v1 uv run pytest -m llm``;
``CONTCHECK_LLM_MODEL`` picks the model (default ``qwen3:8b``).
"""

import json
import os

import httpx
import pytest

from contcheck.llm.models import ChatMessage, ChatOptions, ClientConfig, LLMError, Role
from contcheck.llm.openai_chat import OpenAIChatClient

pytestmark = pytest.mark.llm


@pytest.fixture
def config() -> ClientConfig:
    base_url = os.environ.get("CONTCHECK_LLM_BASE_URL", "")
    if not base_url:
        pytest.skip("CONTCHECK_LLM_BASE_URL is not set")
    key = os.environ.get("CONTCHECK_LLM_API_KEY", "")
    return ClientConfig(
        base_url=base_url,
        model=os.environ.get("CONTCHECK_LLM_MODEL", "qwen3:8b"),
        api_key=key if key else None,
        retries=1,
    )


def test_replies_with_schema_constrained_json_without_reasoning(config: ClientConfig) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    messages = [ChatMessage(Role.USER, 'What is 2 + 3? Reply as JSON: {"answer": <number>}')]

    with httpx.Client() as http:
        reply = OpenAIChatClient(http, config).complete(
            messages, options=ChatOptions(json_schema=schema, max_tokens=50)
        )

    assert "<think>" not in reply
    assert json.loads(reply) == {"answer": 5}


def test_unknown_model_fails_without_retrying(config: ClientConfig) -> None:
    missing = ClientConfig(base_url=config.base_url, model="no-such-model:0b", retries=0)

    with httpx.Client() as http, pytest.raises(LLMError):
        OpenAIChatClient(http, missing).complete(
            [ChatMessage(Role.USER, "hi")], options=ChatOptions(max_tokens=5)
        )
