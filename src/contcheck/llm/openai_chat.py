"""``LLMClient`` over an OpenAI-compatible ``/chat/completions`` API (Ollama, llama.cpp, vLLM).

Tested against Ollama 0.24, whose OpenAI-compatible endpoint ignores Ollama's own
``think`` flag but honours ``reasoning_effort: "none"``.
"""

import threading
import time
from collections.abc import Callable, Sequence

import httpx
from pydantic import ValidationError

from contcheck.llm.models import ChatCompletion, ChatMessage, ChatOptions, ClientConfig, LLMError

COMPLETIONS_PATH = "chat/completions"
RETRY_AFTER = "Retry-After"
TOO_MANY_REQUESTS = 429


class OpenAIChatClient:
    """Sends each chat to ``{base_url}/chat/completions`` and returns the first choice.

    Retries connection errors, timeouts, 429 and 5xx replies with exponential backoff (or
    the server's ``Retry-After``); other 4xx replies fail at once. A semaphore keeps at
    most ``config.concurrency`` requests in flight across threads.
    """

    def __init__(
        self,
        http: httpx.Client,
        config: ClientConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._config = config
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(config.concurrency)
        self._url = f"{config.base_url.rstrip('/')}/{COMPLETIONS_PATH}"

    def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        """The model's reply text to ``messages``; raises ``LLMError`` when retries run out."""
        body = request_body(self._config.model, messages, options)
        headers = auth_headers(self._config.api_key)
        last_problem = "no attempt made"
        retry_after: float | None = None
        for attempt in range(self._config.retries + 1):
            if attempt:
                self._sleep(self._delay(attempt - 1, retry_after))
            with self._slots:
                try:
                    response = self._http.post(
                        self._url, json=body, headers=headers, timeout=self._config.timeout
                    )
                except httpx.TransportError as error:
                    last_problem = f"{type(error).__name__}: {error}"
                    retry_after = None
                    continue
            if _retryable(response.status_code):
                last_problem = _describe(response)
                retry_after = _retry_after(response)
                continue
            if response.is_error:
                raise LLMError(_describe(response))
            return _reply_text(response)
        raise LLMError(f"gave up after {self._config.retries + 1} attempts: {last_problem}")

    def _delay(self, retry: int, retry_after: float | None) -> float:
        wait = self._config.backoff * 2.0**retry if retry_after is None else retry_after
        return min(wait, self._config.max_backoff)


def request_body(
    model: str, messages: Sequence[ChatMessage], options: ChatOptions
) -> dict[str, object]:
    """The JSON body of a ``/chat/completions`` request."""
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "temperature": options.temperature,
        "stream": False,
    }
    if options.max_tokens is not None:
        body["max_tokens"] = options.max_tokens
    if options.json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": options.schema_name, "schema": dict(options.json_schema)},
        }
    if not options.think:
        body["reasoning_effort"] = "none"
    return body


def auth_headers(api_key: str | None) -> dict[str, str]:
    """``Authorization: Bearer`` when a key is set, else nothing."""
    return {} if api_key is None or api_key == "" else {"Authorization": f"Bearer {api_key}"}


def _retryable(status: int) -> bool:
    return status == TOO_MANY_REQUESTS or status >= 500


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get(RETRY_AFTER)
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:
        return None


def _describe(response: httpx.Response) -> str:
    return f"HTTP {response.status_code}: {response.text[:200]}"


def _reply_text(response: httpx.Response) -> str:
    try:
        return ChatCompletion.model_validate_json(response.content).text
    except ValidationError as error:
        raise LLMError(f"unexpected reply: {response.text[:200]}") from error
