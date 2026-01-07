"""Classes for making LLM requests."""

from collections.abc import Iterable
import dataclasses
from typing import Protocol

from openai.types.chat import chat_completion as chat_completion_type
from openai.types.chat import chat_completion_message_param


@dataclasses.dataclass(frozen=True)
class LLMRequest:
    messages: Iterable[chat_completion_message_param.ChatCompletionMessageParam]
    temperature: float = 1.0


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: chat_completion_type.ChatCompletion
    cost: float


# TODO: Move this to its own module.
class LLMClient(Protocol):
    # TODO: expand the request/response model for LLM reqs.
    async def __call__(self, request: LLMRequest) -> LLMResponse: ...
