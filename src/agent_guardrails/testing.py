"""A scripted model for the OpenAI Agents SDK: test guarded tools offline.

``ScriptedModel`` replays a list of turns instead of calling an LLM, so the
real ``Runner`` executes your real tools, with no API key and no network.
Each turn is what the model "says": tool calls (:func:`call`), a final
message (:func:`say`), or both. When the script runs out it says
``"(script ended)"``.

    model = ScriptedModel([
        [call("send_email", to="ops@example.com", subject="Hi", body="...")],
        [say("Sent.")],
    ], input_tokens=1_000, output_tokens=100)
    agent = Agent(name="assistant", model=model, tools=[...])
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agents.items import ModelResponse, TResponseInputItem, TResponseOutputItem
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

_ids = itertools.count(1)

Turn = Sequence[TResponseOutputItem]


def call(name: str, /, **arguments: Any) -> ResponseFunctionToolCall:
    """One tool call the model makes."""
    n = next(_ids)
    return ResponseFunctionToolCall(
        type="function_call",
        id=f"fc_{n}",
        call_id=f"call_{n}",
        name=name,
        arguments=json.dumps(arguments),
    )


def say(text: str) -> ResponseOutputMessage:
    """A message from the model. A turn with only messages ends the run."""
    return ResponseOutputMessage(
        type="message",
        id=f"msg_{next(_ids)}",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class ScriptedModel(Model):
    """Replays scripted turns. Every turn reports the same token usage, which
    makes cost-ceiling tests deterministic."""

    def __init__(
        self,
        turns: Sequence[Turn],
        *,
        name: str = "scripted",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.model = name
        self._turns = list(turns)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        #: How many turns the runner asked for.
        self.calls = 0

    def _next(self) -> list[TResponseOutputItem]:
        self.calls += 1
        if self._turns:
            return list(self._turns.pop(0))
        return [say("(script ended)")]

    def _usage(self) -> Usage:
        return Usage(
            requests=1,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        return ModelResponse(output=self._next(), usage=self._usage(), response_id=None)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        response = Response(
            id=f"resp_{next(_ids)}",
            created_at=time.time(),
            model=self.model,
            object="response",
            output=self._next(),
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            usage=ResponseUsage(
                input_tokens=self.input_tokens,
                input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
                output_tokens=self.output_tokens,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=self.input_tokens + self.output_tokens,
            ),
        )
        yield ResponseCompletedEvent(
            type="response.completed", response=response, sequence_number=0
        )


__all__ = ["ScriptedModel", "Turn", "call", "say"]
