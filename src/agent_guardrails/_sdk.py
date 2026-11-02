"""Helpers for tools built with the OpenAI Agents SDK, without importing it.

The SDK's ``@function_tool`` runs a tool body inside a failure handler that
turns every exception into a returned string ("An error occurred while
running the tool..."). A guard that calls the tool sees a normal result: a
write that failed reads as a success and gets recorded and replayed, and a
read that failed gets cached for the rest of the run.

:func:`without_failure_handling` returns the tool's invoker with that handler
turned off, so the guard sees the exception and decides what it means, plus
the tool's own formatter so a read's error can still be shown the SDK's way.
Without the SDK installed (the core runs without it), tools are used as they
are.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from typing import Any

Invoker = Callable[[Any, str], Awaitable[Any]]
Formatter = Callable[[Any, BaseException], Awaitable[str | None]]


def without_failure_handling(tool: Any) -> tuple[Invoker, Formatter | None]:
    """``(invoker, formatter)`` for ``tool``.

    For an SDK ``FunctionTool`` the invoker raises the body's exceptions, and
    ``formatter(ctx, exc)`` returns what the tool would have shown the model
    (``None`` when the tool was built to raise). For anything else the
    invoker is the tool's own and the formatter is ``None``.
    """
    inner: Invoker = tool.on_invoke_tool
    try:
        from agents.tool import (
            FunctionTool,
            maybe_invoke_function_tool_failure_error_function,
            set_function_tool_failure_error_function,
        )
    except ImportError:  # pragma: no cover - the core runs without the SDK
        return inner, None
    if not isinstance(tool, FunctionTool):
        return inner, None

    raw = copy.copy(tool)  # the copy's invoker is rebound to the copy
    set_function_tool_failure_error_function(raw, None)

    async def formatter(ctx: Any, exc: BaseException) -> str | None:
        return await maybe_invoke_function_tool_failure_error_function(
            function_tool=tool, context=ctx, error=exc
        )

    return raw.on_invoke_tool, formatter


#: Set on a tool that ``gate_tool`` wrapped. Every guard treats such a tool as
#: a write, whatever its own classifier says, so a budget can never cache it.
GATED_MARKER = "_agent_guardrails_gated"


def is_gated(tool: Any) -> bool:
    return bool(getattr(tool, GATED_MARKER, False))


__all__ = ["GATED_MARKER", "Formatter", "Invoker", "is_gated", "without_failure_handling"]
