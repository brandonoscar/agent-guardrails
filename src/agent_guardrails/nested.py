"""Caps on nested agent consults (an agent exposed to another as a tool).

A consult is the most expensive call a run can make: a whole nested agent
with its own turns and its own tool calls. Two failure shapes, both seen in
real runs and both untouched by prompt wording:

* **Over-consulting a one-fact question.** The orchestrator asks the same
  specialist again and again for something one consult answers. Docstring
  rewrites, result footers and "triage first" instructions each moved the
  numbers a little and none bound them. ``NestedLedger`` does: past ``cap``
  consults in a run, the next one is refused with "answer now".
* **Consulting after time is up.** Past the soft deadline a consult cannot
  finish, and it is not free to fail: the call budget is a ``ContextVar`` that
  propagates into the nested run and refuses every read it makes, the nested
  agent reports "the lookup was cut off", the caller reads that as a transient
  failure and consults again. The deadline check runs first, on every run,
  whether or not a cap is bound.

Like every cap here, the slot is reserved before the ``await``: the SDK runs
a turn's tool calls concurrently.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, TypeVar

from agent_guardrails.budget import current_call_ledger, enter_nested, leave_nested

logger = logging.getLogger(__name__)

ToolT = TypeVar("ToolT")

#: Consults a run may make when a cap is bound. Two, so a question that
#: needs a second opinion still gets one, and the third is the one that says
#: "you are looping".
DEFAULT_NESTED_CAP = 2

REFUSAL = (
    "STOP CONSULTING: this run has already consulted {used} nested agent(s), and the "
    "cap is {cap}. Do not call another. The next thing you produce must be the answer, "
    "written from what you have already gathered. If you cannot answer, say so plainly "
    "and name what is missing."
)

NO_TIME_FOR_A_CONSULT = (
    "NOT CONSULTED: this run is out of time, so the nested agent was not called. A "
    "consult is a whole nested run and cannot finish now; starting one spends the "
    "remaining seconds and returns nothing. Write the answer from what you already "
    "have and say plainly which part you could not check. Another consult will be "
    "refused the same way."
)


@dataclass
class NestedLedger:
    """One run's consults."""

    cap: int = DEFAULT_NESTED_CAP
    #: Consults allowed to run, in order.
    allowed: list[str] = field(default_factory=list)
    #: Consults refused after the cap bound, by tool name.
    refused: dict[str, int] = field(default_factory=dict)
    #: Consults refused because the run was out of time.
    refused_for_time: int = 0

    @property
    def used(self) -> int:
        return len(self.allowed)

    def reserve(self, name: str) -> str | None:
        """Take a slot, or return the refusal to hand back to the model.

        Mutates before the caller awaits anything, so concurrent consults in
        one turn cannot all pass a check that was true when each read it.
        """
        if self.used >= self.cap:
            self.refused[name] = self.refused.get(name, 0) + 1
            return REFUSAL.format(used=self.used, cap=self.cap)
        self.allowed.append(name)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "consults": self.used,
            "trail": list(self.allowed),
            "refused": dict(self.refused),
            "refused_for_time": self.refused_for_time,
            "hit_cap": self.used >= self.cap and bool(self.refused),
        }


_nested: ContextVar[NestedLedger | None] = ContextVar(
    "agent_guardrails_nested_ledger", default=None
)


def bind_nested_ledger(
    cap: int | None = DEFAULT_NESTED_CAP,
) -> tuple[NestedLedger | None, Token[NestedLedger | None]]:
    """Bind a consult cap for one run. ``cap=None`` binds no cap (the deadline
    check still applies). Always returns a token to reset with."""
    ledger = NestedLedger(cap=cap) if cap is not None else None
    return ledger, _nested.set(ledger)


def reset_nested_ledger(token: Token[NestedLedger | None]) -> None:
    try:
        _nested.reset(token)
    except ValueError:  # pragma: no cover - token from another context
        logger.debug("nested ledger token belonged to another context")


def current_nested_ledger() -> NestedLedger | None:
    return _nested.get()


def cap_nested_tool(tool: ToolT) -> ToolT:
    """A copy of an agent-as-tool whose consults obey the run's caps.

    Works on any object with ``name`` and an async ``on_invoke_tool``, such as
    the SDK's ``agent.as_tool(...)``. Exceptions from the nested run propagate
    unchanged.
    """
    name = str(getattr(tool, "name", "") or "")
    inner = getattr(tool, "on_invoke_tool", None)
    if not callable(inner):
        return tool

    async def invoke(ctx: Any, payload: str) -> Any:
        # The deadline comes first and applies to every run.
        ledger = current_call_ledger()
        if ledger is not None and ledger.out_of_time:
            nested = _nested.get()
            if nested is not None:
                nested.refused_for_time += 1
            logger.info("consult %s refused: the run is out of time", name)
            return NO_TIME_FOR_A_CONSULT
        nested = _nested.get()
        if nested is not None:
            refusal = nested.reserve(name)
            if refusal is not None:
                logger.info("consult %s refused: cap %d reached", name, nested.cap)
                return refusal
        # The marker wraps the await: the nested run's reads happen while we
        # are suspended, and they need to know who will read their refusals.
        token = enter_nested()
        try:
            return await inner(ctx, payload)
        finally:
            leave_nested(token)

    wrapped = copy.copy(tool)
    wrapped.on_invoke_tool = invoke  # type: ignore[attr-defined]
    return wrapped


def cap_nested_tools(tools: Sequence[ToolT]) -> list[ToolT]:
    return [cap_nested_tool(t) for t in tools]


__all__ = [
    "DEFAULT_NESTED_CAP",
    "NO_TIME_FOR_A_CONSULT",
    "REFUSAL",
    "NestedLedger",
    "bind_nested_ledger",
    "cap_nested_tool",
    "cap_nested_tools",
    "current_nested_ledger",
    "reset_nested_ledger",
]
