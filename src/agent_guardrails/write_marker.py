"""A human-approved write must not be cut off by the run's own deadline.

The soft stops in this package (the soft deadline, the refusal ceiling, the
cost ceiling) exist to end a run that is reading unproductively and to leave
time to write an answer. Every one of them is wrong in the window between a
person approving a write and that write completing. Stopping there leaves
nobody able to say whether the change happened, and an ambiguous write is
never retried by design (a duplicate is worse than a gap), so the user is left
with an unknown.

So while a write is in flight:

* the soft stops are suppressed;
* the hard deadline is stretched by ``WRITE_GRACE_MULTIPLIER``, not removed.
  A genuinely hung write still has to die, or one stuck HTTP call pins a
  worker forever.

Two details that are easy to get wrong:

* The marker must span the ``await``, not the dispatch. The write happens
  while the coroutine is suspended; a flag set and cleared around the call is
  false for the whole window it exists to describe.
* The state is a mutable dict reached through a ``ContextVar``, never a plain
  ``ContextVar.set``. The write may execute inside a nested agent-as-tool run,
  whose task copies the context; a ``set`` in there is invisible to the
  watchdog polling in the parent, a mutation of the shared dict is not.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar, Token
from typing import Any

#: How much longer than the hard deadline a run may take while a write is in
#: flight. Bounded on purpose: a write deserves room to finish, not immunity.
WRITE_GRACE_MULTIPLIER = 1.5

_write_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_guardrails_write_state", default=None
)


def bind_write_marker() -> tuple[dict[str, Any], Token[dict[str, Any] | None]]:
    """Bind a per-run marker. Returns the shared dict and its reset token."""
    state: dict[str, Any] = {"in_flight": 0, "protected": []}
    return state, _write_state.set(state)


def unbind_write_marker(token: Token[dict[str, Any] | None]) -> None:
    with contextlib.suppress(ValueError):
        _write_state.reset(token)


def write_in_flight() -> bool:
    """True while at least one approved write is executing in this run."""
    state = _write_state.get()
    return bool(state and state.get("in_flight", 0) > 0)


@contextlib.contextmanager
def executing_write(action: str) -> Iterator[None]:
    """Mark a write as in flight for the whole of its ``await``.

    A counter, not a boolean: the SDK runs a turn's tool calls concurrently,
    so two writes can overlap, and a boolean would be cleared by whichever
    finished first while the other was still running.

    Unbound (a script, a test, a worker with no run) it is a no-op.
    """
    state = _write_state.get()
    if state is None:
        yield
        return
    state["in_flight"] = state.get("in_flight", 0) + 1
    # Recorded so a run that was stretched says why. A long run with a
    # protected write is not the same finding as a regression.
    state["protected"] = sorted({*state.get("protected", []), action})
    try:
        yield
    finally:
        state["in_flight"] = max(0, state.get("in_flight", 0) - 1)


__all__ = [
    "WRITE_GRACE_MULTIPLIER",
    "bind_write_marker",
    "executing_write",
    "unbind_write_marker",
    "write_in_flight",
]
