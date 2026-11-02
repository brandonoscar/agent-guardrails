"""``RunGuard``: bind every per-run guard for one agent run in one place.

Bind it BEFORE starting the run. The SDK runs the agent loop in its own task,
which copies the context when it is created, so a guard bound afterwards is
invisible to the run. The guards are shared mutable objects, so what the run
records (inside nested agent runs too) is visible here afterwards.

    with RunGuard(scope="user-42", deadline_s=240, cost_ceiling_usd=1.00) as run:
        result = Runner.run_streamed(agent, question)
        async for event in watch_run(result, run):
            ...
    log.info("run summary %s", run.summary())
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, TypeVar

from agent_guardrails.budget import (
    BudgetPolicy,
    CallLedger,
    bind_call_ledger,
    reset_call_ledger,
)
from agent_guardrails.confirm import (
    ConfirmBudget,
    ConfirmChannel,
    Emitter,
    bind_confirm_budget,
    reset_confirm_budget,
)
from agent_guardrails.cost import RunUsage, bind_run_usage, reset_run_usage
from agent_guardrails.nested import NestedLedger, bind_nested_ledger, reset_nested_ledger
from agent_guardrails.watchdog import watch_stream
from agent_guardrails.write_marker import bind_write_marker, unbind_write_marker

T = TypeVar("T")

_scope: ContextVar[str | None] = ContextVar("agent_guardrails_scope", default=None)


def current_scope() -> str | None:
    """The scope (customer, account, user) bound for this run, if any."""
    return _scope.get()


class RunGuard:
    """Per-run guards: approval budget, call budget, nested cap, cost and the
    write marker. Use as a context manager around one run.

    ``scope`` isolates idempotency records (results never replay across it).
    ``deadline_s`` is the run's hard deadline in seconds; the soft deadline
    falls at ``budget.soft_deadline_fraction`` of it. ``nested_cap`` caps
    agent-as-tool consults (``None``: no cap). ``channel`` and ``emitter``
    route this run's approval cards.
    """

    def __init__(
        self,
        *,
        scope: str | None = None,
        deadline_s: float | None = None,
        cost_ceiling_usd: float | None = None,
        budget: BudgetPolicy | None = None,
        nested_cap: int | None = None,
        channel: ConfirmChannel | None = None,
        emitter: Emitter | None = None,
        expired_gate_ceiling: int = 1,
    ) -> None:
        if emitter is not None and channel is None:
            raise ValueError("an emitter needs the channel it emits for")
        self.scope = scope
        self.deadline_s = deadline_s
        self._policy = budget or BudgetPolicy()
        self._cost_ceiling = cost_ceiling_usd
        self._nested_cap = nested_cap
        self._channel = channel
        self._emitter = emitter
        self._expired_gate_ceiling = expired_gate_ceiling
        self._tokens: list[tuple[str, Token[Any]]] = []
        self.calls: CallLedger | None = None
        self.usage: RunUsage | None = None
        self.nested: NestedLedger | None = None
        self.approvals: ConfirmBudget | None = None
        self.writes: dict[str, Any] | None = None

    def __enter__(self) -> RunGuard:
        if self._tokens:
            raise RuntimeError("a RunGuard guards one run; make a new one")
        self._tokens.append(("scope", _scope.set(self.scope)))
        # Approvals first: the call budget's clock leaves out the time a
        # person spends on a card, so it needs the run's approval budget.
        self.approvals, approvals_token = bind_confirm_budget(
            ConfirmBudget(expired_ceiling=self._expired_gate_ceiling)
        )
        self._tokens.append(("approvals", approvals_token))
        self.calls, token = bind_call_ledger(
            self._policy, deadline_s=self.deadline_s, confirm=self.approvals
        )
        self._tokens.append(("calls", token))
        self.usage, usage_token = bind_run_usage(self._cost_ceiling)
        self._tokens.append(("usage", usage_token))
        self.nested, nested_token = bind_nested_ledger(self._nested_cap)
        self._tokens.append(("nested", nested_token))
        self.writes, writes_token = bind_write_marker()
        self._tokens.append(("writes", writes_token))
        if self._channel is not None and self._emitter is not None:
            self._tokens.append(("emitter", self._channel.bind_emitter(self._emitter)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        resets: dict[str, Callable[[Any], None]] = {
            "scope": _scope.reset,
            "calls": reset_call_ledger,
            "usage": reset_run_usage,
            "nested": reset_nested_ledger,
            "approvals": reset_confirm_budget,
            "writes": unbind_write_marker,
        }
        while self._tokens:
            kind, token = self._tokens.pop()
            if kind == "emitter":
                assert self._channel is not None
                self._channel.reset_emitter(token)
            else:
                resets[kind](token)

    def watch(self, stream: AsyncIterable[T], **kwargs: Any) -> AsyncIterator[T]:
        """``watch_stream`` with this run's deadline and soft fraction."""
        kwargs.setdefault("max_wall_clock", self.deadline_s)
        kwargs.setdefault("soft_fraction", self._policy.soft_deadline_fraction)
        return watch_stream(stream, **kwargs)

    def summary(self) -> dict[str, Any]:
        """One dict for your run log: what every guard did."""
        return {
            "scope": self.scope,
            "calls": self.calls.summary() if self.calls else None,
            "usage": self.usage.summary() if self.usage else None,
            "nested": self.nested.summary() if self.nested else None,
            "approvals": self.approvals.summary() if self.approvals else None,
            "protected_writes": list(self.writes.get("protected", [])) if self.writes else [],
        }


__all__ = ["RunGuard", "current_scope"]
