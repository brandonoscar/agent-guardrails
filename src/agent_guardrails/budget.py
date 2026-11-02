"""A run-wide budget on repeated reads: the structural anti-fan-out guard.

Wording does not hold. Rewriting a tool's description to steer the model off
record-by-record lookups tends to move the loop, not end it: the model swaps
one per-record tool for another and the run still dies on its deadline with no
answer. A cap the model cannot talk its way past is the fix; better prose is
not.

Per run, through a ``ContextVar``:

* **Memo.** An identical read (same tool, same arguments) returns the first
  call's answer instead of another round-trip, with a note saying so.
* **In-flight sharing.** An identical read already running is awaited, not
  repeated. The SDK issues a turn's tool calls concurrently, so duplicates
  arrive together and neither would see the other's memo entry.
* **Caps.** A capped tool (or group of tools) refuses after N distinct calls,
  and the refusal names the bulk tool that answers the question in one call.
  A run-wide ceiling sits above them, because nine tools each capped at eight
  is a 72-call budget, not a bound.
* **Refusal ceiling.** A refusal is free to the ledger and expensive on the
  clock: each one still costs a model turn. Past ``refusal_ceiling`` refusals
  the run is looping; the ledger reports ``looping`` and the stream watchdog
  stops the run (a sterner message alone gets ignored).
* **Soft deadline.** Past ``soft_deadline_fraction`` of the run's deadline,
  every budgeted read is refused with "stop reading and answer now", so the
  run has a turn left to write an answer instead of being killed mid-lookup.

Two properties this must never violate:

* **Writes are never memoized and never capped.** Their approval gate and
  idempotency guard own repeat-suppression, and a cap that could silence a
  write is a correctness hole, not a saving. Read-ness comes from the
  fail-closed classifier, so an unrecognized tool is treated as a write and
  left alone.
* **A write invalidates the memo.** A read-back after a write must see the new
  state, not the cached one. A read still in flight when a write starts or
  ends is neither cached nor shared, since it may have read the old state.
* **A failed read is never cached.** An SDK tool's error text is not data.

Reads are told apart by tool, not just by name: two different tools that
happen to share a name (a nested specialist's ``search`` and the
orchestrator's) never answer for each other.

The soft deadline counts active time: the seconds a person spends reading an
approval card don't use up the run's time to read, as they don't for the
watchdog.

A cap must reserve before it awaits. Check-then-act around an ``await`` lets
ten concurrent calls all read an empty ledger, all pass and all execute.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, TypeVar

from agent_guardrails._sdk import is_gated, without_failure_handling
from agent_guardrails.classify import DEFAULT_CLASSIFIER, ToolClassifier
from agent_guardrails.confirm import ConfirmBudget, current_confirm_budget

logger = logging.getLogger(__name__)

ToolT = TypeVar("ToolT")


@dataclass(frozen=True)
class ToolCap:
    """How many distinct calls a tool (or group) gets, and where to send the
    model instead. The refusal is only useful if ``steer`` names the tool that
    answers the question in one call."""

    limit: int
    steer: str = ""


@dataclass(frozen=True)
class BudgetPolicy:
    """The budget's rules for one run.

    ``caps`` is keyed by tool name or by group name. ``group_of`` maps a tool
    to the budget it shares: group a connector's tools by toolkit so adding a
    verb to the toolkit cannot raise its ceiling. ``default_cap`` applies to
    every read with no entry of its own (``None``: uncapped, memo only).
    """

    caps: Mapping[str, ToolCap] = field(default_factory=dict)
    group_of: Callable[[str], str] | None = None
    default_cap: ToolCap | None = None
    run_record_budget: int = 14
    refusal_ceiling: int = 12
    soft_deadline_fraction: float = 0.7
    classifier: ToolClassifier = DEFAULT_CLASSIFIER

    def group(self, name: str) -> str:
        return (self.group_of(name) if self.group_of is not None else name) or name

    def cap_for(self, name: str) -> ToolCap | None:
        return self.caps.get(name) or self.caps.get(self.group(name)) or self.default_cap

    def is_read(self, name: str) -> bool:
        return self.classifier.is_read(name)


OUT_OF_TIME = (
    "STOP READING: this run is nearly out of time. Do not call another lookup. "
    "The next thing you produce must be the answer, written from what you have "
    "already gathered. A partial answer that names what you could not check is "
    "worth everything; running out of time mid-lookup returns nothing at all."
)

#: The same event, worded for a nested agent. It has no user and writes no
#: final answer, so "your next output must be the answer" is not something it
#: can do. Told that, it either reports the cut-off as a provider failure (and
#: the caller retries) or offers to do the work later, to a program that
#: cannot accept.
OUT_OF_TIME_NESTED = (
    "STOP READING: the run is nearly out of time. Do not call another lookup and "
    "do not propose one. Report now, in plain sentences, what you did retrieve and "
    "what you could not. This cut-off is the caller's own time limit, not a provider "
    "failure, so say the lookup was stopped for time; a retry will be refused the "
    "same way. You are being called as a tool by another agent, not by a person: "
    "nobody can answer a question or approve a next step."
)

LOOPING = (
    "STOP: this lookup has been refused repeatedly and the run is looping. The data "
    "you are fetching one record at a time is already in a bulk result from earlier "
    "in this run. Answer from it now; another call will be refused too."
)

#: Read by whichever model is running, including a nested agent whose first
#: call repeats a lookup its caller made. An unattributed "you already did
#: this, move on" is false from in there, and a careful model treats it as
#: prompt injection and ignores it. So: say who is speaking, say the cache is
#: run-wide, and state a fact rather than an instruction.
REPEAT_NOTE = (
    "\n\n(Run-wide read cache: this exact lookup was already made somewhere in this "
    "run, possibly by the agent that called you rather than by you, so the result "
    "above is the cached answer rather than a fresh call. It is the same data either "
    "way; nothing is withheld and no check is skipped.)"
)


@dataclass
class CallLedger:
    """What one run has already asked for."""

    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    memo: dict[tuple[str, str], str] = field(default_factory=dict)
    distinct: dict[str, set[str]] = field(default_factory=dict)
    capped_groups: set[str] = field(default_factory=set)
    refusals: dict[str, int] = field(default_factory=dict)
    memo_hits: dict[str, int] = field(default_factory=dict)
    #: Reads in flight, by tool and arguments, with the write generation each
    #: started in.
    inflight: dict[tuple[str, str], tuple[asyncio.Future[Any], int]] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    #: The run's deadline in seconds, the same one the watchdog enforces.
    deadline_s: float | None = None
    #: The run's approval budget, whose paused time is not counted as elapsed.
    confirm: ConfirmBudget | None = None
    #: Bumped when a write starts and when it ends. A read caches its answer
    #: only if no write started or ended while it ran.
    write_gen: int = 0

    @property
    def elapsed_s(self) -> float:
        """Active seconds since the run started: time spent waiting on a person
        to answer an approval card is left out, as the watchdog leaves it out."""
        paused = self.confirm.paused_s() if self.confirm is not None else 0.0
        return max(0.0, time.monotonic() - self.started - paused)

    @property
    def out_of_time(self) -> bool:
        if self.deadline_s is None:
            return False
        return self.elapsed_s >= self.deadline_s * self.policy.soft_deadline_fraction

    @property
    def looping(self) -> bool:
        return sum(self.refusals.values()) >= self.policy.refusal_ceiling

    @property
    def record_calls(self) -> int:
        """Distinct calls made across every capped group."""
        return sum(len(slots) for g, slots in self.distinct.items() if g in self.capped_groups)

    def invalidate(self) -> None:
        """Drop every cached read; called after any write."""
        self.memo.clear()

    def summary(self) -> dict[str, Any]:
        """What the budget did this run. Log it once per run: a refusal string
        buried in a nested agent's tool output is invisible otherwise."""
        return {
            "record_calls": self.record_calls,
            "run_record_budget": self.policy.run_record_budget,
            "memo_hits": sum(self.memo_hits.values()),
            "refusals": sum(self.refusals.values()),
            "memo_hits_by_tool": dict(self.memo_hits),
            "refusals_by_tool": dict(self.refusals),
            "hit_refusal_ceiling": self.looping,
            "ran_out_of_time": self.out_of_time,
            # None means the run had no deadline at all, which is a different
            # diagnosis from "had one and passed it". Keep the key present.
            "deadline_s": self.deadline_s,
            "elapsed_s": round(self.elapsed_s, 1),
            "distinct_by_group": {g: len(s) for g, s in self.distinct.items()},
        }


_ledger: ContextVar[CallLedger | None] = ContextVar("agent_guardrails_call_ledger", default=None)
_in_nested: ContextVar[bool] = ContextVar("agent_guardrails_in_nested", default=False)


def bind_call_ledger(
    policy: BudgetPolicy | None = None,
    *,
    deadline_s: float | None = None,
    confirm: ConfirmBudget | None = None,
) -> tuple[CallLedger, Token[CallLedger | None]]:
    """Start a fresh budget for one run. Reset with :func:`reset_call_ledger`.

    Pass the same ``deadline_s`` the watchdog enforces; without it the ledger
    has no clock and only the hard cut-off applies. ``confirm`` is the run's
    approval budget (default: the one bound now), whose waits don't count as
    elapsed time.
    """
    ledger = CallLedger(
        policy=policy or BudgetPolicy(),
        deadline_s=deadline_s,
        confirm=confirm if confirm is not None else current_confirm_budget(),
    )
    return ledger, _ledger.set(ledger)


def reset_call_ledger(token: Token[CallLedger | None]) -> None:
    _ledger.reset(token)


def current_call_ledger() -> CallLedger | None:
    return _ledger.get()


def enter_nested() -> Token[bool]:
    """Mark the current context as inside a nested agent consult."""
    return _in_nested.set(True)


def leave_nested(token: Token[bool]) -> None:
    try:
        _in_nested.reset(token)
    except ValueError:  # pragma: no cover - token from another context
        logger.debug("nested marker token belonged to another context")


def in_nested() -> bool:
    return _in_nested.get()


def _arg_key(args_json: str) -> str:
    """A stable key for one argument set. Unparseable input is used verbatim:
    a memo miss is always safe, a false hit is not."""
    try:
        parsed = json.loads(args_json or "{}")
    except (TypeError, ValueError):
        return args_json or ""
    if not isinstance(parsed, dict):
        return args_json or ""
    return json.dumps(parsed, sort_keys=True, default=str)


def _refusal(name: str, group: str, cap: ToolCap, ledger: CallLedger) -> str:
    seen = len(ledger.distinct.get(group, ()))
    if ledger.record_calls >= ledger.policy.run_record_budget:
        reached = f"This run has already looked up {ledger.record_calls} records one at a time"
    elif group != name:
        reached = f"This run has already searched {group} {seen} different ways"
    else:
        reached = f"This run has already called `{name}` on {seen} different records"
    steer = f" {cap.steer}" if cap.steer else ""
    return (
        f"{reached}, which is the cap. Pulling records one by one is what makes a run "
        f"time out before it answers.{steer} Answer with the data you already have, and "
        "say plainly which records you did not open individually."
    )


def budget_tool(tool: ToolT) -> ToolT:
    """A copy of ``tool`` whose calls go through the run's budget.

    Works on any object with ``name`` and an async ``on_invoke_tool(ctx,
    args_json)``, which is the OpenAI Agents SDK's ``FunctionTool`` shape. A
    copy, so the original stays unwrapped for direct tests. With no ledger
    bound the copy behaves exactly like the original.

    A tool ``gate_tool`` wrapped is always a write here, whatever the
    budget's classifier says.
    """
    name = str(getattr(tool, "name", "") or "")
    inner = getattr(tool, "on_invoke_tool", None)
    if not name or not callable(inner):
        return tool
    gated = is_gated(tool)
    # Reads run with the SDK's failure handling off, so an error is seen as an
    # error (never cached) and then shown the way the tool would show it.
    raw_read, show_error = without_failure_handling(tool)
    # Identity, not name: the same tool wrapped twice (an orchestrator and its
    # specialist both given ``search``) shares one cache; two tools that only
    # share a name don't. ``inner`` is held by this closure, so its id can't
    # be reused by another object while the wrapper exists.
    ident = f"{name}@{id(inner):x}"

    async def read_failed(ctx: Any, exc: Exception) -> Any:
        if show_error is None:
            raise exc
        shown = await show_error(ctx, exc)
        if shown is None:
            raise exc
        return shown

    async def invoke(ctx: Any, args_json: str) -> Any:
        ledger = _ledger.get()
        if ledger is None:
            return await inner(ctx, args_json)
        policy = ledger.policy
        if gated or not policy.is_read(name):
            # A write: never capped, never cached, and it makes cached reads
            # stale, including reads that are still running now.
            ledger.write_gen += 1
            try:
                return await inner(ctx, args_json)
            finally:
                ledger.write_gen += 1
                ledger.invalidate()

        key = _arg_key(args_json)
        cached = ledger.memo.get((ident, key))
        if cached is not None:
            ledger.memo_hits[name] = ledger.memo_hits.get(name, 0) + 1
            return cached + REPEAT_NOTE

        running = ledger.inflight.get((ident, key))
        if running is not None and running[1] == ledger.write_gen:
            ledger.memo_hits[name] = ledger.memo_hits.get(name, 0) + 1
            try:
                shared = await asyncio.shield(running[0])
            except Exception as exc:
                return await read_failed(ctx, exc)
            return shared + REPEAT_NOTE if isinstance(shared, str) else shared

        # Out of time applies to every budgeted read, capped or not.
        if ledger.out_of_time:
            ledger.refusals[name] = ledger.refusals.get(name, 0) + 1
            return OUT_OF_TIME_NESTED if in_nested() else OUT_OF_TIME

        group = policy.group(name)
        cap = policy.cap_for(name)
        seen = ledger.distinct.setdefault(group, set())
        # Namespaced by tool: a group spans several tools, and two of them
        # called with the same arguments are different reads.
        slot = f"{ident}:{key}"
        reserved = False
        if cap is not None and slot not in seen:
            ledger.capped_groups.add(group)
            if len(seen) >= cap.limit or ledger.record_calls >= policy.run_record_budget:
                ledger.refusals[name] = ledger.refusals.get(name, 0) + 1
                logger.info(
                    "budget refused %s (group %s: %d/%d, run %d/%d)",
                    name,
                    group,
                    len(seen),
                    cap.limit,
                    ledger.record_calls,
                    policy.run_record_budget,
                )
                return LOOPING if ledger.looping else _refusal(name, group, cap, ledger)
            # Reserve BEFORE awaiting; there is no await between the check and
            # this write, so within one event loop it is atomic.
            seen.add(slot)
            reserved = True

        started_gen = ledger.write_gen
        task: asyncio.Future[Any] = asyncio.ensure_future(raw_read(ctx, args_json))
        ledger.inflight[(ident, key)] = (task, started_gen)
        try:
            result = await task
        except Exception as exc:
            # A failed read must not use up a slot a working call could spend,
            # and its error is never cached.
            if reserved:
                seen.discard(slot)
            return await read_failed(ctx, exc)
        except BaseException:
            if reserved:
                seen.discard(slot)
            raise
        finally:
            if ledger.inflight.get((ident, key), (None, 0))[0] is task:
                ledger.inflight.pop((ident, key), None)
        seen.add(slot)
        if isinstance(result, str) and ledger.write_gen == started_gen:
            ledger.memo[(ident, key)] = result
        return result

    wrapped = copy.copy(tool)
    wrapped.on_invoke_tool = invoke  # type: ignore[attr-defined]
    return wrapped


def budget_tools(tools: Sequence[ToolT]) -> list[ToolT]:
    """Wrap a whole tool list. Order is preserved."""
    return [budget_tool(t) for t in tools]


__all__ = [
    "LOOPING",
    "OUT_OF_TIME",
    "OUT_OF_TIME_NESTED",
    "REPEAT_NOTE",
    "BudgetPolicy",
    "CallLedger",
    "ToolCap",
    "bind_call_ledger",
    "budget_tool",
    "budget_tools",
    "current_call_ledger",
    "enter_nested",
    "in_nested",
    "leave_nested",
    "reset_call_ledger",
]
