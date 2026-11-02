from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_guardrails.budget import (
    OUT_OF_TIME_NESTED,
    BudgetPolicy,
    bind_call_ledger,
    budget_tool,
    in_nested,
    reset_call_ledger,
)
from agent_guardrails.nested import (
    NO_TIME_FOR_A_CONSULT,
    bind_nested_ledger,
    cap_nested_tool,
    reset_nested_ledger,
)

from .tools import FakeTool, Recorder, invoke, tool


def consult_tool(name: str = "ask_researcher", delay: float = 0.0) -> tuple[FakeTool, list[bool]]:
    seen_nested: list[bool] = []

    async def body(ctx: Any, payload: str) -> str:
        seen_nested.append(in_nested())
        await asyncio.sleep(delay)
        return f"findings for {json.loads(payload).get('q')}"

    return FakeTool(name=name, on_invoke_tool=body), seen_nested


async def test_consults_past_the_cap_are_refused() -> None:
    t, seen = consult_tool()
    wrapped = cap_nested_tool(t)
    ledger, token = bind_nested_ledger(cap=2)
    try:
        replies = [await invoke(wrapped, q=str(i)) for i in range(4)]
    finally:
        reset_nested_ledger(token)
    assert len(seen) == 2
    assert replies[2].startswith("STOP CONSULTING")
    assert ledger is not None
    assert ledger.summary()["refused"] == {"ask_researcher": 2}
    assert ledger.summary()["hit_cap"] is True


async def test_the_cap_reserves_before_awaiting() -> None:
    t, seen = consult_tool(delay=0.02)
    wrapped = cap_nested_tool(t)
    _, token = bind_nested_ledger(cap=2)
    try:
        replies = await asyncio.gather(*[invoke(wrapped, q=str(i)) for i in range(6)])
    finally:
        reset_nested_ledger(token)
    assert len(seen) == 2
    assert sum(r.startswith("STOP CONSULTING") for r in replies) == 4


async def test_no_cap_bound_means_no_cap() -> None:
    t, seen = consult_tool()
    wrapped = cap_nested_tool(t)
    ledger, token = bind_nested_ledger(cap=None)
    try:
        for i in range(5):
            await invoke(wrapped, q=str(i))
    finally:
        reset_nested_ledger(token)
    assert ledger is None
    assert len(seen) == 5


async def test_out_of_time_comes_first_even_without_a_cap() -> None:
    t, seen = consult_tool()
    wrapped = cap_nested_tool(t)
    _, token = bind_call_ledger(deadline_s=0.0)
    try:
        assert await invoke(wrapped, q="x") == NO_TIME_FOR_A_CONSULT
    finally:
        reset_call_ledger(token)
    assert seen == []


async def test_a_consult_runs_marked_as_nested() -> None:
    t, seen = consult_tool()
    await invoke(cap_nested_tool(t), q="x")
    assert seen == [True]
    assert not in_nested()


async def test_reads_inside_a_consult_get_the_nested_wording() -> None:
    read, _ = tool("list_orders")
    budgeted_read = budget_tool(read)
    ledger, token = bind_call_ledger(BudgetPolicy())
    replies: list[str] = []

    async def body(ctx: Any, payload: str) -> str:
        ledger.deadline_s = 0.0  # time runs out while the nested agent is working
        replies.append(await invoke(budgeted_read))
        return "partial"

    try:
        await invoke(cap_nested_tool(FakeTool("ask_researcher", body)), q="x")
    finally:
        reset_call_ledger(token)
    assert replies == [OUT_OF_TIME_NESTED]


async def test_errors_from_the_nested_run_propagate() -> None:
    t, _ = tool("ask_researcher", Recorder(fail=True))
    with pytest.raises(ConnectionError):
        await invoke(cap_nested_tool(t), q="x")
    assert not in_nested()


async def test_a_consult_after_a_long_approval_wait_is_not_refused_for_time() -> None:
    from agent_guardrails.confirm import ConfirmBudget

    approvals = ConfirmBudget()
    ledger, token = bind_call_ledger(BudgetPolicy(), deadline_s=1.0, confirm=approvals)
    try:
        approvals.begin_wait()
        await asyncio.sleep(1.0)  # a person reads a card; the run isn't slow
        approvals.end_wait()
        t, _ = consult_tool()
        assert await invoke(cap_nested_tool(t), q="x") == "findings for x"
    finally:
        reset_call_ledger(token)
    assert not ledger.out_of_time
