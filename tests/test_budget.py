from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from agent_guardrails.budget import (
    LOOPING,
    OUT_OF_TIME,
    OUT_OF_TIME_NESTED,
    REPEAT_NOTE,
    BudgetPolicy,
    CallLedger,
    ToolCap,
    bind_call_ledger,
    budget_tool,
    enter_nested,
    leave_nested,
    reset_call_ledger,
)
from agent_guardrails.classify import DEFAULT_CLASSIFIER

from .tools import Recorder, invoke, tool


@pytest.fixture
def ledger() -> Iterator[CallLedger]:
    ledger, token = bind_call_ledger(
        BudgetPolicy(
            caps={
                "get_order": ToolCap(3, "Use `list_orders` for many orders."),
                "CRM": ToolCap(2, "Use the CRM export."),
                "get_invoice": ToolCap(10),
            },
            group_of=lambda name: name.split("_", 1)[0] if name.isupper() else name,
            run_record_budget=6,
            refusal_ceiling=4,
            # CRM_FETCH_CONTACT: a toolkit prefix the classifier must be told about.
            classifier=DEFAULT_CLASSIFIER.extend(namespaces=["CRM"]),
        )
    )
    try:
        yield ledger
    finally:
        reset_call_ledger(token)


async def test_unbound_tools_behave_exactly_like_the_original() -> None:
    t, body = tool("get_order")
    wrapped = budget_tool(t)
    for _ in range(5):
        await invoke(wrapped, id=1)
    assert len(body.calls) == 5


async def test_identical_reads_are_answered_from_the_memo(ledger: CallLedger) -> None:
    t, body = tool("get_order")
    wrapped = budget_tool(t)
    first = await invoke(wrapped, id=1, expand=True)
    second = await invoke(wrapped, expand=True, id=1)  # same call, other key order
    assert len(body.calls) == 1
    assert second == first + REPEAT_NOTE
    assert ledger.memo_hits == {"get_order": 1}


async def test_identical_concurrent_reads_share_one_call(ledger: CallLedger) -> None:
    t, body = tool("search_orders", Recorder(delay=0.05))
    wrapped = budget_tool(t)
    results = await asyncio.gather(*[invoke(wrapped, q="late") for _ in range(4)])
    assert len(body.calls) == 1
    assert sum(r.endswith(REPEAT_NOTE) for r in results) == 3


async def test_a_cap_refuses_and_names_the_bulk_tool(ledger: CallLedger) -> None:
    t, body = tool("get_order")
    wrapped = budget_tool(t)
    for i in range(3):
        await invoke(wrapped, id=i)
    refusal = await invoke(wrapped, id=99)
    assert len(body.calls) == 3
    assert "called `get_order` on 3 different records" in refusal
    assert "Use `list_orders` for many orders." in refusal
    # A repeat of an allowed call is still served from the memo, past the cap.
    assert (await invoke(wrapped, id=0)).endswith(REPEAT_NOTE)


async def test_a_cap_reserves_before_awaiting(ledger: CallLedger) -> None:
    # Ten concurrent distinct calls against a cap of three: check-then-act
    # around the await would let all ten through.
    t, body = tool("get_order", Recorder(delay=0.02))
    wrapped = budget_tool(t)
    results = await asyncio.gather(*[invoke(wrapped, id=i) for i in range(10)])
    assert len(body.calls) == 3
    refused = [r for r in results if not r.startswith("result for")]
    assert len(refused) == 7
    # Past the refusal ceiling (4 here) the refusal turns into the loop warning.
    assert all("which is the cap" in r or r == LOOPING for r in refused)
    assert sum(ledger.refusals.values()) == 7


async def test_a_failed_read_gives_its_slot_back(ledger: CallLedger) -> None:
    body = Recorder(fail=True)
    t, _ = tool("get_order", body)
    wrapped = budget_tool(t)
    for i in range(3):
        with pytest.raises(ConnectionError):
            await invoke(wrapped, id=i)
    body.fail = False
    assert "result for" in await invoke(wrapped, id=7)


async def test_a_group_shares_one_budget(ledger: CallLedger) -> None:
    fetch, _ = tool("CRM_FETCH_CONTACT")
    search, _ = tool("CRM_SEARCH_DEALS")
    await invoke(budget_tool(fetch), id=1)
    await invoke(budget_tool(search), q="x")
    refusal = await invoke(budget_tool(fetch), id=2)
    assert "searched CRM 2 different ways" in refusal
    assert "Use the CRM export." in refusal


async def test_the_run_wide_budget_spans_every_capped_group(ledger: CallLedger) -> None:
    order, _ = tool("get_order")
    crm, _ = tool("CRM_GET_CONTACT")
    for i in range(3):
        await invoke(budget_tool(order), id=i)
    for i in range(2):
        await invoke(budget_tool(crm), id=i)
    invoice, body = tool("get_invoice")
    await invoke(budget_tool(invoice), id=1)  # sixth distinct capped call: allowed
    refusal = await invoke(budget_tool(invoice), id=2)
    assert len(body.calls) == 1
    assert "looked up 6 records one at a time" in refusal


async def test_writes_are_never_capped_or_cached_and_clear_the_memo(ledger: CallLedger) -> None:
    read, read_body = tool("get_order")
    write, write_body = tool("update_order")
    await invoke(budget_tool(read), id=1)
    for _ in range(5):
        await invoke(budget_tool(write), id=1, status="shipped")
    assert len(write_body.calls) == 5
    assert ledger.refusals == {}
    # The read-back after a write sees fresh data, not the cached answer.
    fresh = await invoke(budget_tool(read), id=1)
    assert not fresh.endswith(REPEAT_NOTE)
    assert len(read_body.calls) == 2


async def test_unknown_tools_count_as_writes(ledger: CallLedger) -> None:
    t, body = tool("frobnicate_widget")
    wrapped = budget_tool(t)
    for _ in range(3):
        await invoke(wrapped, id=1)
    assert len(body.calls) == 3  # never memoized: it might write


async def test_past_the_refusal_ceiling_the_run_is_looping(ledger: CallLedger) -> None:
    t, _ = tool("get_order")
    wrapped = budget_tool(t)
    for i in range(3):
        await invoke(wrapped, id=i)
    replies = [await invoke(wrapped, id=100 + i) for i in range(5)]
    assert ledger.looping
    assert replies[-1] == LOOPING
    assert ledger.summary()["hit_refusal_ceiling"] is True


async def test_out_of_time_refuses_every_read(ledger: CallLedger) -> None:
    ledger.deadline_s = 0.0
    t, body = tool("list_orders")
    assert await invoke(budget_tool(t)) == OUT_OF_TIME
    token = enter_nested()
    try:
        assert await invoke(budget_tool(t)) == OUT_OF_TIME_NESTED
    finally:
        leave_nested(token)
    assert body.calls == []
    assert ledger.summary()["ran_out_of_time"] is True


async def test_uncapped_reads_are_memoized_but_never_refused(ledger: CallLedger) -> None:
    t, body = tool("search_orders")
    wrapped = budget_tool(t)
    for i in range(20):
        await invoke(wrapped, q=str(i))
    assert len(body.calls) == 20
    assert ledger.record_calls == 0


def test_summary_keeps_the_deadline_key_when_there_is_none() -> None:
    ledger, token = bind_call_ledger()
    try:
        summary = ledger.summary()
    finally:
        reset_call_ledger(token)
    assert "deadline_s" in summary
    assert summary["deadline_s"] is None


def test_tools_without_an_invoke_seam_pass_through() -> None:
    sentinel = object()
    assert budget_tool(sentinel) is sentinel


class Ledger:
    """A toy record: a read reports its status, a write changes it."""

    def __init__(self) -> None:
        self.status = "unpaid"
        self.reads = 0

    async def get(self, ctx: object, args_json: str) -> str:
        self.reads += 1
        seen = self.status
        await asyncio.sleep(0.05)  # slow enough for a write to finish meanwhile
        return f'{{"status": "{seen}"}}'

    async def pay(self, ctx: object, args_json: str) -> str:
        self.status = "paid"
        return "updated"


async def test_a_read_in_flight_when_a_write_lands_is_not_cached(ledger: CallLedger) -> None:
    from .tools import FakeTool

    record = Ledger()
    get = budget_tool(FakeTool(name="get_invoice", on_invoke_tool=record.get))
    pay = budget_tool(FakeTool(name="pay_invoice", on_invoke_tool=record.pay))
    # One turn: the read starts, the write lands while it runs.
    reading = asyncio.create_task(invoke(get, id=7))
    await asyncio.sleep(0.01)
    await invoke(pay, id=7)
    before = await reading
    assert before == '{"status": "unpaid"}'
    # The read-back in the next turn must see the write, not the stale answer.
    after = await invoke(get, id=7)
    assert after == '{"status": "paid"}'
    assert record.reads == 2


async def test_a_read_that_started_before_a_write_is_not_shared_after_it(
    ledger: CallLedger,
) -> None:
    from .tools import FakeTool

    record = Ledger()
    get = budget_tool(FakeTool(name="get_invoice", on_invoke_tool=record.get))
    pay = budget_tool(FakeTool(name="pay_invoice", on_invoke_tool=record.pay))
    early = asyncio.create_task(invoke(get, id=7))
    await asyncio.sleep(0.01)
    await invoke(pay, id=7)
    late = await invoke(get, id=7)  # must not ride on the early, stale call
    assert late == '{"status": "paid"}'
    assert await early == '{"status": "unpaid"}'


async def test_time_spent_on_an_approval_card_is_not_spent_reading() -> None:
    from agent_guardrails.confirm import ConfirmBudget

    approvals = ConfirmBudget()
    ledger, token = bind_call_ledger(BudgetPolicy(), deadline_s=1.0, confirm=approvals)
    try:
        approvals.begin_wait()
        await asyncio.sleep(1.0)  # a person reads a card for the whole deadline
        approvals.end_wait()
        assert not ledger.out_of_time
        t, body = tool("get_order")
        assert "result for" in await invoke(budget_tool(t), id=1)
        assert len(body.calls) == 1
        await asyncio.sleep(0.8)  # active time now passes the soft deadline (0.7 s)
        assert ledger.out_of_time
    finally:
        reset_call_ledger(token)


async def test_two_tools_that_share_a_name_never_answer_for_each_other(
    ledger: CallLedger,
) -> None:
    from .tools import FakeTool

    async def orchestrator_search(ctx: object, args_json: str) -> str:
        return "orchestrator's index"

    async def specialist_search(ctx: object, args_json: str) -> str:
        return "specialist's index"

    a = budget_tool(FakeTool(name="search", on_invoke_tool=orchestrator_search))
    b = budget_tool(FakeTool(name="search", on_invoke_tool=specialist_search))
    assert await invoke(a, q="x") == "orchestrator's index"
    assert await invoke(b, q="x") == "specialist's index"


async def test_the_same_tool_wrapped_twice_shares_one_cache(ledger: CallLedger) -> None:
    t, body = tool("search_orders")
    first, second = budget_tool(t), budget_tool(t)
    await invoke(first, q="x")
    assert (await invoke(second, q="x")).endswith(REPEAT_NOTE)
    assert len(body.calls) == 1
