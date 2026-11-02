"""The stream watchdog, with fast clocks so the tests take milliseconds."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_guardrails.budget import BudgetPolicy, bind_call_ledger, reset_call_ledger
from agent_guardrails.confirm import ConfirmChannel, bind_confirm_budget, reset_confirm_budget
from agent_guardrails.cost import bind_run_usage, reset_run_usage
from agent_guardrails.watchdog import HardStop, SoftStop, watch_stream
from agent_guardrails.write_marker import bind_write_marker, executing_write, unbind_write_marker

FAST: dict[str, Any] = {"stall_timeout": 0.05, "poll": 0.01}


class Ev:
    """Stands in for the SDK's run-item stream event."""

    def __init__(self, name: str) -> None:
        self.name = name


async def drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in stream]


async def test_events_pass_through() -> None:
    async def gen() -> AsyncIterator[str]:
        for token in "abc":
            yield token

    assert await drain(watch_stream(gen(), **FAST)) == ["a", "b", "c"]


async def test_silence_is_a_stall() -> None:
    async def gen() -> AsyncIterator[str]:
        yield "first"
        await asyncio.sleep(1)
        yield "never"

    with pytest.raises(HardStop) as stop:
        await drain(watch_stream(gen(), **FAST))
    assert stop.value.reason == "stalled"


async def test_a_running_tool_stretches_the_stall_budget() -> None:
    async def gen() -> AsyncIterator[Ev]:
        yield Ev("tool_called")
        await asyncio.sleep(0.08)  # longer than the stall timeout, shorter than 2.5x
        yield Ev("tool_output")

    events = await drain(watch_stream(gen(), **FAST))
    assert [e.name for e in events] == ["tool_called", "tool_output"]


async def test_a_hung_tool_still_dies() -> None:
    async def gen() -> AsyncIterator[Ev]:
        yield Ev("tool_called")
        await asyncio.sleep(1)
        yield Ev("tool_output")

    with pytest.raises(HardStop) as stop:
        await drain(watch_stream(gen(), **FAST))
    assert stop.value.reason == "tool_stalled"


async def test_a_person_reading_a_card_pauses_the_clocks() -> None:
    channel = ConfirmChannel(timeout_s=0.2)
    budget, token = bind_confirm_budget()
    try:

        async def gen() -> AsyncIterator[str]:
            approval = await channel.request(action="send", title="t", details="d")
            yield approval.status

        results = await drain(watch_stream(gen(), max_wall_clock=0.06, **FAST))
    finally:
        reset_confirm_budget(token)
    # Waited 0.2s for an answer with a 0.05s stall timeout and a 0.06s deadline.
    assert results == ["unanswered"]
    assert budget.expired == 1


async def test_errors_are_forwarded_verbatim() -> None:
    class Boom(RuntimeError):
        pass

    async def gen() -> AsyncIterator[str]:
        yield "ok"
        raise Boom("upstream failed")

    with pytest.raises(Boom, match="upstream failed"):
        await drain(watch_stream(gen(), **FAST))


async def test_an_empty_stream_completes() -> None:
    async def gen() -> AsyncIterator[str]:
        return
        yield  # pragma: no cover

    assert await drain(watch_stream(gen(), **FAST)) == []


async def busy(seconds: float) -> AsyncIterator[str]:
    """A run that keeps emitting events, so only a clock can stop it."""
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        yield "token"
        await asyncio.sleep(0.005)


async def test_the_soft_deadline_fires_before_the_hard_one() -> None:
    with pytest.raises(SoftStop) as stop:
        await drain(watch_stream(busy(1), max_wall_clock=0.1, soft_fraction=0.5, **FAST))
    assert stop.value.reason == "soft_deadline"


async def test_the_hard_deadline_without_a_soft_one() -> None:
    with pytest.raises(HardStop) as stop:
        await drain(watch_stream(busy(1), max_wall_clock=0.05, soft_fraction=None, **FAST))
    assert stop.value.reason == "hard_deadline"


async def test_the_cost_ceiling_stops_a_busy_run() -> None:
    usage, token = bind_run_usage(ceiling_usd=0.05)
    try:

        async def spending() -> AsyncIterator[str]:
            for _ in range(100):
                usage.add("m", cost_usd=0.01)
                yield "turn"
                await asyncio.sleep(0.005)

        with pytest.raises(SoftStop) as stop:
            await drain(watch_stream(spending(), **FAST))
    finally:
        reset_run_usage(token)
    assert stop.value.reason == "cost_ceiling"
    assert "$0.05" in str(stop.value)


async def test_the_refusal_ceiling_stops_a_looping_run() -> None:
    ledger, token = bind_call_ledger(BudgetPolicy(refusal_ceiling=3))
    try:

        async def looping() -> AsyncIterator[str]:
            for _ in range(100):
                ledger.refusals["get_order"] = ledger.refusals.get("get_order", 0) + 1
                yield "refused"
                await asyncio.sleep(0.005)

        with pytest.raises(SoftStop) as stop:
            await drain(watch_stream(looping(), **FAST))
    finally:
        reset_call_ledger(token)
    assert stop.value.reason == "refusal_ceiling"


async def test_a_write_in_flight_suppresses_soft_stops_and_stretches_the_hard_one() -> None:
    usage, usage_token = bind_run_usage(ceiling_usd=0.5)
    _, write_token = bind_write_marker()
    state = {"write_finished": False}
    received: list[str] = []
    try:

        async def writing() -> AsyncIterator[str]:
            with executing_write("send_email"):
                usage.add("m", cost_usd=1.0)  # crosses the ceiling mid-write
                # 0.12s of activity: past the 0.1s hard deadline, inside 1.5x.
                async for event in busy(0.12):
                    yield event
            state["write_finished"] = True
            yield "after"

        async def consume() -> None:
            async for event in watch_stream(
                writing(), max_wall_clock=0.1, soft_fraction=0.5, **FAST
            ):
                received.append(event)

        with pytest.raises(SoftStop) as stop:
            await consume()
    finally:
        unbind_write_marker(write_token)
        reset_run_usage(usage_token)
    # Neither the cost ceiling nor either deadline cut the write short; the
    # soft stop came once the write was done.
    assert state["write_finished"]
    assert stop.value.reason == "cost_ceiling"
    assert len(received) > 5


async def test_clocks_are_published() -> None:
    clocks: dict[str, float] = {}

    async def gen() -> AsyncIterator[str]:
        yield "a"

    await drain(watch_stream(gen(), clocks=clocks, **FAST))
    assert "active_s" in clocks
