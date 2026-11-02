from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from agent_guardrails.confirm import (
    MAX_EDIT_VALUE_LEN,
    Approval,
    ConfirmBudget,
    ConfirmCard,
    ConfirmChannel,
    Emitter,
    bind_confirm_budget,
    confirm_waiting,
    render_details,
    reset_confirm_budget,
)


def answer_with(
    channel: ConfirmChannel, approved: bool, edits: dict[str, str] | None = None
) -> Emitter:
    async def emit(card: ConfirmCard) -> None:
        channel.resolve(card.confirm_id, approved, edits=edits)

    return emit


async def ask(channel: ConfirmChannel, **kwargs: Any) -> Approval:
    params: dict[str, Any] = {"action": "send_email", "title": "Send?", "details": "to x"}
    params.update(kwargs)
    return await channel.request(**params)


async def test_explicit_yes_approves() -> None:
    channel = ConfirmChannel()
    with channel.emitting_to(answer_with(channel, True)):
        approval = await ask(channel)
    assert approval.status == "approved"
    assert approval.approved


async def test_no_is_declined() -> None:
    channel = ConfirmChannel()
    with channel.emitting_to(answer_with(channel, False)):
        approval = await ask(channel)
    assert approval.status == "declined"
    assert not approval.approved


async def test_timeout_is_unanswered_not_declined() -> None:
    channel = ConfirmChannel(timeout_s=0.05)
    approval = await ask(channel)
    assert approval.status == "unanswered"
    assert channel.pending_ids() == []


async def test_confirm_or_abort_is_true_only_on_yes() -> None:
    channel = ConfirmChannel(timeout_s=0.05)
    with channel.emitting_to(answer_with(channel, True)):
        assert await channel.confirm_or_abort(action="a", title="t", details="d")
    with channel.emitting_to(answer_with(channel, False)):
        assert not await channel.confirm_or_abort(action="a", title="t", details="d")
    assert not await channel.confirm_or_abort(action="a", title="t", details="d")


async def test_without_an_emitter_cards_queue_for_drain() -> None:
    channel = ConfirmChannel(timeout_s=1)
    task = asyncio.create_task(ask(channel))
    for _ in range(50):
        await asyncio.sleep(0)
        cards = channel.drain()
        if cards:
            break
    assert len(cards) == 1
    assert cards[0].action == "send_email"
    assert channel.resolve(cards[0].confirm_id, True) == "resolved"
    assert (await task).approved


async def test_a_full_queue_fails_closed() -> None:
    channel = ConfirmChannel(timeout_s=1, max_queued=1)
    first = asyncio.create_task(ask(channel))
    await asyncio.sleep(0.01)
    approval = await ask(channel)
    assert approval.status == "not_surfaced"
    first.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first


async def test_an_emitter_failure_falls_back_to_the_queue() -> None:
    channel = ConfirmChannel(timeout_s=1)

    async def broken(card: ConfirmCard) -> None:
        raise RuntimeError("bus down")

    with channel.emitting_to(broken):
        task = asyncio.create_task(ask(channel))
        await asyncio.sleep(0.01)
        cards = channel.drain()
        assert len(cards) == 1
        channel.resolve(cards[0].confirm_id, False)
        assert (await task).status == "declined"


async def test_resolving_an_unknown_card_is_reported() -> None:
    assert ConfirmChannel().resolve("nope", True) == "unknown"


async def test_edits_only_for_declared_fields_and_capped() -> None:
    channel = ConfirmChannel()
    edits = {"body": "x" * (MAX_EDIT_VALUE_LEN + 50), "to": "attacker@example.com", "subject": " "}
    with channel.emitting_to(answer_with(channel, True, edits)):
        approval = await ask(channel, editable_fields=["body", "subject"])
    assert approval.edits is not None
    assert set(approval.edits) == {"body"}  # "to" was never editable; blank subject dropped
    assert len(approval.edits["body"]) == MAX_EDIT_VALUE_LEN


async def test_a_decline_carries_no_edits() -> None:
    channel = ConfirmChannel()
    with channel.emitting_to(answer_with(channel, False, {"body": "new"})):
        approval = await ask(channel, editable_fields=["body"])
    assert approval.edits is None


async def test_gates_with_no_editable_fields_drop_edits() -> None:
    channel = ConfirmChannel()
    with channel.emitting_to(answer_with(channel, True, {"body": "new"})):
        approval = await ask(channel)
    assert approval.approved
    assert approval.edits is None


async def test_one_unanswered_gate_stops_the_run_waiting_again() -> None:
    channel = ConfirmChannel(timeout_s=0.05)
    budget, token = bind_confirm_budget()
    try:
        first = await ask(channel)
        started = asyncio.get_running_loop().time()
        second = await ask(channel)
        elapsed = asyncio.get_running_loop().time() - started
    finally:
        reset_confirm_budget(token)
    assert first.status == "unanswered"
    assert second.status == "unanswered"
    assert elapsed < 0.04  # did not wait another timeout
    assert budget.summary() == {
        "raised": 2,
        "answered": 0,
        "expired": 1,
        "skipped": 1,
        "wasted_s": round(budget.wasted_s, 1),
    }


async def test_concurrent_gates_do_not_cancel_each_other() -> None:
    # Gates in flight prove nothing: three raised together all wait.
    channel = ConfirmChannel(timeout_s=1)
    budget, token = bind_confirm_budget()
    try:
        tasks = [asyncio.create_task(ask(channel, action=f"a{i}")) for i in range(3)]
        await asyncio.sleep(0.01)
        assert budget.waiting == 3
        assert confirm_waiting()
        for card in channel.drain():
            channel.resolve(card.confirm_id, True)
        results = await asyncio.gather(*tasks)
    finally:
        reset_confirm_budget(token)
    assert [r.status for r in results] == ["approved"] * 3
    assert budget.skipped == 0
    assert budget.waiting == 0


async def test_unbound_runs_wait_the_full_timeout_every_time() -> None:
    channel = ConfirmChannel(timeout_s=0.03)
    assert (await ask(channel)).status == "unanswered"
    started = asyncio.get_running_loop().time()
    assert (await ask(channel)).status == "unanswered"
    assert asyncio.get_running_loop().time() - started >= 0.025


def test_budget_ceiling_is_configurable() -> None:
    budget = ConfirmBudget(expired_ceiling=2)
    budget.expired = 1
    assert not budget.approval_is_unreachable()
    budget.expired = 2
    assert budget.approval_is_unreachable()


def test_card_as_dict_round_trips_the_payload() -> None:
    card = ConfirmCard(
        confirm_id="c1",
        action="send_email",
        title="Send?",
        details="to x",
        items=("x",),
        request_body={"to": "x"},
        editable_fields=("body",),
    )
    assert card.as_dict() == {
        "confirm_id": "c1",
        "action": "send_email",
        "title": "Send?",
        "details": "to x",
        "items": ["x"],
        "request_body": {"to": "x"},
        "editable_fields": ["body"],
    }


def test_render_shows_the_body_and_sizes_tables() -> None:
    text = render_details(
        "update_sheet",
        {
            "sheet": "Q3",
            "values": [[1, 2, 3], [4, 5, 6]],
            "blob": "a" * 500,
            "body": "Hello there",
        },
    )
    assert "values: 2 row(s) x 3 column(s)" in text
    assert "1, 2, 3" not in text
    assert "(+380 chars)" in text
    assert text.endswith("body:\nHello there")


def test_render_with_no_arguments() -> None:
    assert render_details("ping", {}) == "ping (no arguments)"


@pytest.mark.parametrize("body", ["", "   "])
def test_render_skips_blank_bodies(body: str) -> None:
    assert "body:\n" not in render_details("send", {"to": "x", "body": body})


async def test_a_gate_cancelled_while_its_card_is_being_shown_leaves_nothing_pending() -> None:
    channel = ConfirmChannel(timeout_s=5)
    shown: list[ConfirmCard] = []
    release = asyncio.Event()

    async def slow_emit(card: ConfirmCard) -> None:
        shown.append(card)
        await release.wait()  # a websocket that takes its time

    with channel.emitting_to(slow_emit):
        task = asyncio.create_task(ask(channel))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert channel.pending_ids() == []
    # A click that arrives for the dead card is not "resolved".
    assert channel.resolve(shown[0].confirm_id, True) == "unknown"


@pytest.mark.parametrize("answer", ["false", "yes", 1, None])
async def test_only_a_real_bool_is_an_answer(answer: Any) -> None:
    channel = ConfirmChannel(timeout_s=0.05)
    task = asyncio.create_task(ask(channel))
    await asyncio.sleep(0.01)
    card = channel.drain()[0]
    with pytest.raises(TypeError, match="True or False"):
        channel.resolve(card.confirm_id, answer)
    # The card is still waiting, so the gate ends unanswered, never approved.
    assert (await task).status == "unanswered"


async def test_paused_time_counts_only_while_a_card_waits() -> None:
    budget = ConfirmBudget()
    assert budget.paused_s() == 0.0
    budget.begin_wait()
    budget.begin_wait()  # two cards at once are one pause, not two
    await asyncio.sleep(0.05)
    budget.end_wait()
    budget.end_wait()
    paused = budget.paused_s()
    assert paused >= 0.04
    await asyncio.sleep(0.05)
    assert budget.paused_s() == paused
