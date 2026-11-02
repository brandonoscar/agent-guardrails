"""End to end against the real OpenAI Agents SDK runner, with a scripted model."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agents import Agent, FunctionTool, Runner, Tool, function_tool
from agents.items import ToolCallOutputItem
from agents.result import RunResult, RunResultStreaming

from agent_guardrails import (
    BudgetPolicy,
    ConfirmCard,
    ConfirmChannel,
    InMemoryIdempotencyStore,
    ModelPrice,
    PriceTable,
    RunGuard,
    SoftStop,
    ToolCap,
    budget_tools,
    cap_nested_tool,
)
from agent_guardrails.budget import REPEAT_NOTE
from agent_guardrails.idempotency import derive_key
from agent_guardrails.openai_agents import (
    UsageHooks,
    gate_tool,
    gate_tools,
    require_approval_for_writes,
    watch_run,
)
from agent_guardrails.stores import STATUS_ABANDONED_UNKNOWN
from agent_guardrails.testing import ScriptedModel, Turn, call, say

EMAIL = {"to": "pat@example.com", "subject": "Invoice", "body": "Attached."}


class Mailbox:
    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []
        self.lookups = 0

        @function_tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email."""
            self.sent.append({"to": to, "subject": subject, "body": body})
            return f"sent #{len(self.sent)} to {to}"

        @function_tool
        def search_contacts(name: str) -> str:
            """Find a contact's email address."""
            self.lookups += 1
            return json.dumps({"name": name, "email": "pat@example.com"})

        self.send_email = send_email
        self.search_contacts = search_contacts


def outputs(result: RunResult | RunResultStreaming) -> list[str]:
    return [str(i.output) for i in result.new_items if isinstance(i, ToolCallOutputItem)]


def answering(channel: ConfirmChannel, approved: bool, edits: dict[str, str] | None = None) -> Any:
    cards: list[ConfirmCard] = []

    async def emit(card: ConfirmCard) -> None:
        cards.append(card)
        channel.resolve(card.confirm_id, approved, edits=edits)

    emit.cards = cards  # type: ignore[attr-defined]
    return emit


async def run_mail(
    script: list[Any], *, approved: bool | None, edits: dict[str, str] | None = None, **gate: Any
) -> tuple[Mailbox, RunResult, list[ConfirmCard]]:
    mailbox = Mailbox()
    channel = ConfirmChannel(timeout_s=0.1)
    tools: list[Tool] = [
        gate_tool(
            t, channel=channel, store=InMemoryIdempotencyStore(), system="the mailbox", **gate
        )
        for t in (mailbox.send_email, mailbox.search_contacts)
    ]
    agent = Agent(name="assistant", model=ScriptedModel(script), tools=tools)
    emitter = answering(channel, approved, edits) if approved is not None else None
    with RunGuard(scope="account-1", channel=channel, emitter=emitter):
        result = await Runner.run(agent, "email Pat the invoice")
    return mailbox, result, (emitter.cards if emitter else [])


async def test_an_approved_write_runs_once_and_a_duplicate_replays() -> None:
    mailbox, result, cards = await run_mail(
        [
            [call("search_contacts", name="Pat")],
            [call("send_email", **EMAIL)],
            [call("send_email", **EMAIL)],  # the model retries the same send
            [say("Sent.")],
        ],
        approved=True,
    )
    assert len(mailbox.sent) == 1
    assert len(cards) == 1
    assert "body:\nAttached." in cards[0].details
    sends = [o for o in outputs(result) if o.startswith("sent")]
    assert sends == ["sent #1 to pat@example.com"] * 2
    assert result.final_output == "Sent."


async def test_a_declined_write_never_runs() -> None:
    mailbox, result, _ = await run_mail(
        [[call("send_email", **EMAIL)], [say("ok")]], approved=False
    )
    assert mailbox.sent == []
    assert "was declined" in outputs(result)[0]
    assert "the mailbox" in outputs(result)[0]


async def test_an_unanswered_write_says_nobody_answered() -> None:
    mailbox, result, _ = await run_mail([[call("send_email", **EMAIL)], [say("ok")]], approved=None)
    assert mailbox.sent == []
    assert "was not answered" in outputs(result)[0]
    assert "not a refusal" in outputs(result)[0]


async def test_the_approver_can_edit_content_but_not_recipients() -> None:
    mailbox, _, cards = await run_mail(
        [[call("send_email", **EMAIL)], [say("ok")]],
        approved=True,
        edits={"body": "Attached, with thanks.", "to": "attacker@example.com"},
        editable=["body", "subject"],
    )
    assert cards[0].editable_fields == ("body", "subject")
    assert mailbox.sent == [{**EMAIL, "body": "Attached, with thanks."}]


async def test_no_scope_fails_closed() -> None:
    mailbox = Mailbox()
    channel = ConfirmChannel(timeout_s=0.1)
    send = gate_tool(mailbox.send_email, channel=channel, store=InMemoryIdempotencyStore())
    agent = Agent(name="a", model=ScriptedModel([[call("send_email", **EMAIL)]]), tools=[send])
    result = await Runner.run(agent, "go")  # no RunGuard: no scope
    assert mailbox.sent == []
    assert "prevents duplicate changes" in outputs(result)[0]


def test_reads_pass_through_and_writes_are_copies_that_keep_their_definition() -> None:
    mailbox = Mailbox()
    channel = ConfirmChannel()
    send, search = gate_tools(
        [mailbox.send_email, mailbox.search_contacts], channel=channel, store=None
    )
    assert search is mailbox.search_contacts
    assert isinstance(send, FunctionTool)
    assert send is not mailbox.send_email
    for attr in ("name", "description", "params_json_schema", "strict_json_schema"):
        assert getattr(send, attr) == getattr(mailbox.send_email, attr)
    assert send.on_invoke_tool is not mailbox.send_email.on_invoke_tool


def test_require_approval_for_writes_uses_the_sdk_flag() -> None:
    mailbox = Mailbox()
    send, search = require_approval_for_writes([mailbox.send_email, mailbox.search_contacts])
    assert isinstance(send, FunctionTool)
    assert send.needs_approval is True
    assert mailbox.send_email.needs_approval is False
    assert search is mailbox.search_contacts


@function_tool
def get_order(order_id: int) -> str:
    """Look up one order."""
    return f"order {order_id}: shipped"


def researcher_tool(hooks: UsageHooks | None = None) -> FunctionTool:
    researcher = Agent(
        name="researcher",
        model=ScriptedModel(
            [[say("finding")] for _ in range(10)],
            name="nested",
            input_tokens=1000,
            output_tokens=100,
        ),
    )
    return researcher.as_tool(
        tool_name="ask_researcher", tool_description="Ask the researcher.", hooks=hooks
    )


async def test_budgets_and_nested_caps_hold_in_a_real_run() -> None:
    tools: list[Tool] = [
        *budget_tools([get_order]),
        cap_nested_tool(researcher_tool()),
    ]
    script: list[Turn] = [
        [call("get_order", order_id=7), call("get_order", order_id=7)],
        [call("ask_researcher", input=f"question {i}") for i in range(4)],
        [say("done")],
    ]
    agent = Agent(name="orchestrator", model=ScriptedModel(script), tools=tools)
    with RunGuard(
        scope="t", nested_cap=2, budget=BudgetPolicy(caps={"get_order": ToolCap(5)})
    ) as run:
        result = await Runner.run(agent, "go")
    got = outputs(result)
    assert got[0] == "order 7: shipped"
    assert got[1] == "order 7: shipped" + REPEAT_NOTE
    assert sum(o.startswith("STOP CONSULTING") for o in got) == 2
    summary = run.summary()
    assert summary["nested"]["consults"] == 2
    assert summary["calls"]["memo_hits"] == 1


async def test_the_cost_ceiling_stops_a_streamed_run_and_counts_nested_spend() -> None:
    prices = PriceTable(
        {
            # Illustrative rates for the scripted models; real rates are yours to supply.
            "orchestrator": ModelPrice(input=10.0, output=40.0),
            "nested": ModelPrice(input=10.0, output=40.0),
        }
    )
    hooks = UsageHooks(prices)
    endless = [[call("ask_researcher", input=f"q{i}")] for i in range(50)]
    agent = Agent(
        name="orchestrator",
        model=ScriptedModel(endless, name="orchestrator", input_tokens=1000, output_tokens=100),
        tools=[cap_nested_tool(researcher_tool(hooks))],
    )
    with RunGuard(scope="t", cost_ceiling_usd=0.05) as run:
        result = Runner.run_streamed(agent, "go", hooks=hooks, max_turns=60)
        with pytest.raises(SoftStop) as stop:
            async for _ in watch_run(result, run, poll=0.01):
                pass
    assert stop.value.reason == "cost_ceiling"
    assert run.usage is not None
    assert run.usage.cost_usd >= 0.05
    assert run.usage.by_model.get("nested", 0) >= 1  # the nested agent's calls counted
    assert run.usage.unpriced_calls == 0
    for _ in range(100):  # cancelled, so the run winds down
        if result.is_complete:
            break
        await asyncio.sleep(0.01)
    assert result.is_complete


async def test_usage_without_prices_is_counted_but_unpriced() -> None:
    agent = Agent(
        name="a",
        model=ScriptedModel([[say("hi")]], input_tokens=10, output_tokens=5),
    )
    with RunGuard(scope="t") as run:
        await Runner.run(agent, "go", hooks=UsageHooks())
    assert run.usage is not None
    assert run.usage.summary()["prompt_tokens"] == 10
    assert run.usage.summary()["cost_is_floor"] is True


# ── tool failures, timeouts and identity (the SDK's own handling is off) ────


class FlakyMailbox:
    """send_email fails the way it's told, then works."""

    def __init__(self, error: BaseException | None, *, timeout: float | None = None) -> None:
        self.sent: list[str] = []
        self.attempts = 0
        self.error = error

        @function_tool(timeout=timeout)
        async def send_email(to: str, body: str) -> str:
            """Send an email."""
            self.attempts += 1
            if self.error is not None:
                error, self.error = self.error, None
                raise error
            self.sent.append(to)
            return f"sent to {to}"

        self.send_email = send_email


async def run_flaky(
    mailbox: FlakyMailbox, script: list[Any], **gate: Any
) -> tuple[RunResult, InMemoryIdempotencyStore]:
    channel = ConfirmChannel(timeout_s=1.0)
    store = InMemoryIdempotencyStore()
    send = gate_tool(mailbox.send_email, channel=channel, store=store, system="the mailbox", **gate)
    agent = Agent(name="a", model=ScriptedModel(script), tools=[send])
    with RunGuard(scope="acct-1", channel=channel, emitter=answering(channel, True)):
        result = await Runner.run(agent, "go")
    return result, store


MAIL = {"to": "pat@example.com", "body": "hi"}


async def test_a_write_that_fails_before_sending_is_retryable_not_a_success() -> None:
    from agent_guardrails import PreSideEffectError

    mailbox = FlakyMailbox(PreSideEffectError("the address is malformed"))
    result, _ = await run_flaky(
        mailbox, [[call("send_email", **MAIL)], [call("send_email", **MAIL)], [say("ok")]]
    )
    first, second = outputs(result)[:2]
    assert "failed before anything was changed in the mailbox" in first
    assert "the address is malformed" in first
    assert "safe to try again" in first
    assert second == "sent to pat@example.com"  # the retry asked again and ran
    assert mailbox.sent == ["pat@example.com"]


async def test_a_write_that_fails_partway_is_never_retried_or_replayed_as_sent() -> None:
    mailbox = FlakyMailbox(ConnectionError("connection reset after the request went out"))
    result, store = await run_flaky(
        mailbox, [[call("send_email", **MAIL)], [call("send_email", **MAIL)], [say("ok")]]
    )
    first, second = outputs(result)[:2]
    row = await store.get(scope="acct-1", key=derive_key("acct-1", "send_email", MAIL))
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN  # not "succeeded" with an error text
    assert "can't confirm whether send_email went through" in first
    assert "won't retry it" in first
    assert "An error occurred" not in first  # not the SDK's "please try again"
    assert "I can't confirm whether I managed to" in second  # needs review: not re-run
    assert mailbox.attempts == 1
    assert mailbox.sent == []


async def test_a_read_that_failed_is_not_cached() -> None:
    attempts = {"n": 0}

    @function_tool
    async def get_invoice(invoice_id: int) -> str:
        """Read one invoice."""
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("upstream timeout")
        return json.dumps({"id": invoice_id, "status": "paid"})

    agent = Agent(
        name="a",
        model=ScriptedModel(
            [
                [call("get_invoice", invoice_id=7)],
                [call("get_invoice", invoice_id=7)],
                [call("get_invoice", invoice_id=7)],
                [say("done")],
            ]
        ),
        tools=budget_tools([get_invoice]),
    )
    with RunGuard(scope="t") as run:
        result = await Runner.run(agent, "go")
    got = outputs(result)
    assert got[0] == "An error occurred while running the tool. Please try again."
    assert got[1] == '{"id": 7, "status": "paid"}'  # ran again, not the cached error
    assert got[2].endswith(REPEAT_NOTE)  # the good answer is the one cached
    assert attempts["n"] == 2
    assert run.calls is not None
    assert run.calls.memo_hits == {"get_invoice": 1}


async def test_a_tool_built_to_raise_still_raises_after_the_record_is_written() -> None:
    @function_tool(failure_error_function=None)
    async def send_email(to: str, body: str) -> str:
        """Send an email."""
        raise ConnectionError("reset after send")

    channel = ConfirmChannel(timeout_s=1.0)
    store = InMemoryIdempotencyStore()
    send = gate_tool(send_email, channel=channel, store=store)
    agent = Agent(name="a", model=ScriptedModel([[call("send_email", **MAIL)]]), tools=[send])
    with (
        RunGuard(scope="acct-1", channel=channel, emitter=answering(channel, True)),
        pytest.raises(Exception, match="reset after send"),
    ):
        await Runner.run(agent, "go")
    row = await store.get(scope="acct-1", key=derive_key("acct-1", "send_email", MAIL))
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN


async def test_the_tool_timeout_covers_the_write_not_the_approval_wait() -> None:
    mailbox = FlakyMailbox(None, timeout=0.2)
    channel = ConfirmChannel(timeout_s=2.0)

    async def slow_person(card: ConfirmCard) -> None:
        async def click() -> None:
            await asyncio.sleep(0.5)  # longer than the tool's own timeout
            channel.resolve(card.confirm_id, True)

        asyncio.get_running_loop().create_task(click())

    send = gate_tool(mailbox.send_email, channel=channel, store=InMemoryIdempotencyStore())
    agent = Agent(
        name="a", model=ScriptedModel([[call("send_email", **MAIL)], [say("ok")]]), tools=[send]
    )
    with RunGuard(scope="acct-1", channel=channel, emitter=slow_person):
        result = await Runner.run(agent, "go")
    assert outputs(result)[0] == "sent to pat@example.com"
    assert mailbox.sent == ["pat@example.com"]


async def test_a_write_that_outlives_its_timeout_is_needs_review() -> None:
    @function_tool(timeout=0.1)
    async def send_email(to: str, body: str) -> str:
        """Send an email."""
        await asyncio.sleep(1)
        return "never"

    channel = ConfirmChannel(timeout_s=1.0)
    store = InMemoryIdempotencyStore()
    send = gate_tool(send_email, channel=channel, store=store, system="the mailbox")
    agent = Agent(
        name="a", model=ScriptedModel([[call("send_email", **MAIL)], [say("ok")]]), tools=[send]
    )
    with RunGuard(scope="acct-1", channel=channel, emitter=answering(channel, True)):
        result = await Runner.run(agent, "go")
    assert "didn't finish within 0.1 seconds" in outputs(result)[0]
    assert "Check the mailbox" in outputs(result)[0]
    row = await store.get(scope="acct-1", key=derive_key("acct-1", "send_email", MAIL))
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN


async def test_two_tools_with_one_name_are_separate_actions_given_a_namespace() -> None:
    sent: list[str] = []

    def make(where: str) -> FunctionTool:
        @function_tool(name_override="send_message")
        def send_message(text: str) -> str:
            """Send a message."""
            sent.append(where)
            return f"posted to {where}"

        return send_message

    channel = ConfirmChannel(timeout_s=1.0)
    store = InMemoryIdempotencyStore()
    chat = gate_tool(make("chat"), channel=channel, store=store, namespace="chat")
    mail = gate_tool(make("mail"), channel=channel, store=store, namespace="mail")
    for tool_ in (chat, mail):
        agent = Agent(
            name="a",
            model=ScriptedModel([[call("send_message", text="hi")], [say("ok")]]),
            tools=[tool_],
        )
        with RunGuard(scope="acct-1", channel=channel, emitter=answering(channel, True)):
            await Runner.run(agent, "go")
    assert sent == ["chat", "mail"]


async def test_a_gated_write_is_never_cached_whatever_the_budget_thinks() -> None:
    mailbox = Mailbox()
    channel = ConfirmChannel(timeout_s=1.0)
    send = gate_tool(mailbox.send_email, channel=channel, store=InMemoryIdempotencyStore())
    # A budget whose classifier wrongly calls send_email a read.
    from agent_guardrails.classify import ToolClassifier

    lenient = BudgetPolicy(
        classifier=ToolClassifier(read_verbs=frozenset({"send"}), write_verbs=frozenset())
    )
    agent = Agent(
        name="a",
        model=ScriptedModel(
            [[call("send_email", **EMAIL)], [call("send_email", **EMAIL)], [say("ok")]]
        ),
        tools=budget_tools([send]),
    )
    with RunGuard(
        scope="acct-1", channel=channel, emitter=answering(channel, True), budget=lenient
    ) as run:
        await Runner.run(agent, "go")
    assert run.calls is not None
    assert run.calls.memo_hits == {}  # the duplicate went to the guard, not a cache
    assert len(mailbox.sent) == 1


def test_the_model_is_resolved_the_way_the_sdk_resolves_it() -> None:
    from agents import RunConfig
    from agents.models import get_default_model

    from agent_guardrails.openai_agents import model_name

    bare = Agent(name="a")
    named = Agent(name="a", model="model-on-agent")
    assert model_name(bare) == get_default_model()  # never "default"
    assert model_name(named) == "model-on-agent"
    assert model_name(named, RunConfig(model="model-in-config")) == "model-in-config"
    assert model_name(bare, RunConfig(model=ScriptedModel([], name="scripted-x"))) == "scripted-x"


async def test_an_unpriced_model_stops_a_run_that_has_a_ceiling() -> None:
    hooks = UsageHooks(PriceTable({"some-other-model": ModelPrice(input=1.0, output=1.0)}))
    endless = [[call("get_order", order_id=i)] for i in range(50)]
    agent = Agent(
        name="a",
        model=ScriptedModel(endless, name="unpriced", input_tokens=1000, output_tokens=100),
        tools=[get_order],
    )
    with RunGuard(scope="t", cost_ceiling_usd=5.00) as run:
        result = Runner.run_streamed(agent, "go", hooks=hooks, max_turns=60)
        with pytest.raises(SoftStop) as stop:
            async for _ in watch_run(result, run, poll=0.01):
                pass
    assert stop.value.reason == "cost_ceiling"
    assert "had no price" in str(stop.value)
    assert run.usage is not None
    assert run.usage.unpriced_calls >= 1
