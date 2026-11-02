"""Adapters for the OpenAI Agents SDK (the ``openai-agents`` extra).

* :func:`gate_tool` / :func:`gate_tools`: every tool that is not provably a
  read asks a person first and runs at most once.
* :func:`require_approval_for_writes`: the same fail-closed classification,
  driving the SDK's own ``needs_approval`` interruption flow instead.
* :class:`UsageHooks`: run hooks that add each model call's tokens and cost
  to the run's :class:`~agent_guardrails.cost.RunUsage`.
* :func:`watch_run`: watch a streamed run and cancel it when a guard stops it.

``budget.budget_tools`` and ``nested.cap_nested_tools`` already work on SDK
tools as they are.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any

from agents import Agent, FunctionTool, RunConfig, RunContextWrapper, RunHooks, Tool
from agents.exceptions import ToolTimeoutError
from agents.items import ModelResponse
from agents.models import get_default_model
from agents.models.interface import Model
from agents.result import RunResultStreaming
from agents.stream_events import StreamEvent
from agents.tool import resolve_function_tool_failure_error_function

from agent_guardrails._sdk import GATED_MARKER, without_failure_handling
from agent_guardrails.classify import DEFAULT_CLASSIFIER, ToolClassifier
from agent_guardrails.confirm import Approval, ConfirmChannel, render_details
from agent_guardrails.cost import PriceTable, record_usage
from agent_guardrails.idempotency import (
    GuardResult,
    PreSideEffectError,
    current_approval,
    derive_key,
    outcome_message,
    run_guarded,
)
from agent_guardrails.run import RunGuard, current_scope
from agent_guardrails.stores import IdempotencyStore
from agent_guardrails.watchdog import GuardStop, watch_stream

logger = logging.getLogger(__name__)

#: How long an identical approved write replays its first result instead of
#: running again.
DEFAULT_WINDOW = timedelta(hours=6)


def _failure_message(exc: BaseException, *, name: str, system: str, timeout_s: float | None) -> str:
    """What the model is told when a gated write raised. The record is already
    written; these words must agree with it."""
    if isinstance(exc, PreSideEffectError):
        reason = str(exc).strip()
        because = f": {reason[:300]}" if reason else ""
        return (
            f"I didn't run {name}: it failed before anything was changed in {system}"
            f"{because}. It is safe to try again once the request is fixed."
        )
    if isinstance(exc, TimeoutError) and timeout_s is not None:
        what = f"it didn't finish within {timeout_s:g} seconds"
    else:
        what = f"it failed partway ({type(exc).__name__})"
    return (
        f"I can't confirm whether {name} went through: {what}. Check {system} before "
        "asking again; I won't retry it on my own."
    )


def gate_tool(
    tool: FunctionTool,
    *,
    channel: ConfirmChannel,
    store: IdempotencyStore | None,
    scope: str | Callable[[], str | None] | None = None,
    window: timedelta | None = DEFAULT_WINDOW,
    classifier: ToolClassifier = DEFAULT_CLASSIFIER,
    editable: Sequence[str] = (),
    render: Callable[[str, Mapping[str, Any]], str] = render_details,
    verify: Callable[[Any], Awaitable[bool]] | None = None,
    required: bool = True,
    system: str = "the connected system",
    namespace: str | None = None,
) -> FunctionTool:
    """Return ``tool`` itself if it is a read, else a gated copy.

    The copy keeps the original's name, schema and approval settings and swaps
    ``on_invoke_tool`` for one that:

    1. parses the arguments (unparseable arguments stay gated);
    2. shows a card built by ``render`` and waits for a person;
    3. runs the original only on an explicit approval, at most once per
       ``scope`` and identical arguments within ``window``;
    4. otherwise returns plain words saying nothing was done, and why.

    The tool body runs with the SDK's failure handling off, so a failure is
    recorded as what it is (retryable for ``PreSideEffectError``, needs-review
    for anything else) and the model is told the same. A tool built with
    ``failure_error_function=None`` re-raises after the record is written.
    The tool's ``timeout`` applies to the body only, never to the person's
    approval wait; a body that times out may have sent, so it is
    needs-review.

    ``scope`` isolates idempotency records: a string, a callable, or ``None``
    to use the bound ``RunGuard``'s scope. With no scope at all and
    ``required=True`` the tool fails closed. ``editable`` names string
    arguments the approver may rewrite (content, never recipients).
    ``namespace`` tells apart two tools with the same name (two connectors'
    ``send_message``) so identical calls to each are separate actions; it
    defaults to the SDK's qualified name for the tool.
    """
    if classifier.is_read(tool.name):
        return tool

    name = tool.name
    original, _ = without_failure_handling(tool)
    # A tool built with failure_error_function=None wants its errors raised.
    reraise = resolve_function_tool_failure_error_function(tool) is None
    timeout_s = tool.timeout_seconds
    raise_on_timeout = tool.timeout_behavior == "raise_exception"
    identity = namespace if namespace is not None else tool.qualified_name

    async def gated_invoke(ctx: Any, payload: str) -> Any:
        parsed_ok = True
        args: dict[str, Any]
        try:
            loaded = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            loaded, parsed_ok = {}, False
        if isinstance(loaded, dict):
            args = loaded
        else:
            args, parsed_ok = {}, False

        run_scope = scope() if callable(scope) else scope
        if run_scope is None:
            run_scope = current_scope()
        extra = identity if identity != name else ""
        key = derive_key(run_scope or "", name, args if parsed_ok else payload, extra=extra)
        fields = [f for f in editable if isinstance(args.get(f), str)] if parsed_ok else []

        async def confirm() -> Approval:
            return await channel.request(
                action=name,
                title=f"Allow {name}?",
                details=render(name, args if parsed_ok else {"arguments": payload}),
                request_body=args if parsed_ok else None,
                editable_fields=fields,
            )

        started = False

        async def execute() -> Any:
            nonlocal started
            started = True
            approval = current_approval()
            body = payload
            if parsed_ok and approval is not None and approval.edits:
                body = json.dumps({**args, **approval.edits})
            if timeout_s is None:
                return await original(ctx, body)
            # The tool's own timeout, around the body alone.
            return await asyncio.wait_for(original(ctx, body), timeout=timeout_s)

        try:
            guard: GuardResult = await run_guarded(
                store=store,
                scope=run_scope or "",
                key=key,
                action=name,
                execute=execute,
                confirm=confirm,
                window=window,
                required=required,
                verify=verify,
            )
        except Exception as exc:
            # The guard has recorded the failure. Say what it means, in the
            # same terms as the record, instead of the SDK's "please try again".
            logger.info("%s failed: %s", name, type(exc).__name__)
            if isinstance(exc, TimeoutError) and raise_on_timeout and timeout_s is not None:
                raise ToolTimeoutError(tool_name=name, timeout_seconds=timeout_s) from exc
            if reraise:
                raise
            if not started:
                # The approval step itself failed: nothing ran.
                return (
                    f"I didn't run {name}: the approval step failed, so nothing was changed "
                    f"in {system}. Ask again to retry."
                )
            return _failure_message(exc, name=name, system=system, timeout_s=timeout_s)
        if guard.succeeded:
            return guard.result
        logger.info("%s not run: %s (%s)", name, guard.outcome, guard.detail)
        return outcome_message(guard, action=f"run {name}", system=system)

    gated = copy.copy(tool)
    gated.on_invoke_tool = gated_invoke
    # The SDK must not put its own clock around the approval wait.
    gated.timeout_seconds = None
    setattr(gated, GATED_MARKER, True)
    return gated


def gate_tools(
    tools: Sequence[FunctionTool],
    *,
    channel: ConfirmChannel,
    store: IdempotencyStore | None,
    **kwargs: Any,
) -> list[Tool]:
    """:func:`gate_tool` over a list: reads pass through, writes are wrapped.
    Returns ``list[Tool]``, which is what ``Agent(tools=...)`` accepts."""
    gated: list[Tool] = [gate_tool(t, channel=channel, store=store, **kwargs) for t in tools]
    writes = [t.name for t, g in zip(tools, gated, strict=True) if g is not t]
    if writes:
        logger.info("gated %d of %d tool(s): %s", len(writes), len(tools), writes)
    return gated


def require_approval_for_writes(
    tools: Sequence[FunctionTool], *, classifier: ToolClassifier = DEFAULT_CLASSIFIER
) -> list[Tool]:
    """Set the SDK's ``needs_approval`` on every tool that is not provably a
    read. Use this if you resume runs with the SDK's interruption flow rather
    than waiting in-band with :func:`gate_tool`. Reads are returned as they are.
    """
    out: list[Tool] = []
    for tool in tools:
        if classifier.is_read(tool.name):
            out.append(tool)
            continue
        gated = copy.copy(tool)
        gated.needs_approval = True
        out.append(gated)
    return out


def _name_of(model: str | Model) -> str:
    if isinstance(model, str):
        return model
    return str(getattr(model, "model", None) or type(model).__name__)


def model_name(agent: Agent[Any], run_config: RunConfig | None = None) -> str:
    """The model a call ran on, resolved the way the SDK resolves it: the run
    config's model first, then the agent's, then the SDK's default."""
    if run_config is not None and run_config.model is not None:
        return _name_of(run_config.model)
    if agent.model is not None:
        return _name_of(agent.model)
    return get_default_model()


class UsageHooks(RunHooks[Any]):
    """Record every model call of a run into the bound ``RunUsage``.

    Pass the same instance to nested agents' ``as_tool(hooks=...)`` so their
    calls count against the parent run's ceiling. ``prices`` turns tokens into
    dollars; without it calls are counted but unpriced. If you pass a
    ``RunConfig`` with a ``model`` to the runner, pass it here too: it decides
    which model ran, and so which price applies.

    With a cost ceiling set, a call that can't be priced stops the run (see
    ``cost``): give every model your agents use a price.
    """

    def __init__(
        self, prices: PriceTable | None = None, *, run_config: RunConfig | None = None
    ) -> None:
        self.prices = prices
        self.run_config = run_config

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        usage = response.usage
        details = usage.input_tokens_details
        record_usage(
            model_name(agent, self.run_config),
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cached_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            cache_write_tokens=int(getattr(details, "cache_write_tokens", 0) or 0),
            prices=self.prices,
        )


async def watch_run(
    result: RunResultStreaming,
    run: RunGuard | None = None,
    *,
    cancel_on_stop: bool = True,
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Yield a streamed run's events under the watchdog.

    On a :class:`~agent_guardrails.watchdog.SoftStop` or ``HardStop`` the run
    is cancelled (unless ``cancel_on_stop=False``) and the stop is re-raised,
    so the caller can write an answer from what was gathered.
    """
    events = result.stream_events()
    watched = run.watch(events, **kwargs) if run is not None else watch_stream(events, **kwargs)
    try:
        async for event in watched:
            yield event
    except GuardStop:
        if cancel_on_stop:
            result.cancel()
        raise


__all__ = [
    "DEFAULT_WINDOW",
    "UsageHooks",
    "gate_tool",
    "gate_tools",
    "model_name",
    "require_approval_for_writes",
    "watch_run",
]
