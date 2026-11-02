"""Stop a streamed run from the outside: stalls, deadlines, loops and cost.

Guards that speak through tool results only reach runs that call tools. The
expensive runs often barely read: their cost is in model turns and nested
consults. A watchdog that polls on a timer sees every run, so it is the one
place a limit cannot be talked past.

``watch_stream`` yields the events of any async stream and raises:

* :class:`SoftStop` when the run should stop and answer from what it has:
  the soft deadline, the refusal ceiling (the run is looping), or the cost
  ceiling. Catch it and write an answer; that is the point of stopping early.
* :class:`HardStop` when the run is dead: no event for ``stall_timeout``, or
  the hard deadline. Nothing more can be done.

Rules it keeps:

* **A person reading an approval card is not a stall.** Both clocks pause while
  a gate is waiting (``paused``).
* **A running tool is not a stall.** The SDK emits nothing between
  ``tool_called`` and ``tool_output``, so a slow tool looks like a dead stream
  on the silence clock alone. While a tool is in flight the stall budget is
  stretched by ``tool_stall_multiplier``, not removed.
* **An approved write in flight is not interrupted by a soft stop**, and the
  hard deadline is stretched by the write grace (see ``write_marker``).
* **The stream is consumed by a pump task**, so the watchdog can time out on
  its queue without cancelling the async generator mid-iteration, which would
  corrupt it.

The watchdog stops reading; it does not cancel the run that produces the
stream. ``openai_agents.watch_run`` does both.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Literal, TypeVar, cast

from agent_guardrails.budget import current_call_ledger
from agent_guardrails.confirm import confirm_waiting
from agent_guardrails.cost import current_run_usage, over_cost_ceiling
from agent_guardrails.write_marker import WRITE_GRACE_MULTIPLIER, write_in_flight

T = TypeVar("T")

SoftReason = Literal["soft_deadline", "refusal_ceiling", "cost_ceiling"]
HardReason = Literal["stalled", "tool_stalled", "hard_deadline"]


class GuardStop(Exception):
    """Base for the watchdog's stops."""


class SoftStop(GuardStop):
    """Stop the run and write an answer from what was gathered."""

    def __init__(self, reason: SoftReason, message: str) -> None:
        super().__init__(message)
        self.reason: SoftReason = reason


class HardStop(GuardStop):
    """The run is stalled or past its hard deadline."""

    def __init__(self, reason: HardReason, message: str) -> None:
        super().__init__(message)
        self.reason: HardReason = reason


def _looping() -> bool:
    ledger = current_call_ledger()
    return ledger is not None and ledger.looping


async def watch_stream(
    stream: AsyncIterable[T],
    *,
    stall_timeout: float = 120.0,
    max_wall_clock: float | None = None,
    soft_fraction: float | None = 0.7,
    poll: float = 1.0,
    tool_stall_multiplier: float = 2.5,
    write_grace: float = WRITE_GRACE_MULTIPLIER,
    paused: Callable[[], bool] = confirm_waiting,
    is_looping: Callable[[], bool] = _looping,
    over_cost: Callable[[], bool] = over_cost_ceiling,
    clocks: dict[str, float] | None = None,
) -> AsyncIterator[T]:
    """Yield ``stream``'s events until it ends or a limit is hit.

    ``max_wall_clock`` is the hard deadline in active seconds (paused time
    excluded); ``None`` disables both deadlines. ``soft_fraction`` places the
    soft deadline at that fraction of it; ``None`` disables the soft one.
    The defaults read the run's bound guards, so a ``RunGuard`` needs no
    extra wiring. ``clocks`` (optional) receives ``active_s`` for the run log.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()
    done = object()

    async def pump() -> None:
        try:
            async for event in stream:
                await queue.put(event)
        except Exception as exc:  # forwarded to the consumer below
            await queue.put(exc)
        finally:
            await queue.put(done)

    pumper = asyncio.create_task(pump())
    soft_at = (
        max_wall_clock * soft_fraction
        if max_wall_clock is not None and soft_fraction is not None
        else None
    )
    idle = 0.0
    active = 0.0
    tools_in_flight = 0
    last_tick = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            waiting_on_person = paused()
            if not waiting_on_person:
                active += now - last_tick
            last_tick = now
            if clocks is not None:
                clocks["active_s"] = round(active, 1)

            writing = write_in_flight()
            if not writing:
                if over_cost():
                    usage = current_run_usage()
                    spent = usage.cost_usd if usage is not None else 0.0
                    if usage is not None and usage.unpriced_calls > 0:
                        raise SoftStop(
                            "cost_ceiling",
                            f"cost ceiling can't be enforced: {usage.unpriced_calls} model "
                            f"call(s) had no price (at least ${spent:.2f} spent), so the run "
                            "was stopped to write an answer from what was already gathered",
                        )
                    raise SoftStop(
                        "cost_ceiling",
                        f"cost ceiling reached (${spent:.2f} spent): stopped to write an "
                        "answer from what was already gathered",
                    )
                if is_looping():
                    raise SoftStop(
                        "refusal_ceiling",
                        "refusal ceiling reached: the run was looping, so it was stopped "
                        "to write an answer from what was already gathered",
                    )
                if soft_at is not None and active >= soft_at:
                    raise SoftStop(
                        "soft_deadline",
                        f"soft deadline reached ({int(soft_at)}s of {int(max_wall_clock or 0)}s): "
                        "stopped to write an answer",
                    )
            if max_wall_clock is not None:
                hard = max_wall_clock * (write_grace if writing else 1.0)
                if active >= hard:
                    raise HardStop("hard_deadline", f"hard deadline of {int(hard)}s exceeded")

            try:
                item = await asyncio.wait_for(queue.get(), timeout=poll)
            except TimeoutError:
                if paused():
                    idle = 0.0
                    continue
                idle += poll
                if tools_in_flight > 0:
                    if idle >= stall_timeout * tool_stall_multiplier:
                        raise HardStop(
                            "tool_stalled", f"a tool ran {int(idle)}s with no stream event"
                        ) from None
                    continue
                if idle >= stall_timeout:
                    raise HardStop("stalled", f"no stream event in {int(idle)}s") from None
                continue

            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            name = getattr(item, "name", None)
            if name == "tool_called":
                tools_in_flight += 1
            elif name == "tool_output":
                tools_in_flight = max(0, tools_in_flight - 1)
            idle = 0.0
            yield cast(T, item)
    finally:
        if not pumper.done():
            pumper.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pumper


__all__ = ["GuardStop", "HardStop", "SoftStop", "watch_stream"]
