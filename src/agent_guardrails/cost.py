"""What did this run cost, and has it crossed its ceiling?

A daily spend total cannot tell one question from another: a several-fold
per-run regression hides for weeks behind an average taken on a cheap day.
``RunUsage`` puts tokens and dollars on the run itself, including every nested
agent run it caused, and ``over_ceiling()`` is what the stream watchdog polls
to stop a run that has spent its budget.

Rules this module keeps:

* **A missing amount never becomes a real-looking figure.** A call that cannot
  be priced (an unknown model, no rate) increments ``unpriced_calls`` instead
  of adding zero. Its tokens still count. While ``unpriced_calls > 0`` the
  dollar total is a floor, and ``summary()`` says so.
* **A ceiling you can't measure is not a ceiling.** With ``ceiling_usd`` set,
  one unpriced call puts the run over it: the watchdog stops the run rather
  than let spend go uncounted. Price every model your agents use, or leave
  the ceiling unset.
* **No prices are written here.** Rates change and a constant in a library is
  a number nobody re-checks. Pass a :class:`PriceTable` you maintain, or a
  ``cost_usd`` you got from your provider or from LiteLLM.
* **Cached and cache-write tokens are subsets of prompt tokens**, as OpenAI
  and LiteLLM report them. Getting that backwards double-counts the cache.
* **Accounting never breaks a call.** Recording errors are logged and dropped.
* **The accumulator is mutated, never re-``set``.** Nested agent-as-tool runs
  execute in tasks that copy the context, so a nested run adds to the same
  object and its spend lands on the parent run's total.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ModelPrice:
    """US dollars per million tokens. ``cached_input`` and ``cache_write``
    default to the full input rate, which overestimates rather than under."""

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None


class PriceTable:
    """Your rates, by model name. Unknown models are unpriced, never free."""

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def cost(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        price = self._prices.get(model)
        if price is None:
            return None
        fresh = max(0, prompt_tokens - cached_tokens - cache_write_tokens)
        cached_rate = price.cached_input if price.cached_input is not None else price.input
        write_rate = price.cache_write if price.cache_write is not None else price.input
        dollars = (
            fresh * price.input
            + cached_tokens * cached_rate
            + cache_write_tokens * write_rate
            + completion_tokens * price.output
        )
        return dollars / 1_000_000


@dataclasses.dataclass
class RunUsage:
    """One run's model spend across every completion it caused."""

    #: Stop the run once ``cost_usd`` reaches this. ``None``: no ceiling.
    ceiling_usd: float | None = None
    calls: int = 0
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    #: Completions that could not be priced. Non-zero means ``cost_usd`` is a
    #: floor, not a total.
    unpriced_calls: int = 0
    #: Calls per model: which tier actually did the work.
    by_model: dict[str, int] = dataclasses.field(default_factory=dict)

    def add(
        self,
        model: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float | None = None,
        zero_is_priced: bool = False,
    ) -> None:
        """Add one completion.

        ``cost_usd`` of ``None`` is unpriced. A zero from a provider is
        indistinguishable from "no rate on file", so it is unpriced too, unless
        ``zero_is_priced`` says it came from a rate you set (a
        :class:`PriceTable` pricing a call with no tokens).
        """
        self.calls += 1
        self.prompt += max(0, int(prompt_tokens))
        self.completion += max(0, int(completion_tokens))
        self.cached += max(0, int(cached_tokens))
        self.cache_write += max(0, int(cache_write_tokens))
        self.by_model[model] = self.by_model.get(model, 0) + 1
        if cost_usd is None or cost_usd < 0 or (cost_usd == 0 and not zero_is_priced):
            self.unpriced_calls += 1
        else:
            self.cost_usd += float(cost_usd)

    def over_ceiling(self) -> bool:
        """At or past the ceiling, or unable to tell: with a ceiling set, an
        unpriced call means the spend is unknown, and that stops the run."""
        if self.ceiling_usd is None:
            return False
        return self.unpriced_calls > 0 or self.cost_usd >= self.ceiling_usd

    @property
    def fresh_prompt(self) -> int:
        """Prompt tokens that were neither a cache read nor a cache write."""
        return max(0, self.prompt - self.cached - self.cache_write)

    @property
    def cache_read_pct(self) -> float | None:
        """Share of prompt tokens served from cache. ``None`` with no prompt
        tokens: 0.0 would read as "the cache is broken", a different and more
        alarming statement than "nothing ran"."""
        if self.prompt <= 0:
            return None
        return round(100.0 * self.cached / self.prompt, 1)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "cached_tokens": self.cached,
            "cache_write_tokens": self.cache_write,
            "fresh_prompt_tokens": self.fresh_prompt,
            "cache_read_pct": self.cache_read_pct,
            "cost_usd": round(self.cost_usd, 4),
            "ceiling_usd": self.ceiling_usd,
            "over_ceiling": self.over_ceiling(),
            "unpriced_calls": self.unpriced_calls,
            "cost_is_floor": self.unpriced_calls > 0,
            "by_model": dict(self.by_model),
        }


_usage: ContextVar[RunUsage | None] = ContextVar("agent_guardrails_run_usage", default=None)


def bind_run_usage(
    ceiling_usd: float | None = None,
) -> tuple[RunUsage, Token[RunUsage | None]]:
    """Start accounting for one run. Reset with :func:`reset_run_usage`."""
    usage = RunUsage(ceiling_usd=ceiling_usd)
    return usage, _usage.set(usage)


def reset_run_usage(token: Token[RunUsage | None]) -> None:
    try:
        _usage.reset(token)
    except ValueError:  # pragma: no cover - token from another context
        logger.debug("run usage token belonged to another context")


def current_run_usage() -> RunUsage | None:
    return _usage.get()


def over_cost_ceiling() -> bool:
    usage = _usage.get()
    return usage is not None and usage.over_ceiling()


def record_usage(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float | None = None,
    prices: PriceTable | None = None,
) -> None:
    """Add one completion to the current run. A no-op outside a bound run.

    Give ``cost_usd`` when your provider or LiteLLM priced the call, or
    ``prices`` to price it from your table.
    """
    usage = _usage.get()
    if usage is None:
        return
    try:
        from_table = False
        if cost_usd is None and prices is not None:
            cost_usd = prices.cost(
                model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            from_table = cost_usd is not None
        if cost_usd is None and usage.ceiling_usd is not None:
            logger.warning(
                "no price for model %r: the run's cost ceiling can't be enforced, so it "
                "counts as reached",
                model,
            )
        usage.add(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            zero_is_priced=from_table,
        )
    except Exception:
        # Accounting never breaks a call. It does count it: a call that could
        # not be recorded at all is unpriced, so a ceiling still holds.
        logger.debug("usage accounting failed for one call", exc_info=True)
        usage.unpriced_calls += 1


__all__ = [
    "ModelPrice",
    "PriceTable",
    "RunUsage",
    "bind_run_usage",
    "current_run_usage",
    "over_cost_ceiling",
    "record_usage",
    "reset_run_usage",
]
