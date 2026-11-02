from __future__ import annotations

import asyncio

import pytest

from agent_guardrails.cost import (
    ModelPrice,
    PriceTable,
    RunUsage,
    bind_run_usage,
    current_run_usage,
    over_cost_ceiling,
    record_usage,
    reset_run_usage,
)

# Illustrative rates for tests only. This library ships no prices.
PRICES = PriceTable(
    {
        "small": ModelPrice(input=1.0, output=4.0, cached_input=0.1, cache_write=1.25),
        "plain": ModelPrice(input=2.0, output=8.0),
    }
)


def test_price_table_splits_fresh_cached_and_written_tokens() -> None:
    cost = PRICES.cost(
        "small",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        cached_tokens=400_000,
        cache_write_tokens=100_000,
    )
    # 500k fresh x $1 + 400k cached x $0.10 + 100k written x $1.25 + 500k out x $4
    assert cost == pytest.approx(0.5 + 0.04 + 0.125 + 2.0)


def test_missing_cache_rates_charge_the_full_input_rate() -> None:
    cost = PRICES.cost(
        "plain", prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000
    )
    assert cost == pytest.approx(2.0)


def test_an_unknown_model_is_unpriced_not_free() -> None:
    assert PRICES.cost("mystery", prompt_tokens=10, completion_tokens=10) is None
    usage, token = bind_run_usage()
    try:
        record_usage("mystery", prompt_tokens=10, completion_tokens=10, prices=PRICES)
    finally:
        reset_run_usage(token)
    assert usage.unpriced_calls == 1
    assert usage.prompt == 10
    assert usage.summary()["cost_is_floor"] is True


def test_a_zero_cost_is_treated_as_unpriced() -> None:
    usage = RunUsage()
    usage.add("m", prompt_tokens=5, cost_usd=0.0)
    assert usage.unpriced_calls == 1
    assert usage.cost_usd == 0.0


def test_the_ceiling() -> None:
    usage = RunUsage(ceiling_usd=0.01)
    usage.add("m", cost_usd=0.004)
    assert not usage.over_ceiling()
    usage.add("m", cost_usd=0.006)
    assert usage.over_ceiling()
    assert RunUsage().over_ceiling() is False


def test_recording_outside_a_run_is_a_no_op() -> None:
    assert current_run_usage() is None
    record_usage("small", prompt_tokens=1, completion_tokens=1, prices=PRICES)
    assert not over_cost_ceiling()


async def test_nested_tasks_add_to_the_parent_runs_total() -> None:
    # A nested agent run executes in a task that copies the context. It must
    # add to the same object, not a copy.
    usage, token = bind_run_usage(ceiling_usd=1.0)
    try:

        async def nested() -> None:
            record_usage("small", prompt_tokens=1_000, completion_tokens=100, cost_usd=0.6)

        await asyncio.gather(asyncio.create_task(nested()), asyncio.create_task(nested()))
        assert over_cost_ceiling()
    finally:
        reset_run_usage(token)
    assert usage.calls == 2
    assert usage.cost_usd == pytest.approx(1.2)
    assert usage.by_model == {"small": 2}


def test_cache_read_share_is_unknown_with_no_prompt() -> None:
    assert RunUsage().cache_read_pct is None
    usage = RunUsage()
    usage.add("m", prompt_tokens=200, cached_tokens=50, cost_usd=0.01)
    assert usage.cache_read_pct == 25.0
    assert usage.fresh_prompt == 150


def test_accounting_errors_never_break_a_call() -> None:
    class Broken(PriceTable):
        def cost(self, model: str, **kwargs: int) -> float | None:
            raise RuntimeError("bad table")

    usage, token = bind_run_usage()
    try:
        record_usage("small", prompt_tokens=1, prices=Broken({}))
    finally:
        reset_run_usage(token)
    assert usage.calls == 0
    assert usage.unpriced_calls == 1  # not recorded, but not forgotten either


def test_with_a_ceiling_an_unpriced_call_counts_as_over_it() -> None:
    usage, token = bind_run_usage(ceiling_usd=1.00)
    try:
        record_usage("small", prompt_tokens=10, completion_tokens=10, prices=PRICES)
        assert not over_cost_ceiling()
        record_usage("mystery", prompt_tokens=10, completion_tokens=10, prices=PRICES)
        assert over_cost_ceiling()  # spend is unknown now, so the ceiling holds
    finally:
        reset_run_usage(token)
    assert usage.summary()["over_ceiling"] is True


def test_without_a_ceiling_an_unpriced_call_stops_nothing() -> None:
    usage = RunUsage()
    usage.add("mystery", prompt_tokens=10)
    assert usage.unpriced_calls == 1
    assert not usage.over_ceiling()


def test_a_priced_model_with_no_tokens_is_priced_at_zero() -> None:
    usage, token = bind_run_usage(ceiling_usd=1.00)
    try:
        record_usage("small", prompt_tokens=0, completion_tokens=0, prices=PRICES)
    finally:
        reset_run_usage(token)
    assert usage.unpriced_calls == 0
    assert not usage.over_ceiling()
