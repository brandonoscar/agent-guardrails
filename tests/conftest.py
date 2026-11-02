from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import Any

import pytest

from agent_guardrails.stores import DEFAULT_LEASE, IdempotencyStore, InMemoryIdempotencyStore

DSN = os.environ.get("AGENT_GUARDRAILS_TEST_DSN", "")
REQUIRE_POSTGRES = os.environ.get("AGENT_GUARDRAILS_REQUIRE_POSTGRES") == "1"

try:  # the SDK's tracing would try to export spans; these tests make no network calls
    from agents import set_tracing_disabled

    set_tracing_disabled(True)
except ImportError:  # pragma: no cover - the core runs without the SDK
    pass

MakeStore = Callable[..., Awaitable[IdempotencyStore]]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DSN:
        return
    if REQUIRE_POSTGRES:
        raise pytest.UsageError(
            "AGENT_GUARDRAILS_REQUIRE_POSTGRES=1 but AGENT_GUARDRAILS_TEST_DSN is not set"
        )
    skip = pytest.mark.skip(reason="set AGENT_GUARDRAILS_TEST_DSN to run the Postgres tests")
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def make_store(request: pytest.FixtureRequest) -> AsyncIterator[MakeStore]:
    """Build stores of one kind. Every store test runs against the in-memory
    store and, with a DSN, against Postgres, so the two cannot drift apart."""
    pools: list[Any] = []

    async def make(lease: timedelta = DEFAULT_LEASE) -> IdempotencyStore:
        if request.param == "memory":
            return InMemoryIdempotencyStore(lease=lease)
        import asyncpg

        from agent_guardrails.postgres import PostgresIdempotencyStore, create_schema

        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=12)
        pools.append(pool)
        await create_schema(pool)
        return PostgresIdempotencyStore(pool, lease=lease)

    try:
        yield make
    finally:
        for pool in pools:
            await pool.close()


@pytest.fixture
async def store(make_store: MakeStore) -> IdempotencyStore:
    return await make_store()


@pytest.fixture
def scope() -> str:
    """A fresh scope per test, so tests sharing a database never collide."""
    return f"test-{uuid.uuid4().hex[:12]}"
