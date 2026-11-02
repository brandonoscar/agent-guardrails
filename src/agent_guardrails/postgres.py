"""A durable ``IdempotencyStore`` on Postgres (asyncpg).

Install the ``postgres`` extra and create the table once with
:data:`SCHEMA_SQL` (or :func:`create_schema`). Every time comparison uses the
database's clock, so app servers with skewed clocks agree on leases and
windows.

The claim is a single statement. ``INSERT ... ON CONFLICT DO UPDATE ... WHERE``
takes the row over only when its window has ended and its status allows it;
otherwise nothing is returned and the caller reads the existing row. Under
concurrency exactly one caller gets a row back. Every later update of a held
row names the claim's fencing token (see ``stores``), so a caller that lost
its claim can't overwrite the row of whoever holds it now.

One store instance per process: ``owner`` identifies it, and
``release_awaiting()`` at shutdown releases only that owner's parked claims.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from agent_guardrails.stores import DEFAULT_LEASE, ClaimResult, ClaimRow, new_token

TABLE = "agent_action_idempotency"

SCHEMA_SQL = f"""
create table if not exists {TABLE} (
    scope        text        not null,
    idem_key     text        not null,  -- sha256 hex of the full normalized action
    action       text        not null,
    status       text        not null check (status in (
                     'in_progress', 'awaiting_approval', 'succeeded',
                     'failed_retryable', 'abandoned_unknown', 'aborted')),
    result       jsonb,                 -- replayed on a repeat
    error        text,
    attempts     integer     not null default 1,
    lease_until  timestamptz,           -- reaper input; the holder renews it
    expires_at   timestamptz,           -- end of the dedup window, set at success; null = none
    dedup_window interval,              -- the window the action asked for
    claim_token  text,                  -- fencing token of the current holder
    owner        text,                  -- store instance (process) of the holder
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    primary key (scope, idem_key)
);
create index if not exists {TABLE}_status_lease_idx on {TABLE} (status, lease_until);
"""

_RETURNING = (
    "scope, idem_key, action, status, result, error, attempts, lease_until, expires_at, "
    "dedup_window, owner"
)


async def create_schema(pool: Any) -> None:
    """Create the table and index if they do not exist."""
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


def _row(record: Any) -> ClaimRow:
    raw = record["result"]
    return ClaimRow(
        scope=record["scope"],
        key=record["idem_key"],
        action=record["action"],
        status=record["status"],
        result=json.loads(raw) if isinstance(raw, str) else raw,
        error=record["error"],
        attempts=record["attempts"],
        lease_until=record["lease_until"],
        expires_at=record["expires_at"],
        window=record["dedup_window"],
        owner=record["owner"],
    )


def _count(status: str) -> int:
    """asyncpg returns command tags such as ``UPDATE 3``."""
    try:
        return int(status.split()[-1])
    except (ValueError, IndexError):
        return 0


class PostgresIdempotencyStore:
    """asyncpg-backed store. Pass an ``asyncpg.Pool``."""

    def __init__(
        self, pool: Any, *, lease: timedelta = DEFAULT_LEASE, owner: str | None = None
    ) -> None:
        self._pool = pool
        self._lease = lease
        #: This store instance's identity; ``release_awaiting`` touches only
        #: claims it made.
        self.owner = owner or new_token()

    @property
    def lease(self) -> timedelta:
        return self._lease

    async def claim(
        self, *, scope: str, key: str, action: str, window: timedelta | None
    ) -> ClaimResult:
        token = new_token()
        async with self._pool.acquire() as conn:
            won = await conn.fetchrow(
                f"""
                insert into {TABLE} as t
                    (scope, idem_key, action, status, lease_until, expires_at,
                     dedup_window, claim_token, owner)
                values ($1, $2, $3, 'in_progress', now() + $4::interval, null,
                        $5::interval, $6, $7)
                on conflict (scope, idem_key) do update
                   set action = excluded.action,
                       status = 'in_progress',
                       result = null,
                       error = null,
                       attempts = 1,
                       lease_until = excluded.lease_until,
                       expires_at = null,
                       dedup_window = excluded.dedup_window,
                       claim_token = excluded.claim_token,
                       owner = excluded.owner,
                       updated_at = now()
                 where t.expires_at is not null
                   and t.expires_at < now()
                   and t.status in ('succeeded', 'aborted', 'failed_retryable')
                returning {_RETURNING}
                """,
                scope,
                key,
                action,
                self._lease,
                window,
                token,
                self.owner,
            )
            if won is not None:
                return ClaimResult(won=True, row=_row(won), token=token)
            existing = await conn.fetchrow(
                f"select {_RETURNING} from {TABLE} where scope = $1 and idem_key = $2",
                scope,
                key,
            )
            if existing is None:  # deleted between the two statements
                raise RuntimeError(f"claim row for {action} vanished; retry the claim")
            return ClaimResult(won=False, row=_row(existing))

    async def get(self, *, scope: str, key: str) -> ClaimRow | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                f"select {_RETURNING} from {TABLE} where scope = $1 and idem_key = $2",
                scope,
                key,
            )
            return _row(record) if record is not None else None

    async def _finalize(
        self, scope: str, key: str, token: str, status: str, result: Any, error: str | None
    ) -> bool:
        async with self._pool.acquire() as conn:
            done = await conn.execute(
                f"""
                update {TABLE}
                   set status = $4,
                       result = $5::jsonb,
                       error = $6,
                       -- The window starts at success, so approval and
                       -- execution time never eat into it.
                       expires_at = case when $4 = 'succeeded'
                                         then now() + dedup_window
                                         else expires_at end,
                       updated_at = now()
                 where scope = $1 and idem_key = $2 and claim_token = $3
                """,
                scope,
                key,
                token,
                status,
                json.dumps(result, default=str) if result is not None else None,
                error,
            )
            return _count(done) == 1

    async def finalize_success(self, *, scope: str, key: str, token: str, result: Any) -> bool:
        return await self._finalize(scope, key, token, "succeeded", result, None)

    async def finalize_retryable(self, *, scope: str, key: str, token: str, error: str) -> bool:
        return await self._finalize(scope, key, token, "failed_retryable", None, error)

    async def finalize_abandoned(self, *, scope: str, key: str, token: str, error: str) -> bool:
        return await self._finalize(scope, key, token, "abandoned_unknown", None, error)

    async def finalize_aborted(self, *, scope: str, key: str, token: str, reason: str) -> bool:
        return await self._finalize(scope, key, token, "aborted", None, reason)

    async def reclaim(self, *, scope: str, key: str, window: timedelta | None) -> str | None:
        token = new_token()
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                f"""
                update {TABLE}
                   set status = 'in_progress', attempts = attempts + 1, error = null,
                       result = null, expires_at = null, dedup_window = $4::interval,
                       claim_token = $5, owner = $6,
                       lease_until = now() + $3::interval, updated_at = now()
                 where scope = $1 and idem_key = $2
                   and status in ('failed_retryable', 'aborted')
                returning scope
                """,
                scope,
                key,
                self._lease,
                window,
                token,
                self.owner,
            )
            return token if record is not None else None

    async def _flip(self, scope: str, key: str, token: str, frm: str, to: str) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                f"""
                update {TABLE}
                   set status = $5, lease_until = now() + $6::interval, updated_at = now()
                 where scope = $1 and idem_key = $2 and claim_token = $3 and status = $4
                """,
                scope,
                key,
                token,
                frm,
                to,
                self._lease,
            )
            return _count(status) == 1

    async def mark_awaiting_approval(self, *, scope: str, key: str, token: str) -> bool:
        return await self._flip(scope, key, token, "in_progress", "awaiting_approval")

    async def resume_from_awaiting(self, *, scope: str, key: str, token: str) -> bool:
        return await self._flip(scope, key, token, "awaiting_approval", "in_progress")

    async def renew(self, *, scope: str, key: str, token: str) -> bool:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                f"""
                update {TABLE}
                   set lease_until = now() + $4::interval, updated_at = now()
                 where scope = $1 and idem_key = $2 and claim_token = $3
                   and status in ('in_progress', 'awaiting_approval')
                """,
                scope,
                key,
                token,
                self._lease,
            )
            return _count(status) == 1

    async def reap_stale(self) -> int:
        async with self._pool.acquire() as conn:
            ambiguous = await conn.execute(
                f"""
                update {TABLE}
                   set status = 'abandoned_unknown',
                       error = coalesce(error, 'reaped: in_progress lease expired'),
                       updated_at = now()
                 where status = 'in_progress' and lease_until < now()
                """
            )
            parked = await conn.execute(
                f"""
                update {TABLE}
                   set status = 'failed_retryable',
                       error = coalesce(error, 'reaped: awaiting_approval lease expired'),
                       updated_at = now()
                 where status = 'awaiting_approval' and lease_until < now()
                """
            )
            return _count(ambiguous) + _count(parked)

    async def release_awaiting(self) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                f"""
                update {TABLE}
                   set status = 'failed_retryable',
                       error = coalesce(error, 'released: process shutdown during approval wait'),
                       updated_at = now()
                 where status = 'awaiting_approval' and owner = $1
                """,
                self.owner,
            )
            return _count(status)


__all__ = ["SCHEMA_SQL", "TABLE", "PostgresIdempotencyStore", "create_schema"]
