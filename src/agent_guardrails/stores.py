"""Where an action's claim lives, so it can fire at most once.

One row per action, keyed by ``(scope, key)``. The row is both the claim that
picks a single runner and the record a repeat replays. ``scope`` is whatever
you isolate by (a customer, an account, a user); results never cross it.

Status machine. There is no generic auto-retried "failed":

* ``in_progress``: claimed; the side effect may be running.
* ``awaiting_approval``: claimed and parked at the approval gate; the side
  effect has provably not started.
* ``succeeded``: done; its result is replayed on a repeat.
* ``failed_retryable``: failed provably before the side effect; safe to retry.
* ``abandoned_unknown``: failed in a way that may have fired the side effect
  (timeout, dropped connection, a 5xx after sending). Never retried by itself;
  a person has to look.
* ``aborted``: the gate was declined or unanswered; nothing fired, so a later
  attempt starts fresh.

The dedup window starts when the action succeeds, not when it was claimed,
so a slow approval or a slow write can't eat into it. A ``succeeded`` row whose
window has ended is taken over by the next claim, so the same action asked for
again next week runs instead of replaying last week's result. A
``failed_retryable`` or ``aborted`` row is reclaimed at once. An
``abandoned_unknown`` row is never taken over, window or not.

Claim ownership. A won claim (or reclaim) carries a fresh fencing ``token``.
Every later change to the row (park, resume, renew, finalize) names that token
and changes nothing if the row has moved on: a caller whose claim was reaped or
released and then reclaimed by someone else can no longer overwrite the new
holder's row. A store instance is one ``owner`` (one per process), and
``release_awaiting()`` releases only that owner's parked claims.

The holder renews its lease while it waits for approval and while it executes
(``guarded_action`` does this), so the reaper only ever reaps the claims of a
process that stopped renewing them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

STATUS_IN_PROGRESS = "in_progress"
STATUS_AWAITING_APPROVAL = "awaiting_approval"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED_RETRYABLE = "failed_retryable"
STATUS_ABANDONED_UNKNOWN = "abandoned_unknown"
STATUS_ABORTED = "aborted"

#: Statuses whose row a new claim may take over once its window has ended.
TAKEOVER_AFTER_WINDOW: frozenset[str] = frozenset(
    {STATUS_SUCCEEDED, STATUS_ABORTED, STATUS_FAILED_RETRYABLE}
)

#: How long a claim may go without its holder renewing it before the reaper
#: treats the holder as dead. The holder renews every third of it.
DEFAULT_LEASE = timedelta(minutes=15)

#: Why an ``aborted`` row was aborted, stored in its ``error``. A caller waiting
#: on another's claim reads it, so "nobody answered" is never reported as "no".
ABORT_DECLINED = "declined"
ABORT_UNANSWERED = "unanswered"
ABORT_NOT_SURFACED = "not_surfaced"
#: The gate failed or the task was cancelled before the side effect started.
ABORT_STOPPED = "stopped_before_execution"


def new_token() -> str:
    """A fresh fencing token for one claim."""
    return uuid.uuid4().hex


@dataclasses.dataclass
class ClaimRow:
    """One action's row, as the guard needs to see it."""

    scope: str
    key: str
    action: str
    status: str
    result: Any | None
    error: str | None
    attempts: int
    lease_until: datetime | None
    #: End of the dedup window. Set when the action succeeds; ``None`` before
    #: that, and for an action claimed with no window.
    expires_at: datetime | None
    #: The window the action asked for, applied at success.
    window: timedelta | None = None
    #: The store instance (process) holding or last holding the claim.
    owner: str | None = None


@dataclasses.dataclass
class ClaimResult:
    """``won`` is True for the single caller that must go on to approve and
    execute; ``token`` is then its fencing token. Otherwise ``row`` is the
    existing row to branch on and ``token`` is ``None``."""

    won: bool
    row: ClaimRow
    token: str | None = None


class IdempotencyStore(Protocol):
    """What the guard needs from storage.

    Every method that changes a claimed row takes the claim's ``token`` and
    returns ``False`` (changing nothing) when the row is no longer held under
    that token.
    """

    @property
    def lease(self) -> timedelta: ...

    async def claim(
        self, *, scope: str, key: str, action: str, window: timedelta | None
    ) -> ClaimResult: ...

    async def get(self, *, scope: str, key: str) -> ClaimRow | None: ...

    async def finalize_success(self, *, scope: str, key: str, token: str, result: Any) -> bool: ...

    async def finalize_retryable(self, *, scope: str, key: str, token: str, error: str) -> bool: ...

    async def finalize_abandoned(self, *, scope: str, key: str, token: str, error: str) -> bool: ...

    async def finalize_aborted(self, *, scope: str, key: str, token: str, reason: str) -> bool: ...

    async def reclaim(self, *, scope: str, key: str, window: timedelta | None) -> str | None: ...

    async def mark_awaiting_approval(self, *, scope: str, key: str, token: str) -> bool: ...

    async def resume_from_awaiting(self, *, scope: str, key: str, token: str) -> bool: ...

    async def renew(self, *, scope: str, key: str, token: str) -> bool: ...

    async def reap_stale(self) -> int: ...

    async def release_awaiting(self) -> int: ...


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryIdempotencyStore:
    """A process-local store. Right for tests, scripts and single-process
    apps; it forgets everything on restart, so a multi-process or restartable
    deployment wants a durable store such as ``PostgresIdempotencyStore``."""

    def __init__(self, *, lease: timedelta = DEFAULT_LEASE, owner: str | None = None) -> None:
        self._lease = lease
        #: This store instance's identity; ``release_awaiting`` touches only
        #: claims it made.
        self.owner = owner or new_token()
        self._rows: dict[tuple[str, str], ClaimRow] = {}
        self._tokens: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        #: Overridable clock, for tests of leases and windows.
        self.now = _now

    @property
    def lease(self) -> timedelta:
        return self._lease

    async def claim(
        self, *, scope: str, key: str, action: str, window: timedelta | None
    ) -> ClaimResult:
        async with self._lock:
            now = self.now()
            existing = self._rows.get((scope, key))
            if existing is not None and not (
                existing.status in TAKEOVER_AFTER_WINDOW
                and existing.expires_at is not None
                and existing.expires_at < now
            ):
                return ClaimResult(won=False, row=dataclasses.replace(existing))
            token = new_token()
            row = ClaimRow(
                scope=scope,
                key=key,
                action=action,
                status=STATUS_IN_PROGRESS,
                result=None,
                error=None,
                attempts=1,
                lease_until=now + self._lease,
                expires_at=None,
                window=window,
                owner=self.owner,
            )
            self._rows[(scope, key)] = row
            self._tokens[(scope, key)] = token
            return ClaimResult(won=True, row=dataclasses.replace(row), token=token)

    async def get(self, *, scope: str, key: str) -> ClaimRow | None:
        row = self._rows.get((scope, key))
        return dataclasses.replace(row) if row is not None else None

    def _held(self, scope: str, key: str, token: str) -> ClaimRow | None:
        row = self._rows.get((scope, key))
        if row is None or self._tokens.get((scope, key)) != token:
            return None
        return row

    async def _finalize(
        self,
        scope: str,
        key: str,
        token: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> bool:
        async with self._lock:
            row = self._held(scope, key, token)
            if row is None:
                return False
            row.status = status
            row.result = result
            row.error = error
            if status == STATUS_SUCCEEDED and row.window is not None:
                row.expires_at = self.now() + row.window
            return True

    async def finalize_success(self, *, scope: str, key: str, token: str, result: Any) -> bool:
        return await self._finalize(scope, key, token, STATUS_SUCCEEDED, result=result)

    async def finalize_retryable(self, *, scope: str, key: str, token: str, error: str) -> bool:
        return await self._finalize(scope, key, token, STATUS_FAILED_RETRYABLE, error=error)

    async def finalize_abandoned(self, *, scope: str, key: str, token: str, error: str) -> bool:
        return await self._finalize(scope, key, token, STATUS_ABANDONED_UNKNOWN, error=error)

    async def finalize_aborted(self, *, scope: str, key: str, token: str, reason: str) -> bool:
        return await self._finalize(scope, key, token, STATUS_ABORTED, error=reason)

    async def reclaim(self, *, scope: str, key: str, window: timedelta | None) -> str | None:
        """Flip a ``failed_retryable`` or ``aborted`` row back to
        ``in_progress`` under a new token. One winner; only it gets a token."""
        async with self._lock:
            row = self._rows.get((scope, key))
            if row is None or row.status not in (STATUS_FAILED_RETRYABLE, STATUS_ABORTED):
                return None
            token = new_token()
            row.status = STATUS_IN_PROGRESS
            row.attempts += 1
            row.error = None
            row.result = None
            row.lease_until = self.now() + self._lease
            row.expires_at = None
            row.window = window
            row.owner = self.owner
            self._tokens[(scope, key)] = token
            return token

    async def _flip(self, scope: str, key: str, token: str, frm: str, to: str) -> bool:
        async with self._lock:
            row = self._held(scope, key, token)
            if row is None or row.status != frm:
                return False
            row.status = to
            row.lease_until = self.now() + self._lease
            return True

    async def mark_awaiting_approval(self, *, scope: str, key: str, token: str) -> bool:
        return await self._flip(scope, key, token, STATUS_IN_PROGRESS, STATUS_AWAITING_APPROVAL)

    async def resume_from_awaiting(self, *, scope: str, key: str, token: str) -> bool:
        return await self._flip(scope, key, token, STATUS_AWAITING_APPROVAL, STATUS_IN_PROGRESS)

    async def renew(self, *, scope: str, key: str, token: str) -> bool:
        """Push the lease out again. ``False`` once the claim is no longer held."""
        async with self._lock:
            row = self._held(scope, key, token)
            if row is None or row.status not in (STATUS_IN_PROGRESS, STATUS_AWAITING_APPROVAL):
                return False
            row.lease_until = self.now() + self._lease
            return True

    async def reap_stale(self) -> int:
        """Expire dead claims, split by what can be proved.

        An expired ``in_progress`` claim may have fired its side effect, so it
        becomes ``abandoned_unknown``. An expired ``awaiting_approval`` claim
        provably did not, so it becomes ``failed_retryable`` and the user's
        retry simply works.
        """
        async with self._lock:
            now = self.now()
            reaped = 0
            for row in self._rows.values():
                if row.lease_until is None or row.lease_until >= now:
                    continue
                if row.status == STATUS_IN_PROGRESS:
                    row.status = STATUS_ABANDONED_UNKNOWN
                    row.error = row.error or "reaped: in_progress lease expired"
                    reaped += 1
                elif row.status == STATUS_AWAITING_APPROVAL:
                    row.status = STATUS_FAILED_RETRYABLE
                    row.error = row.error or "reaped: awaiting_approval lease expired"
                    reaped += 1
            return reaped

    async def release_awaiting(self) -> int:
        """At shutdown: this store's parked claims become retryable at once, so
        a retry works as soon as the next process is up. Claims parked by any
        other store instance are left alone."""
        async with self._lock:
            released = 0
            for row in self._rows.values():
                if row.status == STATUS_AWAITING_APPROVAL and row.owner == self.owner:
                    row.status = STATUS_FAILED_RETRYABLE
                    row.error = row.error or "released: process shutdown during approval wait"
                    released += 1
            return released


__all__ = [
    "ABORT_DECLINED",
    "ABORT_NOT_SURFACED",
    "ABORT_STOPPED",
    "ABORT_UNANSWERED",
    "DEFAULT_LEASE",
    "STATUS_ABANDONED_UNKNOWN",
    "STATUS_ABORTED",
    "STATUS_AWAITING_APPROVAL",
    "STATUS_FAILED_RETRYABLE",
    "STATUS_IN_PROGRESS",
    "STATUS_SUCCEEDED",
    "TAKEOVER_AFTER_WINDOW",
    "ClaimResult",
    "ClaimRow",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "new_token",
]
