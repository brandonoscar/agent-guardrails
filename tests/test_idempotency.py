"""At-most-once execution. Every test runs against each store (see conftest)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent_guardrails.confirm import Approval, ApprovalStatus
from agent_guardrails.idempotency import (
    GuardResult,
    PreSideEffectError,
    current_approval,
    derive_key,
    guarded_action,
    outcome_message,
    run_guarded,
)
from agent_guardrails.stores import (
    ABORT_STOPPED,
    STATUS_ABANDONED_UNKNOWN,
    STATUS_ABORTED,
    STATUS_AWAITING_APPROVAL,
    STATUS_FAILED_RETRYABLE,
    STATUS_IN_PROGRESS,
    STATUS_SUCCEEDED,
    IdempotencyStore,
)
from agent_guardrails.write_marker import bind_write_marker, unbind_write_marker, write_in_flight

from .conftest import MakeStore

WINDOW = timedelta(hours=6)


def answer(
    status: ApprovalStatus, edits: dict[str, str] | None = None
) -> Callable[[], Awaitable[Approval]]:
    async def confirm() -> Approval:
        return Approval(status, "card-1", edits)

    return confirm


class Counter:
    def __init__(self, result: Any = "done", error: BaseException | None = None) -> None:
        self.calls = 0
        self.result = result
        self.error = error

    async def __call__(self) -> Any:
        self.calls += 1
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return self.result


async def run(
    store: IdempotencyStore,
    scope: str,
    execute: Callable[[], Awaitable[Any]],
    *,
    key: str = "k1",
    confirm: Callable[[], Awaitable[Approval]] | None = None,
    **kwargs: Any,
) -> GuardResult:
    return await guarded_action(
        store=store,
        scope=scope,
        key=key,
        action="send_email",
        execute=execute,
        confirm=confirm if confirm is not None else answer("approved"),
        window=kwargs.pop("window", WINDOW),
        **kwargs,
    )


async def test_double_approval_fires_once(store: IdempotencyStore, scope: str) -> None:
    execute = Counter("sent-123")
    first = await run(store, scope, execute)
    second = await run(store, scope, execute)
    assert (first.outcome, second.outcome) == ("executed", "replayed")
    assert second.result == "sent-123"
    assert execute.calls == 1


async def test_concurrent_duplicates_execute_once(store: IdempotencyStore, scope: str) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "once"

    first = asyncio.create_task(run(store, scope, slow))
    await started.wait()
    others = [asyncio.create_task(run(store, scope, slow, poll=0.01)) for _ in range(5)]
    await asyncio.sleep(0.05)
    release.set()
    results = await asyncio.gather(first, *others)
    assert calls == 1
    assert results[0].outcome == "executed"
    assert {r.outcome for r in results[1:]} == {"replayed"}


async def test_distinct_actions_both_run(store: IdempotencyStore, scope: str) -> None:
    execute = Counter()
    a = await run(store, scope, execute, key=derive_key(scope, "send_email", {"to": "a"}))
    b = await run(store, scope, execute, key=derive_key(scope, "send_email", {"to": "b"}))
    assert (a.outcome, b.outcome) == ("executed", "executed")
    assert execute.calls == 2


async def test_pre_side_effect_failure_is_retryable(store: IdempotencyStore, scope: str) -> None:
    failing = Counter(error=PreSideEffectError("bad address"))
    with pytest.raises(PreSideEffectError):
        await run(store, scope, failing)
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_FAILED_RETRYABLE
    working = Counter("sent")
    retry = await run(store, scope, working)
    assert retry.outcome == "executed"
    assert working.calls == 1
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.attempts == 2


async def test_ambiguous_failure_is_never_refired(store: IdempotencyStore, scope: str) -> None:
    class ApiError(Exception):
        status = 502
        path = "/messages"

    failing = Counter(error=ApiError("gateway timeout after send"))
    with pytest.raises(ApiError):
        await run(store, scope, failing)
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN
    assert row.error is not None
    assert row.error.startswith("ambiguous:ApiError status=502 path=/messages")
    assert "gateway timeout after send" in row.error

    retry = Counter()
    again = await run(store, scope, retry)
    assert again.outcome == "blocked_needs_review"
    assert retry.calls == 0


async def test_cancellation_mid_write_is_ambiguous(store: IdempotencyStore, scope: str) -> None:
    started = asyncio.Event()

    async def hangs() -> str:
        started.set()
        await asyncio.sleep(10)
        return "never"

    task = asyncio.create_task(run(store, scope, hangs))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN


@pytest.mark.parametrize(
    ("status", "outcome"),
    [("declined", "declined"), ("unanswered", "unanswered"), ("not_surfaced", "unanswered")],
)
async def test_not_approved_runs_nothing_and_a_later_attempt_is_fresh(
    store: IdempotencyStore, scope: str, status: ApprovalStatus, outcome: str
) -> None:
    execute = Counter()
    first = await run(store, scope, execute, confirm=answer(status))
    assert first.outcome == outcome
    assert execute.calls == 0
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABORTED

    second = await run(store, scope, execute)
    assert second.outcome == "executed"
    assert execute.calls == 1


async def test_a_gate_that_raises_aborts(store: IdempotencyStore, scope: str) -> None:
    async def broken() -> Approval:
        raise RuntimeError("card service down")

    execute = Counter()
    with pytest.raises(RuntimeError):
        await run(store, scope, execute, confirm=broken)
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABORTED
    assert execute.calls == 0


async def test_out_of_band_approval_skips_the_gate(store: IdempotencyStore, scope: str) -> None:
    execute = Counter()
    result = await guarded_action(
        store=store,
        scope=scope,
        key="k1",
        action="send_email",
        execute=execute,
        confirm=None,
        window=WINDOW,
    )
    assert result.outcome == "executed"


async def test_verify_false_or_raising_is_needs_review_and_not_rerun(
    store: IdempotencyStore, scope: str
) -> None:
    async def not_found(result: Any) -> bool:
        return False

    async def errors(result: Any) -> bool:
        raise TimeoutError

    execute = Counter()
    a = await run(store, scope, execute, key="a", verify=not_found)
    b = await run(store, scope, execute, key="b", verify=errors, verify_fail_tag="x")
    assert (a.outcome, a.detail) == ("blocked_needs_review", "verify_failed:not_found")
    assert (b.outcome, b.detail) == ("blocked_needs_review", "verify_error:TimeoutError")
    again = await run(store, scope, execute, key="a")
    assert again.outcome == "blocked_needs_review"
    assert execute.calls == 2


async def test_verify_true_records_success(store: IdempotencyStore, scope: str) -> None:
    async def found(result: Any) -> bool:
        return bool(result == "done")

    result = await run(store, scope, Counter(), verify=found)
    assert result.outcome == "executed"
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_SUCCEEDED


async def test_record_sees_fresh_runs_and_replays(store: IdempotencyStore, scope: str) -> None:
    seen: list[str] = []

    async def record(result: GuardResult) -> None:
        seen.append(result.outcome)

    await run(store, scope, Counter(), record=record)
    await run(store, scope, Counter(), record=record)
    assert seen == ["executed", "replayed"]


async def test_execute_sees_the_approval_and_the_write_marker(
    store: IdempotencyStore, scope: str
) -> None:
    seen: dict[str, Any] = {}

    async def execute() -> str:
        approval = current_approval()
        seen["edits"] = approval.edits if approval else None
        seen["writing"] = write_in_flight()
        return "ok"

    _, token = bind_write_marker()
    try:
        await run(store, scope, execute, confirm=answer("approved", {"body": "edited"}))
        assert not write_in_flight()
    finally:
        unbind_write_marker(token)
    assert seen == {"edits": {"body": "edited"}, "writing": True}
    assert current_approval() is None


async def test_the_window_ends_and_the_action_runs_again(
    store: IdempotencyStore, scope: str
) -> None:
    execute = Counter()
    short = timedelta(milliseconds=50)
    assert (await run(store, scope, execute, window=short)).outcome == "executed"
    assert (await run(store, scope, execute, window=short)).outcome == "replayed"
    await asyncio.sleep(0.15)
    assert (await run(store, scope, execute, window=short)).outcome == "executed"
    assert execute.calls == 2


async def test_no_window_replays_forever(store: IdempotencyStore, scope: str) -> None:
    execute = Counter()
    await run(store, scope, execute, window=None)
    await asyncio.sleep(0.05)
    assert (await run(store, scope, execute, window=None)).outcome == "replayed"
    assert execute.calls == 1


async def test_an_abandoned_action_stays_blocked_after_its_window(
    store: IdempotencyStore, scope: str
) -> None:
    short = timedelta(milliseconds=50)
    with pytest.raises(ConnectionError):
        await run(store, scope, Counter(error=ConnectionError()), window=short)
    await asyncio.sleep(0.15)
    retry = Counter()
    assert (await run(store, scope, retry, window=short)).outcome == "blocked_needs_review"
    assert retry.calls == 0


async def test_reaper_splits_dead_claims_by_what_can_be_proved(
    make_store: MakeStore, scope: str
) -> None:
    store = await make_store(lease=timedelta(milliseconds=50))
    running = await store.claim(scope=scope, key="running", action="a", window=WINDOW)
    parked = await store.claim(scope=scope, key="parked", action="a", window=WINDOW)
    assert running.won
    assert parked.won
    assert parked.token is not None
    assert await store.mark_awaiting_approval(scope=scope, key="parked", token=parked.token)
    await asyncio.sleep(0.15)
    # The reaper is table-wide: on a shared database another process's reaper
    # may get to these rows first, so check where they ended up, not who
    # moved them.
    await store.reap_stale()
    running_row = await store.get(scope=scope, key="running")
    parked_row = await store.get(scope=scope, key="parked")
    assert running_row is not None
    assert parked_row is not None
    # In progress may have fired: needs review. Parked at the gate did not: retry.
    assert running_row.status == STATUS_ABANDONED_UNKNOWN
    assert parked_row.status == STATUS_FAILED_RETRYABLE


async def test_shutdown_releases_parked_claims(store: IdempotencyStore, scope: str) -> None:
    parked_claim = await store.claim(scope=scope, key="parked", action="a", window=WINDOW)
    assert parked_claim.token is not None
    await store.mark_awaiting_approval(scope=scope, key="parked", token=parked_claim.token)
    await store.claim(scope=scope, key="running", action="a", window=WINDOW)
    assert await store.release_awaiting() >= 1
    parked = await store.get(scope=scope, key="parked")
    running = await store.get(scope=scope, key="running")
    assert parked is not None
    assert running is not None
    assert parked.status == STATUS_FAILED_RETRYABLE
    assert running.status == STATUS_IN_PROGRESS


async def test_a_claim_released_during_approval_is_not_executed(
    store: IdempotencyStore, scope: str
) -> None:
    async def approve_after_shutdown() -> Approval:
        await store.release_awaiting()  # the process "shut down" mid-wait
        return Approval("approved", "card-1")

    execute = Counter()
    result = await run(store, scope, execute, confirm=approve_after_shutdown)
    assert result.outcome == "already_pending"
    assert execute.calls == 0
    assert "ask me again" in outcome_message(result, action="send it")


async def test_concurrent_claims_have_one_winner(store: IdempotencyStore, scope: str) -> None:
    claims = await asyncio.gather(
        *[store.claim(scope=scope, key="k", action="a", window=WINDOW) for _ in range(10)]
    )
    assert sum(c.won for c in claims) == 1


async def test_concurrent_reclaims_have_one_winner(store: IdempotencyStore, scope: str) -> None:
    claim = await store.claim(scope=scope, key="k", action="a", window=WINDOW)
    assert claim.token is not None
    await store.finalize_aborted(scope=scope, key="k", token=claim.token, reason="declined")
    tokens = await asyncio.gather(
        *[store.reclaim(scope=scope, key="k", window=WINDOW) for _ in range(10)]
    )
    won = [t for t in tokens if t is not None]
    assert len(won) == 1
    assert won[0] != claim.token


# ── claim ownership (fencing) ──────────────────────────────────────────────


async def test_a_stale_holder_changes_nothing_after_its_claim_moves_on(
    store: IdempotencyStore, scope: str
) -> None:
    first = await store.claim(scope=scope, key="k", action="a", window=WINDOW)
    stale = first.token
    assert stale is not None
    assert await store.mark_awaiting_approval(scope=scope, key="k", token=stale)
    assert await store.release_awaiting() >= 1  # the claim is released
    fresh = await store.reclaim(scope=scope, key="k", window=WINDOW)  # and taken by another
    assert fresh is not None
    assert await store.finalize_success(scope=scope, key="k", token=fresh, result="sent")
    # The first holder's card expires now. None of its writes may land.
    assert not await store.finalize_aborted(scope=scope, key="k", token=stale, reason="x")
    assert not await store.finalize_abandoned(scope=scope, key="k", token=stale, error="x")
    assert not await store.finalize_retryable(scope=scope, key="k", token=stale, error="x")
    assert not await store.finalize_success(scope=scope, key="k", token=stale, result="y")
    assert not await store.mark_awaiting_approval(scope=scope, key="k", token=stale)
    assert not await store.resume_from_awaiting(scope=scope, key="k", token=stale)
    assert not await store.renew(scope=scope, key="k", token=stale)
    row = await store.get(scope=scope, key="k")
    assert row is not None
    assert (row.status, row.result) == (STATUS_SUCCEEDED, "sent")


async def test_a_released_claim_and_a_late_expiry_never_run_the_action_twice(
    store: IdempotencyStore, scope: str
) -> None:
    """The reviewer's sequence: A parks, a shutdown releases it, B reclaims and
    runs, A's card expires, then C asks again. One execution."""
    execute = Counter("sent")
    a_parked = asyncio.Event()
    a_expire = asyncio.Event()

    async def a_gate() -> Approval:
        a_parked.set()
        await a_expire.wait()
        return Approval("unanswered", "card-a")

    a = asyncio.create_task(run(store, scope, execute, confirm=a_gate))
    await a_parked.wait()
    await store.release_awaiting()
    b = await run(store, scope, execute)
    a_expire.set()
    a_result = await a
    c = await run(store, scope, execute)
    assert (a_result.outcome, b.outcome, c.outcome) == ("unanswered", "executed", "replayed")
    assert execute.calls == 1


async def test_release_awaiting_touches_only_this_stores_claims(
    make_store: MakeStore, scope: str
) -> None:
    mine = await make_store()
    other = await make_store()  # another process on the same table
    claim = await mine.claim(scope=scope, key="k", action="a", window=WINDOW)
    assert claim.token is not None
    assert await mine.mark_awaiting_approval(scope=scope, key="k", token=claim.token)
    await other.release_awaiting()
    row = await mine.get(scope=scope, key="k")
    assert row is not None
    assert row.status == STATUS_AWAITING_APPROVAL
    assert await mine.release_awaiting() >= 1
    row = await mine.get(scope=scope, key="k")
    assert row is not None
    assert row.status == STATUS_FAILED_RETRYABLE


async def test_the_holder_renews_its_lease_so_a_slow_approval_is_never_reaped(
    make_store: MakeStore, scope: str
) -> None:
    # Margins wide enough for a loaded CI box: the holder renews every 0.2 s,
    # so only a stall of more than 0.4 s could let the reaper in.
    store = await make_store(lease=timedelta(milliseconds=600))
    execute = Counter("sent")
    seen: set[str] = set()

    async def slow_gate() -> Approval:
        for _ in range(15):  # 1.5 s of approval wait, two and a half leases long
            await asyncio.sleep(0.1)
            await store.reap_stale()
            row = await store.get(scope=scope, key="k1")
            assert row is not None
            seen.add(row.status)
        return Approval("approved", "card-1")

    result = await run(store, scope, execute, confirm=slow_gate)
    assert result.outcome == "executed"
    assert seen == {STATUS_AWAITING_APPROVAL}  # never reaped while its holder lived
    assert execute.calls == 1


# ── what the outcome and the row say ──────────────────────────────────────


@pytest.mark.parametrize(
    ("peer", "outcome"),
    [("declined", "declined"), ("unanswered", "unanswered"), ("not_surfaced", "unanswered")],
)
async def test_a_waiting_duplicate_reports_why_the_peer_stopped(
    store: IdempotencyStore, scope: str, peer: ApprovalStatus, outcome: str
) -> None:
    execute = Counter()
    parked = asyncio.Event()
    answer_now = asyncio.Event()

    async def peer_gate() -> Approval:
        parked.set()
        await answer_now.wait()
        return Approval(peer, "card-a")

    first = asyncio.create_task(run(store, scope, execute, confirm=peer_gate))
    await parked.wait()
    waiting = asyncio.create_task(run(store, scope, execute, poll=0.01))
    await asyncio.sleep(0.05)
    answer_now.set()
    await first
    result = await waiting
    assert result.outcome == outcome
    assert execute.calls == 0


async def test_the_window_starts_when_the_action_succeeds(
    store: IdempotencyStore, scope: str
) -> None:
    execute = Counter()
    window = timedelta(seconds=1)

    async def slow_approval() -> Approval:
        await asyncio.sleep(1.2)  # longer than the whole window
        return Approval("approved", "card-1")

    first = await run(store, scope, execute, confirm=slow_approval, window=window)
    second = await run(store, scope, execute, window=window)
    assert (first.outcome, second.outcome) == ("executed", "replayed")
    assert execute.calls == 1


class SlowParking:
    """A store whose park and un-park calls are slow, like a far database."""

    def __init__(self, inner: IdempotencyStore, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def mark_awaiting_approval(self, **kwargs: Any) -> bool:
        await asyncio.sleep(self._delay)
        return await self._inner.mark_awaiting_approval(**kwargs)


async def test_a_cancellation_while_parking_aborts_instead_of_needing_review(
    store: IdempotencyStore, scope: str
) -> None:
    slow: Any = SlowParking(store, delay=0.2)
    execute = Counter()
    task = asyncio.create_task(run(slow, scope, execute))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert (row.status, row.error) == (STATUS_ABORTED, ABORT_STOPPED)
    await store.reap_stale()  # nothing here for the reaper to call "may have fired"
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABORTED
    retry = await run(store, scope, execute)
    assert retry.outcome == "executed"
    assert execute.calls == 1


async def test_a_store_down_at_claim_fails_closed(scope: str) -> None:
    class Down:
        lease = timedelta(minutes=15)

        async def claim(self, **kwargs: Any) -> Any:
            raise ConnectionError("database unreachable")

    execute = Counter()
    down: Any = Down()
    result = await run(down, scope, execute)
    assert result.outcome == "blocked_store_unavailable"
    assert execute.calls == 0


async def test_a_success_the_store_cannot_record_still_reports_the_write(
    make_store: MakeStore, scope: str
) -> None:
    store = await make_store(lease=timedelta(milliseconds=100))

    class CantRecord:
        def __getattr__(self, name: str) -> Any:
            return getattr(store, name)

        async def finalize_success(self, **kwargs: Any) -> bool:
            raise ConnectionError("database went away after the send")

    execute = Counter("sent")
    flaky: Any = CantRecord()
    result = await run(flaky, scope, execute)
    assert result.outcome == "executed"
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_IN_PROGRESS  # left for the reaper
    await asyncio.sleep(0.25)
    await store.reap_stale()
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_ABANDONED_UNKNOWN  # needs review, never re-run


async def test_results_that_are_not_plain_json_are_stored_and_replayed(
    store: IdempotencyStore, scope: str
) -> None:
    sent_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    execute = Counter({"id": "m-1", "sent_at": sent_at, "score": float("nan")})
    first = await run(store, scope, execute)
    second = await run(store, scope, execute)
    assert first.outcome == "executed"
    assert first.result is not None
    assert first.result["sent_at"] is sent_at  # the caller gets what execute returned
    assert second.outcome == "replayed"
    assert second.result == {"id": "m-1", "sent_at": "2026-01-02 03:04:05+00:00", "score": "NaN"}
    row = await store.get(scope=scope, key="k1")
    assert row is not None
    assert row.status == STATUS_SUCCEEDED


def test_derive_key_ignores_argument_order() -> None:
    assert derive_key("s", "a", {"x": 1, "y": 2}) == derive_key("s", "a", {"y": 2, "x": 1})


def test_derive_key_uses_the_full_value() -> None:
    long = "x" * 10_000
    assert derive_key("s", "a", {"body": long + "1"}) != derive_key("s", "a", {"body": long + "2"})


def test_derive_key_is_scoped() -> None:
    base = derive_key("s", "a", {"x": 1})
    assert base != derive_key("other", "a", {"x": 1})
    assert base != derive_key("s", "b", {"x": 1})
    assert base != derive_key("s", "a", {"x": 1}, extra="session-2")


async def test_required_with_no_store_fails_closed() -> None:
    execute = Counter()
    result = await run_guarded(
        store=None,
        scope="s",
        key="k",
        action="send_email",
        execute=execute,
        confirm=answer("approved"),
        window=WINDOW,
    )
    assert result.outcome == "blocked_store_unavailable"
    assert execute.calls == 0


async def test_an_empty_scope_counts_as_no_store(store: IdempotencyStore) -> None:
    execute = Counter()
    result = await run_guarded(
        store=store,
        scope="",
        key="k",
        action="send_email",
        execute=execute,
        confirm=answer("approved"),
        window=WINDOW,
    )
    assert result.outcome == "blocked_store_unavailable"
    assert execute.calls == 0


async def test_not_required_still_asks_and_runs_without_dedup() -> None:
    execute = Counter()
    kwargs: dict[str, Any] = {
        "store": None,
        "scope": "s",
        "key": "k",
        "action": "send_email",
        "execute": execute,
        "window": WINDOW,
        "required": False,
    }
    declined = await run_guarded(confirm=answer("declined"), **kwargs)
    first = await run_guarded(confirm=answer("approved"), **kwargs)
    second = await run_guarded(confirm=answer("approved"), **kwargs)
    assert declined.outcome == "declined"
    assert (first.outcome, second.outcome) == ("executed", "executed")
    assert execute.calls == 2


async def test_run_guarded_with_a_store_is_guarded_action(
    store: IdempotencyStore, scope: str
) -> None:
    execute = Counter()
    for _ in range(2):
        await run_guarded(
            store=store,
            scope=scope,
            key="k",
            action="send_email",
            execute=execute,
            confirm=answer("approved"),
            window=WINDOW,
        )
    assert execute.calls == 1


@pytest.mark.parametrize(
    ("outcome", "words"),
    [
        ("declined", "was declined"),
        ("unanswered", "was not a refusal"),
        ("blocked_store_unavailable", "prevents duplicate changes"),
        ("blocked_needs_review", "won't retry it"),
        ("already_pending", "already in progress"),
    ],
)
def test_every_failure_says_nothing_was_done_by_this_call(outcome: Any, words: str) -> None:
    message = outcome_message(GuardResult(outcome), action="send the email", system="Gmail")
    assert words in message
    assert "send the email" in message


@pytest.mark.parametrize("outcome", ["executed", "replayed"])
def test_success_outcomes_have_no_failure_message(outcome: Any) -> None:
    with pytest.raises(ValueError, match="success outcome"):
        outcome_message(GuardResult(outcome), action="x")
