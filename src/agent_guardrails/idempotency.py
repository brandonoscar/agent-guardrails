"""At-most-once execution for irreversible actions.

``guarded_action`` owns one action's life: claim, approve, execute, record.
An approved side effect fires at most once across double-approval, retries,
concurrent duplicates and process restarts. When the guard cannot prove the
side effect did not already happen, it refuses to fire it again.

Failure classification, fail-safe by default. The guard has to know whether a
failure came before the external side effect or at/after it:

* ``execute`` raises :class:`PreSideEffectError`: the failure provably came
  first (validation, a bad argument). The row becomes ``failed_retryable`` and
  a later attempt may approve and run it again.
* ``execute`` raises anything else (a timeout, a dropped connection, a 5xx
  after sending, any unclassified error): the side effect may have fired. The
  row becomes ``abandoned_unknown`` and is never retried by itself.

The burden of proof is on "safe to retry": a tool opts in by raising
``PreSideEffectError``. Silence means do not fire again.

Anything that stops the guard before ``execute`` starts (a declined or
unanswered card, a gate that raised, a cancellation while parking the claim)
aborts the claim with its reason, so a later attempt starts fresh and a caller
waiting on this one is told what really happened.

The holder of a claim renews its lease in the background while it waits for
approval and while it executes, and every change it makes to the row names
its fencing token (see ``stores``). A holder that lost its claim to the
reaper, a shutdown release and someone else's reclaim can never overwrite the
new holder's row or execute on it.

The key: :func:`derive_key` hashes the full normalized action (sorted keys,
canonical JSON), never a truncated or masked form, so two actions that differ
only in a long tail get different keys. Only the digest is stored.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, TypeVar

from agent_guardrails.confirm import Approval
from agent_guardrails.stores import (
    ABORT_DECLINED,
    ABORT_NOT_SURFACED,
    ABORT_STOPPED,
    ABORT_UNANSWERED,
    STATUS_ABANDONED_UNKNOWN,
    STATUS_ABORTED,
    STATUS_AWAITING_APPROVAL,
    STATUS_FAILED_RETRYABLE,
    STATUS_IN_PROGRESS,
    STATUS_SUCCEEDED,
    IdempotencyStore,
)
from agent_guardrails.write_marker import executing_write

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PreSideEffectError(Exception):
    """Raise from ``execute`` for a failure that provably came before the
    side effect. Only this class makes an action retryable."""


GuardOutcome = Literal[
    "executed",  # ran exactly once, now
    "replayed",  # an earlier success in the window; not run again, its result returned
    "declined",  # the person said no; nothing fired
    "unanswered",  # nobody answered, or the card could not be shown; nothing fired
    "already_pending",  # another caller holds the claim; THIS call ran nothing
    "blocked_needs_review",  # may have fired before; refuse to fire again
    "blocked_store_unavailable",  # no store and one is required; nothing fired
]

#: The only outcomes after which a tool may say its action happened. An
#: allowlist on purpose: a tool that checks the outcomes it knows and falls
#: through to its success line for the rest reports ``already_pending`` (this
#: call ran nothing) as done.
SUCCESS_OUTCOMES: frozenset[str] = frozenset({"executed", "replayed"})

DETAIL_WAIT_EXHAUSTED = "pending wait exhausted"
DETAIL_RECLAIM_LOST = "reclaim lost to another caller"
DETAIL_PEER_FAILED = "the other caller failed before the side effect"
DETAIL_RELEASED_AT_GATE = "claim released while awaiting approval"
DETAIL_STORE_ERROR = "the idempotency store could not be reached"
#: Pending details where nothing is running, so asking again is safe.
_NOTHING_RUNNING = frozenset({DETAIL_PEER_FAILED, DETAIL_RELEASED_AT_GATE})

_PENDING_WAIT_S = 30.0
_PENDING_POLL_S = 0.25


@dataclass(frozen=True)
class GuardResult:
    outcome: GuardOutcome
    result: Any | None = None
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome in SUCCESS_OUTCOMES


_current_approval: ContextVar[Approval | None] = ContextVar(
    "agent_guardrails_current_approval", default=None
)


def current_approval() -> Approval | None:
    """The approval that authorized the ``execute`` now running, if any.

    Read it inside ``execute`` to apply the approver's edits
    (``approval.edits``) or to log which card authorized the write.
    """
    return _current_approval.get()


@contextlib.contextmanager
def _approval_bound(approval: Approval | None) -> Iterator[None]:
    token = _current_approval.set(approval)
    try:
        yield
    finally:
        _current_approval.reset(token)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def derive_key(scope: str, action: str, values: Any, *, extra: str = "") -> str:
    """sha256 over scope, action, ``extra`` and the full normalized values.

    Argument order never matters. ``extra`` folds in anything that tells two
    otherwise identical actions apart (a session, a target URL).
    """
    material = "\x1f".join((scope, action, extra, _canonical(values)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def outcome_message(guard: GuardResult, *, action: str, system: str = "the system") -> str:
    """Plain words for an outcome outside :data:`SUCCESS_OUTCOMES`.

    ``action`` is a verb phrase ("send the email"), ``system`` is where the
    user would look to check ("Gmail"). Every message says nothing was done
    by THIS call, because that is the one thing the user must not misread.
    """
    if guard.outcome == "declined":
        return f"I didn't {action}: the request was declined, so nothing was changed in {system}."
    if guard.outcome == "unanswered":
        return (
            f"I didn't {action}: the approval request was not answered, so nothing was "
            f"changed in {system}. This was not a refusal; ask again when someone can approve it."
        )
    if guard.outcome == "blocked_store_unavailable":
        return (
            f"I didn't {action}: the record that prevents duplicate changes is unavailable, "
            f"so nothing was changed in {system}. Try again shortly."
        )
    if guard.outcome == "blocked_needs_review":
        return (
            f"I can't confirm whether I managed to {action}: an earlier attempt may have "
            f"gone through. Check {system} before asking again; I won't retry it on my own."
        )
    if guard.outcome == "already_pending":
        if guard.detail in _NOTHING_RUNNING:
            return (
                f"I didn't {action}, and nothing was changed in {system}. The request "
                "stopped before anything went out; ask me again to retry."
            )
        return (
            f"I didn't {action}: the same request is already in progress, maybe waiting for "
            f"approval on an earlier card, so I didn't start a second one. Check {system} "
            "before asking again."
        )
    raise ValueError(f"{guard.outcome!r} is a success outcome; report the result instead")


async def _await_terminal(
    store: IdempotencyStore, *, scope: str, key: str, pending_wait: float, poll: float
) -> GuardResult:
    """Wait while another caller holds the claim. Never executes."""
    waited = 0.0
    while waited < pending_wait:
        await asyncio.sleep(poll)
        waited += poll
        row = await store.get(scope=scope, key=key)
        if row is None or row.status in (STATUS_IN_PROGRESS, STATUS_AWAITING_APPROVAL):
            continue
        if row.status == STATUS_SUCCEEDED:
            return GuardResult("replayed", row.result, "the other caller succeeded")
        if row.status == STATUS_ABORTED:
            # The reason decides the words: "nobody answered" must never be
            # reported as "the request was declined".
            if row.error == ABORT_DECLINED:
                return GuardResult("declined", detail="the other caller's card was declined")
            if row.error in (ABORT_UNANSWERED, ABORT_NOT_SURFACED):
                return GuardResult("unanswered", detail="nobody answered the other caller's card")
            return GuardResult("already_pending", detail=DETAIL_PEER_FAILED)
        if row.status == STATUS_ABANDONED_UNKNOWN:
            return GuardResult(
                "blocked_needs_review", detail="the other caller's result is unknown"
            )
        return GuardResult("already_pending", detail=DETAIL_PEER_FAILED)
    return GuardResult("already_pending", detail=DETAIL_WAIT_EXHAUSTED)


async def _run_verify(
    verify: Callable[[T], Awaitable[bool]], result: T, action: str, fail_tag: str
) -> tuple[bool, str]:
    """A read-back check. Returning False or raising both mean "not confirmed";
    an erroring verifier is never read as success. It must only read."""
    try:
        ok = await verify(result)
    except Exception as exc:
        tag = f"verify_error:{type(exc).__name__}"
        logger.warning("%s: read-back raised %s; needs review", action, tag)
        return False, tag
    if not ok:
        logger.warning("%s: read-back could not confirm the write; needs review", action)
        return False, fail_tag
    return True, ""


def _ambiguous_detail(exc: BaseException) -> str:
    """``ambiguous:<Class>`` plus what is needed to diagnose it later.

    Keep why, not just what: an abandoned action is never retried, so the
    stored error is the only record of what the other system said.
    """
    label = f"ambiguous:{type(exc).__name__}"
    parts: list[str] = []
    for attr in ("status", "status_code", "code", "path"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            parts.append(f"{attr}={value}")
    message = str(exc).strip()
    if message:
        parts.append(message)
    return f"{label} {' '.join(parts)}"[:500] if parts else label


def _jsonable(value: Any) -> Any:
    """Make a result storable for replay, the same way on every store.

    A replay returns this form: plain JSON, with anything else (a datetime, a
    Decimal) turned into its string. NaN and infinity are not JSON, so they
    are stored as the strings ``"NaN"`` and ``"Infinity"``.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="json")
    try:
        # parse_constant turns NaN and the infinities into strings on the way
        # back, so the stored form is valid JSON for every store.
        return json.loads(json.dumps(value, default=str), parse_constant=str)
    except (TypeError, ValueError):  # a circular structure, say
        return str(value)


def _not_approved(approval: Approval) -> GuardResult:
    if approval.status == "declined":
        return GuardResult("declined", detail="declined at the gate")
    if approval.status == "not_surfaced":
        return GuardResult("unanswered", detail="the approval card could not be shown")
    return GuardResult("unanswered", detail="nobody answered the approval card")


_ABORT_REASONS = {
    "declined": ABORT_DECLINED,
    "unanswered": ABORT_UNANSWERED,
    "not_surfaced": ABORT_NOT_SURFACED,
}


async def _quietly(write: Awaitable[bool], what: str, action: str) -> bool:
    """Run one store write after the fact. Shielded, so a second cancellation
    can't cut it off half done, and never raising: the outcome the caller is
    about to report matters more than the bookkeeping, and the reaper cleans
    up a row this left behind."""
    try:
        return await asyncio.shield(write)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("%s: could not %s", action, what, exc_info=True)
        return False


async def _renew_lease(
    store: IdempotencyStore, scope: str, key: str, token: str, every: float, action: str
) -> None:
    """Keep a held claim alive. Stops when the claim is no longer held."""
    while True:
        await asyncio.sleep(every)
        try:
            held = await store.renew(scope=scope, key=key, token=token)
        except Exception:
            logger.warning("%s: could not renew the claim's lease", action, exc_info=True)
            continue
        if not held:
            logger.warning("%s: the claim is no longer held; stopped renewing", action)
            return


def _renewal_period(store: IdempotencyStore) -> float | None:
    lease = getattr(store, "lease", None)
    if not isinstance(lease, timedelta) or lease.total_seconds() <= 0:
        return None
    return lease.total_seconds() / 3


async def guarded_action(
    *,
    store: IdempotencyStore,
    scope: str,
    key: str,
    action: str,
    execute: Callable[[], Awaitable[T]],
    confirm: Callable[[], Awaitable[Approval]] | None,
    window: timedelta | None,
    record: Callable[[GuardResult], Awaitable[None]] | None = None,
    verify: Callable[[T], Awaitable[bool]] | None = None,
    verify_fail_tag: str = "verify_failed:not_found",
    pending_wait: float = _PENDING_WAIT_S,
    poll: float = _PENDING_POLL_S,
) -> GuardResult:
    """Run an irreversible action at most once.

    ``confirm`` asks a person (for example
    ``lambda: channel.request(action=..., title=..., details=...)``), or is
    ``None`` when approval already happened out of band (a person clicked
    "send" in your UI before this ran). At-most-once comes from the claim,
    not from the approval, so it holds either way.

    ``window`` is how long a success is replayed instead of run again,
    counted from the success. ``verify`` is an optional read-only check run
    after ``execute`` and before success is recorded. ``record`` is called on
    a fresh success and on a replay, so an audit trail sees both.

    ``execute`` raising re-raises here after the row is recorded:
    ``PreSideEffectError`` as retryable, anything else as needs-review.
    """
    try:
        claim = await store.claim(scope=scope, key=key, action=action, window=window)
    except Exception:
        logger.warning("%s: idempotency store unavailable at claim", action, exc_info=True)
        return GuardResult("blocked_store_unavailable", detail=DETAIL_STORE_ERROR)

    token = claim.token
    if not claim.won:
        status = claim.row.status
        if status == STATUS_SUCCEEDED:
            res = GuardResult("replayed", claim.row.result, "an earlier success in the window")
            if record is not None:
                await record(res)
            return res
        if status in (STATUS_IN_PROGRESS, STATUS_AWAITING_APPROVAL):
            return await _await_terminal(
                store, scope=scope, key=key, pending_wait=pending_wait, poll=poll
            )
        if status == STATUS_ABANDONED_UNKNOWN:
            return GuardResult("blocked_needs_review", detail=claim.row.error or status)
        if status not in (STATUS_FAILED_RETRYABLE, STATUS_ABORTED):
            return GuardResult("blocked_needs_review", detail=f"unknown status {status!r}")
        # Nothing fired; safe to run again. One caller wins the flip back to
        # in_progress, and a loser waits instead of racing a second run.
        try:
            token = await store.reclaim(scope=scope, key=key, window=window)
        except Exception:
            logger.warning("%s: idempotency store unavailable at reclaim", action, exc_info=True)
            return GuardResult("blocked_store_unavailable", detail=DETAIL_STORE_ERROR)
        if token is None:
            return GuardResult("already_pending", detail=DETAIL_RECLAIM_LOST)
    if token is None:  # a store that won without a token can't fence its writes
        return GuardResult("blocked_needs_review", detail="the store returned no claim token")

    every = _renewal_period(store)
    renewer = (
        asyncio.create_task(_renew_lease(store, scope, key, token, every, action))
        if every is not None
        else None
    )
    try:
        return await _hold(
            store=store,
            scope=scope,
            key=key,
            token=token,
            action=action,
            execute=execute,
            confirm=confirm,
            record=record,
            verify=verify,
            verify_fail_tag=verify_fail_tag,
        )
    finally:
        if renewer is not None:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await renewer


async def _hold(
    *,
    store: IdempotencyStore,
    scope: str,
    key: str,
    token: str,
    action: str,
    execute: Callable[[], Awaitable[T]],
    confirm: Callable[[], Awaitable[Approval]] | None,
    record: Callable[[GuardResult], Awaitable[None]] | None,
    verify: Callable[[T], Awaitable[bool]] | None,
    verify_fail_tag: str,
) -> GuardResult:
    """Everything after the claim is won: approve, execute, record."""

    def abort(reason: str) -> Awaitable[bool]:
        return _quietly(
            store.finalize_aborted(scope=scope, key=key, token=token, reason=reason),
            "abort the claim",
            action,
        )

    approval: Approval | None = None
    try:
        if confirm is not None:
            # Park the claim while a person decides, so a crash during the
            # wait is provably before the side effect and reaps to retryable,
            # not to a scary needs-review row. Parking is an improvement,
            # never load-bearing.
            try:
                parked = await store.mark_awaiting_approval(scope=scope, key=key, token=token)
            except Exception:
                logger.warning("%s: could not park the claim at the gate", action, exc_info=True)
                parked = False
            approval = await confirm()
            if not approval.approved:
                await abort(_ABORT_REASONS.get(approval.status, ABORT_UNANSWERED))
                return _not_approved(approval)
            # Un-park before executing, so a crash from here on is ambiguous
            # again. Losing this (the reaper or a shutdown release got there
            # first) means the claim is no longer ours: never execute on it.
            if parked:
                try:
                    resumed = await store.resume_from_awaiting(scope=scope, key=key, token=token)
                except Exception:
                    logger.warning("%s: could not un-park the claim", action, exc_info=True)
                    resumed = False
                if not resumed:
                    await abort(ABORT_STOPPED)
                    return GuardResult("already_pending", detail=DETAIL_RELEASED_AT_GATE)
    except BaseException:
        # The gate failed, or the task was cancelled while parking, waiting or
        # un-parking. Nothing fired, so the row must not read as in progress
        # (the reaper would call that "may have fired").
        await abort(ABORT_STOPPED)
        raise

    try:
        # The marker spans the await: the write happens while we are suspended.
        with executing_write(action), _approval_bound(approval):
            result = await execute()
    except PreSideEffectError as exc:
        await _quietly(
            store.finalize_retryable(
                scope=scope, key=key, token=token, error=f"pre_side_effect:{type(exc).__name__}"
            ),
            "record a retryable failure",
            action,
        )
        raise
    except BaseException as exc:
        await _quietly(
            store.finalize_abandoned(
                scope=scope, key=key, token=token, error=_ambiguous_detail(exc)
            ),
            "record an ambiguous failure",
            action,
        )
        logger.warning("%s: ambiguous failure %s; not retried", action, type(exc).__name__)
        raise

    if verify is not None:
        verified, tag = await _run_verify(verify, result, action, verify_fail_tag)
        if not verified:
            await _quietly(
                store.finalize_abandoned(scope=scope, key=key, token=token, error=tag),
                "record a failed read-back",
                action,
            )
            return GuardResult("blocked_needs_review", detail=tag)

    # The side effect happened. A store that can't record it now must not turn
    # that into an error: the row stays held, and the reaper marks it for
    # review, so it is never fired again.
    stored = await _quietly(
        store.finalize_success(scope=scope, key=key, token=token, result=_jsonable(result)),
        "record the success",
        action,
    )
    if not stored:
        logger.warning("%s: ran, but the success was not recorded under this claim", action)
    res = GuardResult("executed", result, "ran now")
    if record is not None:
        await record(res)
    return res


async def run_guarded(
    *,
    store: IdempotencyStore | None,
    scope: str,
    key: str,
    action: str,
    execute: Callable[[], Awaitable[T]],
    confirm: Callable[[], Awaitable[Approval]] | None,
    window: timedelta | None,
    required: bool = True,
    record: Callable[[GuardResult], Awaitable[None]] | None = None,
    verify: Callable[[T], Awaitable[bool]] | None = None,
    verify_fail_tag: str = "verify_failed:not_found",
) -> GuardResult:
    """:func:`guarded_action` with a policy for when there is no store.

    With a store and a non-empty ``scope`` it is exactly ``guarded_action``.
    Without one:

    * ``required=True`` (the default): fail closed. Nothing runs, and the
      outcome is ``blocked_store_unavailable`` so the tool can say so.
    * ``required=False`` (local development): still ask for approval and run
      on a yes, but without dedup, and log a warning every time.

    An empty ``scope`` counts as no store: dedup without a scope would let one
    customer's identical action replay another customer's result.
    """
    if store is not None and scope:
        return await guarded_action(
            store=store,
            scope=scope,
            key=key,
            action=action,
            execute=execute,
            confirm=confirm,
            window=window,
            record=record,
            verify=verify,
            verify_fail_tag=verify_fail_tag,
        )

    reason = "no scope" if not scope else "no idempotency store"
    if required:
        logger.warning("%s: %s and dedup is required; not executing", action, reason)
        return GuardResult("blocked_store_unavailable", detail=reason)

    logger.warning("%s: %s; running WITHOUT dedup (required=False)", action, reason)
    approval: Approval | None = None
    if confirm is not None:
        approval = await confirm()
        if not approval.approved:
            return _not_approved(approval)
    with executing_write(action), _approval_bound(approval):
        result = await execute()
    if verify is not None:
        verified, tag = await _run_verify(verify, result, action, verify_fail_tag)
        if not verified:
            return GuardResult("blocked_needs_review", detail=tag)
    res = GuardResult("executed", result, "ran without dedup")
    if record is not None:
        await record(res)
    return res


__all__ = [
    "SUCCESS_OUTCOMES",
    "GuardOutcome",
    "GuardResult",
    "PreSideEffectError",
    "current_approval",
    "derive_key",
    "guarded_action",
    "outcome_message",
    "run_guarded",
]
