"""The approval gate: a tool that writes asks a person first, and runs only on
an explicit yes.

``ConfirmChannel.request()`` is the enforcement primitive. A side-effecting
tool awaits it inside its own body (or is wrapped so that it does, see
``openai_agents.gate_tool``), so the model cannot reach the side effect
without passing through the gate, whatever it remembered to call first. A
"please confirm before writing" line in a prompt is a convention the model can
skip; an ``await`` inside the tool is not.

Fail-closed in every direction:

* only an explicit approval returns ``approved``;
* a timeout is not a yes;
* a card that could not be shown to anyone is not a yes;
* edits ride only on an approval, and only for fields the gate declared
  editable.

Unanswered is not declined. A gate that timed out reports ``unanswered``, so
the agent can tell the user the truth ("nobody answered, nothing was done")
instead of reporting a decision nobody made ("you said no").

An unanswered gate is also expensive: the run sits silent for the whole
timeout, and a model that retries the write buys another full timeout each
time. ``ConfirmBudget`` bounds that per run: once one gate in the run has
expired unanswered, later gates return ``unanswered`` at once. It does not cap
concurrent gates. The SDK runs a turn's tool calls in parallel, and several
gates raised in the same second are distinct actions a present user would
answer one by one, so the budget only reacts to an expiry that has already
happened.

Per-run state lives in mutable objects reached through ``ContextVar``s. Nested
agent-as-tool runs execute in tasks that copy the context: a ``ContextVar.set``
inside one is invisible to the parent, but a shared object mutated inside one
is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar, Token
from typing import Any, Literal

logger = logging.getLogger(__name__)

ApprovalStatus = Literal["approved", "declined", "unanswered", "not_surfaced"]
ResolveStatus = Literal["resolved", "unknown"]

#: Default seconds a gate waits for a person.
DEFAULT_TIMEOUT_S = 300.0

#: Gates that may run the full timeout unanswered before the run stops
#: waiting. One: a five-minute silence is not ambiguous, and a second one
#: tells you nothing the first did not.
RUN_EXPIRED_GATE_CEILING = 1

#: Cap on one edited field's value. An approver pasting a novel into the gate
#: is a mistake, not an edit.
MAX_EDIT_VALUE_LEN = 8_000


@dataclasses.dataclass(frozen=True)
class ConfirmCard:
    """What the person is asked to approve. Serialize it however your UI needs."""

    confirm_id: str
    action: str
    title: str
    details: str
    items: tuple[str, ...] = ()
    #: The actual outbound payload, so the approver sees what will really be
    #: sent rather than a summary of it.
    request_body: Mapping[str, Any] | None = None
    #: Fields of ``request_body`` the approver may rewrite before approving.
    editable_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "confirm_id": self.confirm_id,
            "action": self.action,
            "title": self.title,
            "details": self.details,
            "items": list(self.items),
        }
        if self.request_body is not None:
            out["request_body"] = dict(self.request_body)
        if self.editable_fields:
            out["editable_fields"] = list(self.editable_fields)
        return out


@dataclasses.dataclass(frozen=True)
class Approval:
    """The answer to one gate."""

    status: ApprovalStatus
    confirm_id: str
    #: The approver's edits, filtered to the declared editable fields. Only
    #: ever set on an approval.
    edits: Mapping[str, str] | None = None

    @property
    def approved(self) -> bool:
        return self.status == "approved"


@dataclasses.dataclass
class ConfirmBudget:
    """What one run's approval gates cost, and whether to keep waiting."""

    expired_ceiling: int = RUN_EXPIRED_GATE_CEILING
    #: Gates raised, whatever their outcome.
    raised: int = 0
    #: Gates a person answered, yes or no.
    answered: int = 0
    #: Gates that ran the full timeout with no answer.
    expired: int = 0
    #: Gates not waited on because an earlier one had already expired.
    skipped: int = 0
    #: Gates waiting for a person right now. A stream watchdog pauses its
    #: clocks while this is above zero: a person reading a card is a
    #: legitimate silence.
    waiting: int = 0
    #: Seconds spent waiting on gates nobody answered.
    wasted_s: float = 0.0
    #: Seconds during which at least one gate was waiting, finished intervals
    #: only (see :meth:`paused_s`).
    paused_total_s: float = 0.0
    _pause_started: float | None = dataclasses.field(default=None, init=False, repr=False)

    def begin_wait(self) -> None:
        """A gate starts waiting for a person."""
        if self.waiting == 0:
            self._pause_started = time.monotonic()
        self.waiting += 1

    def end_wait(self) -> None:
        """A gate stopped waiting (answered, expired or cancelled)."""
        self.waiting = max(0, self.waiting - 1)
        if self.waiting == 0 and self._pause_started is not None:
            self.paused_total_s += time.monotonic() - self._pause_started
            self._pause_started = None

    def paused_s(self) -> float:
        """Seconds this run has spent with a person reading a card, so far.

        The run's deadlines are measured in active time, which excludes this:
        a person reading a card is not the run being slow.
        """
        open_interval = (
            time.monotonic() - self._pause_started if self._pause_started is not None else 0.0
        )
        return self.paused_total_s + open_interval

    def approval_is_unreachable(self) -> bool:
        """Has this run already shown that nobody is answering?

        Reads ``expired``, not ``raised``: a gate in flight proves nothing,
        and counting it would make a batch of concurrent gates cancel each
        other.
        """
        return self.expired >= self.expired_ceiling

    def summary(self) -> dict[str, float | int]:
        return {
            "raised": self.raised,
            "answered": self.answered,
            "expired": self.expired,
            "skipped": self.skipped,
            "wasted_s": round(self.wasted_s, 1),
        }


_confirm_budget: ContextVar[ConfirmBudget | None] = ContextVar(
    "agent_guardrails_confirm_budget", default=None
)


def bind_confirm_budget(
    budget: ConfirmBudget | None = None,
) -> tuple[ConfirmBudget, Token[ConfirmBudget | None]]:
    """Bind a fresh budget for one run. Reset with :func:`reset_confirm_budget`."""
    budget = budget if budget is not None else ConfirmBudget()
    return budget, _confirm_budget.set(budget)


def reset_confirm_budget(token: Token[ConfirmBudget | None]) -> None:
    _confirm_budget.reset(token)


def current_confirm_budget() -> ConfirmBudget | None:
    """The run's budget, or ``None`` outside a bound run.

    ``None`` means unbounded: a gate raised from a script or a test waits the
    full timeout every time.
    """
    return _confirm_budget.get()


def confirm_waiting() -> bool:
    """True while a gate in the current run is waiting for a person."""
    budget = _confirm_budget.get()
    return budget is not None and budget.waiting > 0


Emitter = Callable[[ConfirmCard], Awaitable[None]]


class ConfirmChannel:
    """Shows approval cards to a person and waits for the answer.

    One channel per process is the usual shape: it holds the pending answers,
    and your API layer calls :meth:`resolve` when the person clicks. Bind an
    emitter per run (:meth:`bind_emitter`) so each session's cards reach that
    session's screen; with no emitter bound, cards wait in a bounded queue
    that a caller drains with :meth:`drain`.
    """

    def __init__(
        self,
        *,
        emitter: Emitter | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_queued: int = 100,
    ) -> None:
        self.timeout_s = timeout_s
        self._default_emitter = emitter
        self._emitter: ContextVar[Emitter | None] = ContextVar(
            f"agent_guardrails_emitter_{id(self)}", default=None
        )
        self._pending: dict[str, asyncio.Future[tuple[bool, dict[str, str] | None]]] = {}
        self._queue: asyncio.Queue[ConfirmCard] = asyncio.Queue(maxsize=max_queued)

    # ── wiring ────────────────────────────────────────────────────────────

    def bind_emitter(self, emitter: Emitter) -> Token[Emitter | None]:
        """Route this run's cards to ``emitter``. Returns a reset token."""
        return self._emitter.set(emitter)

    def reset_emitter(self, token: Token[Emitter | None]) -> None:
        self._emitter.reset(token)

    @contextlib.contextmanager
    def emitting_to(self, emitter: Emitter) -> Iterator[None]:
        """``with channel.emitting_to(send_card): ...`` for one run."""
        token = self.bind_emitter(emitter)
        try:
            yield
        finally:
            self.reset_emitter(token)

    def drain(self) -> list[ConfirmCard]:
        """Cards waiting to be shown (the no-emitter path). Never blocks."""
        cards: list[ConfirmCard] = []
        while True:
            try:
                cards.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return cards

    def pending_ids(self) -> list[str]:
        """Gates currently waiting for an answer."""
        return list(self._pending)

    def resolve(
        self, confirm_id: str, approved: bool, *, edits: Mapping[str, str] | None = None
    ) -> ResolveStatus:
        """Deliver a person's answer. ``"unknown"`` when nothing is waiting on it
        (already answered, timed out, cancelled, or never raised here).

        ``approved`` must be a real ``bool``. Anything else (the string
        ``"false"``, which is truthy) raises ``TypeError`` and leaves the card
        waiting: a malformed answer is a bug in the caller, never a yes.
        """
        if not isinstance(approved, bool):
            raise TypeError(f"approved must be True or False, not {type(approved).__name__}")
        future = self._pending.pop(confirm_id, None)
        if future is None or future.done():
            return "unknown"
        # Edits ride only on an approval. A decline's edits are meaningless and
        # must never reach an executor.
        future.set_result((approved, dict(edits) if approved and edits else None))
        return "resolved"

    # ── the gate ──────────────────────────────────────────────────────────

    async def request(
        self,
        *,
        action: str,
        title: str,
        details: str,
        items: Sequence[str] = (),
        request_body: Mapping[str, Any] | None = None,
        editable_fields: Sequence[str] = (),
        timeout_s: float | None = None,
    ) -> Approval:
        """Show a card and wait for the answer. Only an explicit yes approves."""
        confirm_id = uuid.uuid4().hex
        budget = _confirm_budget.get()
        if budget is not None:
            budget.raised += 1
            if budget.approval_is_unreachable():
                # Nobody answered the last one. Don't hold the run open for an
                # answer this run has already shown is not coming; asking again
                # starts a fresh run with a fresh budget.
                budget.skipped += 1
                logger.info(
                    "gate %s not raised: %d gate(s) already expired unanswered this run",
                    action,
                    budget.expired,
                )
                return Approval("unanswered", confirm_id)

        card = ConfirmCard(
            confirm_id=confirm_id,
            action=action,
            title=title,
            details=details,
            items=tuple(items),
            request_body=dict(request_body) if request_body is not None else None,
            editable_fields=tuple(editable_fields),
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[bool, dict[str, str] | None]] = loop.create_future()
        timeout = self.timeout_s if timeout_s is None else timeout_s
        started = time.monotonic()
        waiting = False
        # Registered BEFORE the card goes out, so an answer that comes back
        # while the emitter is still returning has somewhere to land. Removed
        # in the finally whatever happens next, a cancellation mid-emit
        # included, so a late click on a dead card reads "unknown".
        self._pending[confirm_id] = future
        try:
            if not await self._surface(card):
                return Approval("not_surfaced", confirm_id)
            if budget is not None:
                budget.begin_wait()
                waiting = True
            try:
                approved, edits = await asyncio.wait_for(future, timeout=timeout)
            except TimeoutError:
                if budget is not None:
                    budget.expired += 1
                    budget.wasted_s += max(0.0, time.monotonic() - started)
                return Approval("unanswered", confirm_id)
        finally:
            self._pending.pop(confirm_id, None)
            if not future.done():
                future.cancel()
            if budget is not None and waiting:
                budget.end_wait()

        if budget is not None:
            budget.answered += 1
        if not approved:
            return Approval("declined", confirm_id)
        return Approval("approved", confirm_id, _allowed_edits(edits, card.editable_fields))

    async def confirm_or_abort(
        self,
        *,
        action: str,
        title: str,
        details: str,
        items: Sequence[str] = (),
        request_body: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """``True`` only on an explicit approval. For tools that write inline:

        ``if not await channel.confirm_or_abort(...): return "Not sent."``
        """
        approval = await self.request(
            action=action,
            title=title,
            details=details,
            items=items,
            request_body=request_body,
            timeout_s=timeout_s,
        )
        return approval.approved

    async def _surface(self, card: ConfirmCard) -> bool:
        emitter = self._emitter.get() or self._default_emitter
        if emitter is not None:
            try:
                await emitter(card)
                return True
            except Exception:
                # An emit failure must not skip the gate: fall back to the queue.
                logger.warning("confirm emitter failed; queueing the card", exc_info=True)
        try:
            self._queue.put_nowait(card)
        except asyncio.QueueFull:
            # The card cannot reach anyone, so the write must not proceed.
            logger.warning("confirm queue full; refusing %s without approval", card.action)
            return False
        return True


_MAX_VALUE_LEN = 120
_BODY_PREVIEW_LEN = 400
_BODY_KEYS = ("body", "message", "text", "content", "html_body", "message_body")


def _truncate(value: Any, limit: int = _MAX_VALUE_LEN) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}… (+{len(text) - limit} chars)"


def render_details(action: str, args: Mapping[str, Any]) -> str:
    """A readable preview of a pending write, for the card a person approves.

    Three rules from approvals that went wrong:

    * Show the message body. A card that names the recipient but not the text
      lets a prompt-injected draft be approved without anyone reading it.
    * Never dump a table. A ``values`` matrix makes a card nobody reads; say
      how big it is instead.
    * Truncate everything else, so one base64 blob cannot bury the rest.
    """
    if not args:
        return f"{action} (no arguments)"
    lines: list[str] = []
    body_key = next(
        (k for k in _BODY_KEYS if isinstance(args.get(k), str) and str(args[k]).strip()), None
    )
    for key, value in args.items():
        if key == body_key:
            continue
        if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
            cols = max(len(row) for row in value)
            lines.append(f"{key}: {len(value)} row(s) x {cols} column(s)")
        else:
            lines.append(f"{key}: {_truncate(value)}")
    text = f"{action}\n" + "\n".join(lines)
    if body_key is not None:
        text += f"\n\n{body_key}:\n{_truncate(str(args[body_key]).strip(), _BODY_PREVIEW_LEN)}"
    return text


def _allowed_edits(
    edits: Mapping[str, str] | None, editable_fields: Sequence[str]
) -> dict[str, str] | None:
    """Only declared fields survive, as length-capped, non-blank strings."""
    if not edits or not editable_fields:
        return None
    allowed = set(editable_fields)
    kept = {
        key: str(value)[:MAX_EDIT_VALUE_LEN]
        for key, value in edits.items()
        if key in allowed and str(value).strip()
    }
    return kept or None


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_EDIT_VALUE_LEN",
    "RUN_EXPIRED_GATE_CEILING",
    "Approval",
    "ApprovalStatus",
    "ConfirmBudget",
    "ConfirmCard",
    "ConfirmChannel",
    "Emitter",
    "bind_confirm_budget",
    "confirm_waiting",
    "current_confirm_budget",
    "render_details",
    "reset_confirm_budget",
]
