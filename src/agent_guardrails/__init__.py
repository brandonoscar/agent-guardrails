"""Guardrails for tool-using agents.

* Fail-closed write approval: a tool that is not provably a read asks a
  person first and runs only on an explicit yes (``classify``, ``confirm``).
* At-most-once writes: an approved action fires once across retries,
  duplicates and restarts, and is never re-fired when the outcome is unknown
  (``idempotency``, ``stores``, ``postgres``).
* Run-wide call budgets that survive nested agent calls: memo, caps, a refusal
  ceiling and a soft deadline (``budget``), plus a cap on nested consults
  (``nested``).
* A per-run cost ceiling (``cost``) and a stream watchdog that enforces it
  together with stalls and deadlines (``watchdog``).

The core has no dependencies. ``agent_guardrails.openai_agents`` and
``agent_guardrails.testing`` need the ``openai-agents`` extra;
``agent_guardrails.postgres`` needs the ``postgres`` extra.
"""

from agent_guardrails.budget import (
    BudgetPolicy,
    CallLedger,
    ToolCap,
    bind_call_ledger,
    budget_tool,
    budget_tools,
    current_call_ledger,
    reset_call_ledger,
)
from agent_guardrails.classify import (
    DEFAULT_CLASSIFIER,
    ToolClassifier,
    is_write_tool,
)
from agent_guardrails.confirm import (
    Approval,
    ConfirmBudget,
    ConfirmCard,
    ConfirmChannel,
    render_details,
)
from agent_guardrails.cost import (
    ModelPrice,
    PriceTable,
    RunUsage,
    bind_run_usage,
    current_run_usage,
    record_usage,
    reset_run_usage,
)
from agent_guardrails.idempotency import (
    SUCCESS_OUTCOMES,
    GuardResult,
    PreSideEffectError,
    current_approval,
    derive_key,
    guarded_action,
    outcome_message,
    run_guarded,
)
from agent_guardrails.nested import (
    NestedLedger,
    bind_nested_ledger,
    cap_nested_tool,
    cap_nested_tools,
    reset_nested_ledger,
)
from agent_guardrails.run import RunGuard, current_scope
from agent_guardrails.stores import IdempotencyStore, InMemoryIdempotencyStore
from agent_guardrails.watchdog import GuardStop, HardStop, SoftStop, watch_stream
from agent_guardrails.write_marker import executing_write, write_in_flight

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CLASSIFIER",
    "SUCCESS_OUTCOMES",
    "Approval",
    "BudgetPolicy",
    "CallLedger",
    "ConfirmBudget",
    "ConfirmCard",
    "ConfirmChannel",
    "GuardResult",
    "GuardStop",
    "HardStop",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "ModelPrice",
    "NestedLedger",
    "PreSideEffectError",
    "PriceTable",
    "RunGuard",
    "RunUsage",
    "SoftStop",
    "ToolCap",
    "ToolClassifier",
    "__version__",
    "bind_call_ledger",
    "bind_nested_ledger",
    "bind_run_usage",
    "budget_tool",
    "budget_tools",
    "cap_nested_tool",
    "cap_nested_tools",
    "current_approval",
    "current_call_ledger",
    "current_run_usage",
    "current_scope",
    "derive_key",
    "executing_write",
    "guarded_action",
    "is_write_tool",
    "outcome_message",
    "record_usage",
    "render_details",
    "reset_call_ledger",
    "reset_nested_ledger",
    "reset_run_usage",
    "run_guarded",
    "watch_stream",
    "write_in_flight",
]
