"""A runaway agent, stopped by structure instead of by prompt wording.

Runs offline with scripted models. The orchestrator does what real models do
on a hard question: it looks records up one at a time, repeats lookups, and
keeps consulting a nested "researcher" agent that is itself an agent-as-tool.

    python examples/runaway.py

What you should see:

* the repeated lookup is answered from the run-wide cache;
* the per-record lookups stop at their cap, and the refusal names the bulk tool;
* the third consult is refused (nested cap of 2);
* the run is stopped by its cost ceiling, nested spend included, and the
  caller gets a ``SoftStop`` to write an answer from what was gathered.
"""

from __future__ import annotations

import asyncio
import json

from agents import Agent, Runner, Tool, function_tool, set_tracing_disabled

from agent_guardrails import (
    BudgetPolicy,
    ModelPrice,
    PriceTable,
    RunGuard,
    SoftStop,
    ToolCap,
    budget_tools,
    cap_nested_tool,
)
from agent_guardrails.openai_agents import UsageHooks, watch_run
from agent_guardrails.testing import ScriptedModel, Turn, call, say

# Made-up rates for the scripted models. Supply your provider's real rates.
PRICES = PriceTable(
    {
        "orchestrator": ModelPrice(input=10.0, output=40.0),
        "researcher": ModelPrice(input=10.0, output=40.0),
    }
)


@function_tool
def get_invoice(invoice_id: int) -> str:
    """Look up one invoice."""
    return json.dumps({"id": invoice_id, "status": "unpaid", "amount": 100 + invoice_id})


@function_tool
def list_unpaid_invoices() -> str:
    """Every unpaid invoice in one call."""
    return json.dumps([{"id": i, "amount": 100 + i} for i in range(1, 30)])


async def main() -> None:
    set_tracing_disabled(True)
    hooks = UsageHooks(PRICES)
    researcher = Agent(
        name="researcher",
        model=ScriptedModel(
            [[say("The overdue ones cluster at month end.")] for _ in range(20)],
            name="researcher",
            input_tokens=3_000,
            output_tokens=300,
        ),
    )
    tools: list[Tool] = [
        *budget_tools([get_invoice, list_unpaid_invoices]),
        cap_nested_tool(
            researcher.as_tool(
                tool_name="ask_researcher",
                tool_description="Ask the research agent.",
                hooks=hooks,
            )
        ),
    ]
    script: list[Turn] = [
        [call("get_invoice", invoice_id=1), call("get_invoice", invoice_id=1)],
        [call("get_invoice", invoice_id=i) for i in range(2, 9)],
        *[[call("ask_researcher", input=f"What about invoice {i}?")] for i in range(1, 40)],
    ]
    orchestrator = Agent(
        name="orchestrator",
        model=ScriptedModel(script, name="orchestrator", input_tokens=3_000, output_tokens=300),
        tools=tools,
    )
    policy = BudgetPolicy(
        caps={"get_invoice": ToolCap(5, "Use `list_unpaid_invoices` for all of them at once.")}
    )

    with RunGuard(scope="demo", budget=policy, nested_cap=2, cost_ceiling_usd=0.50) as run:
        result = Runner.run_streamed(orchestrator, "Which invoices are overdue?", hooks=hooks)
        try:
            async for event in watch_run(result, run, poll=0.05):
                if getattr(event, "name", None) == "tool_output":
                    output = str(getattr(getattr(event, "item", None), "output", ""))
                    print(f"tool -> {' '.join(output.split())[:110]}")
        except SoftStop as stop:
            print(f"\nSTOPPED ({stop.reason}): {stop}")

    summary = run.summary()
    print("\nWhat the guards did:")
    print(f"  memo hits:        {summary['calls']['memo_hits']}")
    print(f"  refused lookups:  {summary['calls']['refusals']}")
    print(f"  consults allowed: {summary['nested']['consults']} (cap {summary['nested']['cap']})")
    print(f"  model calls:      {summary['usage']['calls']} {summary['usage']['by_model']}")
    print(f"  spend:            ${summary['usage']['cost_usd']:.2f} (ceiling $0.50)")


if __name__ == "__main__":
    asyncio.run(main())
