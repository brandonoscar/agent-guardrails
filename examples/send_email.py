"""An agent that sends email, with the write gated and deduplicated.

Runs offline: a scripted model stands in for the LLM, so no API key is needed.
The script makes the agent look up a contact, send an email, then send the
exact same email again (a retry, as models do).

    python examples/send_email.py          # asks you to approve in the terminal
    python examples/send_email.py --yes    # approve without asking
    python examples/send_email.py --no     # decline without asking

What you should see: one approval card for the first send, nothing for the
lookup (a read), and no second card or second email for the retry.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from agents import Agent, Runner, function_tool, set_tracing_disabled

from agent_guardrails import ConfirmCard, ConfirmChannel, InMemoryIdempotencyStore, RunGuard
from agent_guardrails.confirm import Emitter
from agent_guardrails.openai_agents import gate_tools
from agent_guardrails.testing import ScriptedModel, call, say

OUTBOX: list[dict[str, str]] = []


@function_tool
def search_contacts(name: str) -> str:
    """Find a contact's email address."""
    return json.dumps({"name": name, "email": "pat@example.com"})


@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. (Fake: it only appends to OUTBOX.)"""
    OUTBOX.append({"to": to, "subject": subject, "body": body})
    return f"Email sent to {to}."


def terminal_approver(channel: ConfirmChannel, auto: bool | None) -> Emitter:
    async def show(card: ConfirmCard) -> None:
        print(f"\n--- approval needed: {card.title}\n{card.details}\n---")
        if auto is None:
            reply = await asyncio.to_thread(input, "Approve? [y/N] ")
            approved = reply.strip().lower() in {"y", "yes"}
        else:
            approved = auto
            print("approved" if approved else "declined")
        channel.resolve(card.confirm_id, approved)

    return show


async def main(auto: bool | None) -> None:
    set_tracing_disabled(True)
    channel = ConfirmChannel(timeout_s=120)
    tools = gate_tools(
        [search_contacts, send_email],
        channel=channel,
        store=InMemoryIdempotencyStore(),
        system="your mailbox",
    )
    model = ScriptedModel(
        [
            [call("search_contacts", name="Pat")],
            [call("send_email", to="pat@example.com", subject="Invoice", body="Attached.")],
            [call("send_email", to="pat@example.com", subject="Invoice", body="Attached.")],
            [say("Done.")],
        ]
    )
    agent = Agent(name="assistant", model=model, tools=tools)

    emitter = terminal_approver(channel, auto)
    with RunGuard(scope="demo-user", channel=channel, emitter=emitter) as run:
        result = await Runner.run(agent, "Email Pat the invoice.")

    print("\nTool results the model saw:")
    for item in result.new_items:
        output = getattr(item, "output", None)
        if output is not None:
            print(f"  - {output}")
    print(f"\nEmails actually sent: {len(OUTBOX)}")
    print(f"Approval cards shown: {run.summary()['approvals']['raised']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--yes", action="store_true", help="approve without asking")
    choice.add_argument("--no", action="store_true", help="decline without asking")
    args = parser.parse_args()
    asyncio.run(main(True if args.yes else False if args.no else None))
