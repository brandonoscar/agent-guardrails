# agent-guardrails

[![CI](https://github.com/brandonoscar/agent-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/brandonoscar/agent-guardrails/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Guardrails for tool-using agents on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python):

- **Fail-closed write approval.** Any tool that is not provably a read asks a person before it runs. Unknown tools count as writes.
- **At-most-once writes.** An approved action fires once across retries, duplicate calls and restarts, and is never fired again when the outcome is unknown.
- **Run budgets the model can't talk past.** Repeated reads are cached, per-record lookups are capped, nested agent consults are capped, and each run has a cost ceiling. The limits hold inside nested agent-as-tool runs too.

The core is plain Python with no dependencies. The SDK adapter and the Postgres store are optional extras.

## The problem

Two things go wrong once an agent can call tools.

**Writes nobody approved.** "Confirm with the user before sending" is a line in a prompt, and models skip it. They also retry: a send that timed out gets sent again, and a person who approved one email ends up approving two, or none, because the second never asked. When a send fails halfway, nobody can say whether it went out.

**Runs that never finish.** On a hard question the model looks records up one at a time, repeats the same lookup, and calls a nested specialist agent that does the same thing with its own turns and tokens. Rewording the tool descriptions moves the loop to a different tool. The run hits its deadline with nothing to show and still bills every call.

Both are fixed in code. A gate the tool awaits can't be skipped. A cap that refuses the call doesn't depend on the model reading a hint.

## Try it in 30 seconds

No API key needed: the examples use a scripted model, so the real SDK runner executes real tools offline.

```bash
git clone https://github.com/brandonoscar/agent-guardrails
cd agent-guardrails
pip install -e ".[openai-agents]"
python examples/send_email.py --yes
python examples/runaway.py
```

`send_email.py` looks up a contact, sends an email, then sends the same email again the way a retrying model does:

```text
--- approval needed: Allow send_email?
send_email
to: pat@example.com
subject: Invoice

body:
Attached.
---
approved

Emails actually sent: 1
Approval cards shown: 1
```

The lookup never asks (it's a read), and the retry gets the first result back without a second card or a second email. Drop `--yes` to answer the card yourself, or pass `--no`.

`runaway.py` runs an orchestrator that hydrates invoices one at a time and keeps consulting a nested agent:

```text
STOPPED (cost_ceiling): cost ceiling reached ($0.50 spent): stopped to write an answer from what was already gathered

What the guards did:
  memo hits:        1
  refused lookups:  3
  consults allowed: 2 (cap 2)
  model calls:      12 {'orchestrator': 10, 'researcher': 2}
  spend:            $0.50 (ceiling $0.50)
```

## Install

Not on PyPI. Install it from GitHub (add `@<commit>` to the URL to pin one):

```bash
pip install "agent-guardrails @ git+https://github.com/brandonoscar/agent-guardrails"
```

Extras go after the name, as in `agent-guardrails[openai-agents,postgres] @ git+...`: `openai-agents` for the SDK adapter, `postgres` for the Postgres store. The core needs neither. A package named `agent-guardrails` on PyPI is a different project, so `pip install agent-guardrails` won't get you this one.

## Usage

```python
from agents import Agent, Runner
from agent_guardrails import ConfirmChannel, InMemoryIdempotencyStore, RunGuard, SoftStop
from agent_guardrails.openai_agents import UsageHooks, gate_tools, watch_run

channel = ConfirmChannel()  # one per process; your API answers its cards
store = InMemoryIdempotencyStore()  # PostgresIdempotencyStore in production
tools = gate_tools([search_contacts, send_email], channel=channel, store=store)
agent = Agent(name="assistant", model=MODEL, tools=tools)
hooks = UsageHooks(prices)  # your PriceTable, with a price for MODEL


async def answer(question: str, user_id: str, show_card) -> str:
    guard = RunGuard(scope=user_id, deadline_s=240, cost_ceiling_usd=1.00)
    with guard, channel.emitting_to(show_card):
        result = Runner.run_streamed(agent, question, hooks=hooks)
        try:
            async for _ in watch_run(result, guard):
                pass
        except SoftStop:
            return "I stopped early. Here is what I found so far: ..."
    return str(result.final_output)
```

`show_card` is your function that puts the card in front of a person (a websocket, a Slack message). With a cost ceiling set, a model call that can't be priced stops the run, so give every model your agents use a price. Add budgets for reads and nested agents:

```python
from agent_guardrails import BudgetPolicy, ToolCap, budget_tools, cap_nested_tool

policy = BudgetPolicy(caps={"get_invoice": ToolCap(5, "Use `list_unpaid_invoices`.")})
researcher_tool = cap_nested_tool(
    researcher.as_tool(tool_name="ask_researcher", tool_description="...", hooks=hooks)
)
agent = Agent(name="assistant", tools=[*budget_tools(tools), researcher_tool])
guard = RunGuard(scope=user_id, deadline_s=240, budget=policy, nested_cap=2)
```

When the person clicks, your API layer answers the card:

```python
@app.post("/approvals/{confirm_id}")
async def answer(confirm_id: str, body: Answer):
    return {"status": channel.resolve(confirm_id, body.approved, edits=body.edits)}
```

`resolve` returns `"unknown"` when nothing is waiting any more (already answered, timed out, or its run was cancelled). It takes a real `True` or `False`: anything else, such as the string `"false"` from an untyped form, raises `TypeError` and leaves the card waiting.

## What each guard does

### Fail-closed classification (`classify`)

A tool name is a read only when its first word is a read verb (`get`, `list`, `search`, `fetch`, ...), no word is a write verb (`send`, `create`, `delete`, `set`, `export`, ...), and no word joins two actions (`and`, `or`, `then`). Everything else, including names the classifier has never seen, is a write: `list_and_pay_invoices`, `get_or_create_folder`, `run_query` and `export_to_s3` are all gated. Verbs match whole words of the name, so `get_settings` is a read: `SET` does not fire inside `SETTINGS`.

Whatever comes before the last `.`, `/`, `:` or `__` is a namespace and is skipped, so `crm.contacts.search` is a read. A toolkit prefix joined with a single `_` (`GMAIL_FETCH_EMAILS`) looks like any other first word, so it is gated until you declare it. `extend` returns a new classifier and changes nothing in place; pass that one classifier to every guard, so they agree:

```python
from agent_guardrails import DEFAULT_CLASSIFIER

classifier = DEFAULT_CLASSIFIER.extend(namespaces=["GMAIL"], known_reads=["weather_now"])
tools = gate_tools(raw_tools, channel=channel, store=store, classifier=classifier)
policy = BudgetPolicy(classifier=classifier)
```

`known_reads` names reads the rule can't prove. A tool `gate_tool` wrapped is a write to the read budget whatever its classifier says, so a budget can never cache it.

### The approval gate (`confirm`)

`ConfirmChannel.request()` shows a card and waits. Only an explicit yes approves. A timeout comes back as `unanswered`, not `declined`, so the agent can tell the user nobody answered instead of claiming they said no. A card that couldn't be shown to anyone fails closed.

The card shows the real payload, including the message body, so a prompt-injected draft can't be approved sight unseen. An approver may edit fields you mark editable (the body, never the recipient).

Once one gate in a run expires unanswered, later gates in that run return at once instead of each waiting another full timeout. Gates raised at the same time all wait: the SDK runs a turn's tool calls in parallel and each is a separate decision.

### At-most-once execution (`idempotency`, `stores`)

`guarded_action` claims the action (keyed by a hash of the full arguments), asks for approval, runs it, and records the result. A duplicate inside the window replays the recorded result. Failures are sorted by what can be proved:

| What happened | Status | Retried? |
|---|---|---|
| `execute` raised `PreSideEffectError` | `failed_retryable` | yes |
| Any other error, a timeout or a cancellation while `execute` runs | `abandoned_unknown` | never by itself |
| Declined or unanswered | `aborted` | yes, with a new card |
| The gate failed, or a cancellation came before `execute` started | `aborted` | yes, with a new card |
| Process died while waiting for approval | `failed_retryable` (reaper) | yes |
| Process died while executing | `abandoned_unknown` (reaper) | never by itself |

A write tool may only report success for `executed` or `replayed`. `outcome_message()` words every other outcome so the user knows nothing was done by that call, and a duplicate waiting on another caller's card says whether that card was declined or went unanswered.

The window counts from the success, so a slow approval or a slow write doesn't shorten it. Each claim carries a fencing token: every later change to its row names the token, so a caller whose claim was released or reaped and then taken by someone else can't overwrite the new holder's row or execute on it. The holder renews its lease while it waits for approval and while it executes, so the reaper only reaps the claims of a process that is gone.

With the SDK, `gate_tool` runs the tool body with the SDK's own failure handling off. A raised error is recorded as what it is, and the model is told the same: safe to try again after `PreSideEffectError`, otherwise check before asking again. The tool's `timeout` covers the body only, never the person's approval wait; a body that times out may have sent, so it needs review. Two tools that share a name (two connectors' `send_message`) are told apart by `namespace=`, which defaults to the SDK's qualified tool name.

### Read budget (`budget`)

Per run: identical reads are answered from a memo, identical reads in flight share one call, capped tools refuse after N distinct calls and name the bulk tool to use instead, and a run-wide ceiling sits above the per-tool caps. Writes are never cached or capped, and any write clears the memo so a read-back sees fresh data; a read still running when a write starts or ends is neither cached nor shared. A failed read is never cached. Reads are told apart by tool, not by name, so a nested agent's `search` never answers for the orchestrator's. Caps reserve their slot before awaiting, because the SDK runs a turn's tool calls concurrently and a check-then-act cap lets all of them through.

Past the refusal ceiling the run is looping, and the watchdog stops it. Past 70% of the deadline every read is refused with "answer now", so the run keeps a turn to write an answer. The deadline counts active time: minutes a person spends reading an approval card don't use up the run's time to read.

### Nested agent caps (`nested`)

`cap_nested_tool` wraps an `agent.as_tool(...)`. Past the cap, a consult is refused. Past the soft deadline, every consult is refused, because a nested run started then can't finish and its reads would be refused anyway. Refusals read inside a nested run are worded for an agent, not a person.

### Cost ceiling and watchdog (`cost`, `watchdog`)

`UsageHooks` records every model call's tokens into the run's `RunUsage`, priced by your `PriceTable`. The model is the one the SDK used: the `RunConfig`'s model (pass the same `run_config` to `UsageHooks`), then the agent's, then the SDK's default. A call with no price counts as unpriced, and the total is then reported as a floor, never as a real-looking figure. With a ceiling set, an unpriced call stops the run: a ceiling you can't measure isn't one. No prices ship with the library.

`watch_run` reads the stream and stops the run on a stall, the hard deadline (`HardStop`), or the soft deadline, refusal ceiling and cost ceiling (`SoftStop`). It pauses while a person is reading an approval card, and it never fires a soft stop while an approved write is executing: the hard deadline is stretched by 1.5x instead.

## Production notes

- **Store.** `InMemoryIdempotencyStore` forgets everything on restart. Use one `PostgresIdempotencyStore` per process (the `postgres` extra, schema in `agent_guardrails.postgres.SCHEMA_SQL`), call `reap_stale()` every few minutes and `release_awaiting()` at shutdown. `release_awaiting()` releases only the claims that store made, never another process's.
- **Scope.** Pass `scope` to `RunGuard` (or `gate_tool`). With no scope a gated tool refuses to run rather than share dedup records across customers.
- **Bind before you run.** `RunGuard` must be entered before `Runner.run_streamed`, which copies the context into its own task.
- **Nested spend.** Pass the same `UsageHooks` to each nested `as_tool(hooks=...)`, or nested calls won't count toward the ceiling.
- **The SDK's own approvals.** If you resume runs with the SDK's `needs_approval` interruptions instead of waiting in-band, `require_approval_for_writes(tools)` applies the same fail-closed classification to that flow.

## Testing your own guarded tools

`agent_guardrails.testing.ScriptedModel` replays a script of tool calls through the real runner:

```python
from agent_guardrails.testing import ScriptedModel, call, say

model = ScriptedModel(
    [[call("send_email", to="a@example.com", subject="Hi", body="...")], [say("Sent.")]]
)
agent = Agent(name="test", model=model, tools=tools)
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # Postgres store tests skip without a database
AGENT_GUARDRAILS_TEST_DSN=postgresql://postgres@localhost:5432/postgres pytest
ruff check . && ruff format --check . && mypy
```

CI runs the suite on Python 3.11 to 3.13 against Postgres 16, checks that the core imports with no dependencies, runs both examples, and scans the history with gitleaks.

Pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems privately, as [SECURITY.md](SECURITY.md) describes, not in an issue. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE).
