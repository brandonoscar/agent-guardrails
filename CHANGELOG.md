# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-02

First release.

### Added

- Fail-closed write approval: `ToolClassifier`, `ConfirmChannel`, and `gate_tool` / `gate_tools` for OpenAI Agents SDK tools. Unknown tools count as writes.
- At-most-once actions: `guarded_action` and `run_guarded`, with an in-memory store and a Postgres store.
- Run budgets: per-tool caps, a cache for repeated reads, a per-run record budget and refusal ceiling (`BudgetPolicy`), and a cap on nested agent consults (`NestedLedger`).
- A per-run cost ceiling (`RunUsage`, `PriceTable`, `UsageHooks`) and a stream watchdog (`watch_run`) with soft and hard stops.
- `RunGuard`, which binds all of the above to one run, and `testing.ScriptedModel` for offline tests.
- Examples: `examples/send_email.py` and `examples/runaway.py`.

[Unreleased]: https://github.com/brandonoscar/agent-guardrails/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/brandonoscar/agent-guardrails/releases/tag/v0.1.0
