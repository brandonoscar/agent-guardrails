# Security policy

## Reporting a vulnerability

Report it privately on GitHub: open the **Security** tab and choose **Report a vulnerability**. Please don't open a public issue for it.

Say what you ran, what happened and what should have happened. Expect a first reply within a week. Once a fix ships, the advisory is published and credits you, unless you'd rather not be named.

## What counts

- A gated tool that runs without an approval, or an unknown tool treated as a read.
- An approved action that executes twice under one idempotency key, or runs again after its outcome was unknown.
- A cap, budget, deadline or cost ceiling that a run gets past.
- Idempotency records from one `scope` that another scope can see or replay.

## Supported versions

Only the latest release gets fixes.
