"""A minimal stand-in for the SDK's FunctionTool, so the core tests need no SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTool:
    name: str
    on_invoke_tool: Callable[[Any, str], Awaitable[Any]]
    description: str = ""


@dataclass
class Recorder:
    """A tool body that records each call and answers from ``respond``."""

    delay: float = 0.0
    fail: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, ctx: Any, args_json: str) -> str:
        args = json.loads(args_json or "{}")
        self.calls.append(args)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("upstream down")
        return f"result for {json.dumps(args, sort_keys=True)}"


def tool(name: str, body: Recorder | None = None) -> tuple[FakeTool, Recorder]:
    body = body or Recorder()
    return FakeTool(name=name, on_invoke_tool=body), body


async def invoke(t: FakeTool, **args: Any) -> Any:
    return await t.on_invoke_tool(None, json.dumps(args))
