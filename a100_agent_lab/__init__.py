from __future__ import annotations

import importlib
import sys
import warnings


warnings.warn(
    "a100_agent_lab is deprecated; use agentcore_server instead.",
    DeprecationWarning,
    stacklevel=2,
)

_MODULES = [
    "agents",
    "agents.agent",
    "api",
    "api.client",
    "events",
    "executor",
    "executor.actions",
    "executor.executor",
    "executor.plan",
    "executor.proposal",
    "generation",
    "generation.config",
    "generation.result",
    "generation.stream",
    "health",
    "logging",
    "logging.events",
    "logging.writer",
    "planner",
    "planning",
    "planning.llm",
    "planning.planner",
    "runtime",
    "runtime.base",
    "runtime.health",
    "runtime.lmdeploy",
    "runtime.server_process",
    "runtime.sglang",
    "runtime.transformers",
    "server",
    "server.app",
    "server.events",
    "server.schemas",
    "server.state",
    "sessions",
    "sessions.session",
    "sessions.store",
    "tasks",
    "tasks.task",
    "workspace",
    "workspace.files",
    "workspace.git",
    "workspace.workspace",
]

for _name in _MODULES:
    sys.modules[f"a100_agent_lab.{_name}"] = importlib.import_module(f"agentcore_server.{_name}")

from agentcore_server import *  # noqa: F401,F403,E402
