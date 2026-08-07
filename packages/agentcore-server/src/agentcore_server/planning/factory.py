from __future__ import annotations

from typing import Any

from agentcore_server.planning.exploration import ExplorationLimits
from agentcore_server.planning.iterative import IterativeLLMPlanner
from agentcore_server.planning.llm import SimpleLLMPlanner


def build_planner(
    config: dict[str, Any],
    *,
    mode_override: str | None = None,
):
    planner_config = config.get("planner", {})
    if not isinstance(planner_config, dict):
        raise ValueError("planner configuration must be a mapping")
    mode = mode_override or planner_config.get("mode", "simple")
    max_tokens = planner_config.get("max_tokens", 1024)
    temperature = planner_config.get("temperature", 0.0)
    if mode == "simple":
        return SimpleLLMPlanner(max_tokens=max_tokens, temperature=temperature)
    if mode != "iterative":
        raise ValueError(f"unknown planner mode: {mode}")

    workspace_config = config.get("workspace", {})
    checks = workspace_config.get("checks", {}) if isinstance(workspace_config, dict) else {}
    if not isinstance(checks, dict):
        raise ValueError("workspace checks configuration must be a mapping")
    return IterativeLLMPlanner(
        max_tokens=max_tokens,
        temperature=temperature,
        limits=ExplorationLimits.from_config(planner_config.get("exploration")),
        check_names=tuple(checks),
    )
