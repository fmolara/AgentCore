from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from agentcore_server.generation.config import GenerationConfig


TOOL_AGENT_SYSTEM_PROMPT = """You are a careful coding agent operating on a real workspace.
Use the native tools to inspect files before editing. Do not stop at an up-front ActionPlan: the host requests operator approval for each concrete side-effecting tool call. Prefer edit for exact localized changes to existing files; use write_file mainly for new files. Never rewrite a complete project when a local edit is sufficient. If a large edit is rejected, split it into smaller localized exact edits. Inspect tool failures and recover instead of claiming success. Do not finish merely because edits were applied. When the task explicitly requests configured build or test checks, run them and repair any failures before finishing. Inspect git_diff after the final changes. Never claim a tool succeeded unless its tool result says so. Do not create commits. Return a concise final summary only after required checks and diff inspection are complete. Do not expose hidden reasoning."""


@dataclass(frozen=True)
class DecodedToolDelta:
    visible_text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()


class ToolProtocolAdapter:
    """Small model-protocol boundary around one shared tool loop."""

    name = "openai"

    def encode_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(message) for message in messages]

    def request_options(self, generation: GenerationConfig) -> dict[str, Any]:
        return {}

    def template_options(self, generation: GenerationConfig) -> dict[str, Any]:
        return {}

    def decode_delta(self, delta: dict[str, Any]) -> DecodedToolDelta:
        return DecodedToolDelta(
            visible_text=delta.get("content") or "",
            tool_calls=tuple(delta.get("tool_calls") or ()),
        )


class QwenToolProtocol(ToolProtocolAdapter):
    name = "qwen"

    def request_options(self, generation: GenerationConfig) -> dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": generation.enable_thinking}}

    def template_options(self, generation: GenerationConfig) -> dict[str, Any]:
        return {"enable_thinking": generation.enable_thinking}


class MistralToolProtocol(ToolProtocolAdapter):
    name = "mistral"

    def encode_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _encode_openai_tool_arguments(messages)


class HarmonyToolProtocol(ToolProtocolAdapter):
    name = "harmony"

    def __init__(self, *, reasoning_effort: str = "medium") -> None:
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Harmony reasoning_effort must be low, medium, or high")
        self.reasoning_effort = reasoning_effort

    def encode_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for message in _encode_openai_tool_arguments(messages):
            item = dict(message)
            if item.get("role") == "system":
                item["role"] = "developer"
            encoded.append(item)
        return encoded

    def request_options(self, generation: GenerationConfig) -> dict[str, Any]:
        return {"reasoning_effort": self.reasoning_effort}


def protocol_from_config(data: dict[str, Any] | None) -> ToolProtocolAdapter:
    config = data or {}
    if not isinstance(config, dict):
        raise ValueError("tool_agent configuration must be a mapping")
    name = str(config.get("protocol", "qwen")).strip().lower()
    if name == "qwen":
        return QwenToolProtocol()
    if name == "mistral":
        return MistralToolProtocol()
    if name in {"harmony", "gpt-oss", "gpt_oss"}:
        return HarmonyToolProtocol(
            reasoning_effort=str(config.get("reasoning_effort", "medium"))
        )
    raise ValueError(f"unsupported tool protocol: {name}")


def _encode_openai_tool_arguments(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render assistant tool arguments in the OpenAI wire representation."""
    encoded: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        calls = item.get("tool_calls")
        if not calls:
            encoded.append(item)
            continue
        encoded_calls: list[dict[str, Any]] = []
        for call in calls:
            encoded_call = dict(call)
            function = dict(encoded_call.get("function") or {})
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(
                    arguments if arguments is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            encoded_call["function"] = function
            encoded_calls.append(encoded_call)
        item["tool_calls"] = encoded_calls
        encoded.append(item)
    return encoded
