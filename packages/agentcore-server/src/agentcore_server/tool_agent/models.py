from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from agentcore_server.generation import ToolCall, ToolResult
from agentcore_server.tasks import TaskReport


@dataclass(frozen=True)
class QwenToolAgentLimits:
    max_model_turns: int = 32
    max_total_tool_calls: int = 64
    max_tool_calls_per_turn: int = 8
    max_consecutive_tool_failures: int = 4
    max_single_tool_result_bytes: int = 64 * 1024
    max_total_tool_result_bytes: int = 512 * 1024
    default_read_lines: int = 120
    max_read_lines: int = 500
    max_directory_depth: int = 4
    max_search_results: int = 100
    context_safety_margin_tokens: int = 128
    minimum_output_tokens: int = 256

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "QwenToolAgentLimits":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("tool_agent configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("unknown tool_agent setting(s): " + ", ".join(unknown))
        for name, value in data.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"tool_agent setting '{name}' must be a positive integer")
        if data.get("max_single_tool_result_bytes", cls.max_single_tool_result_bytes) < 256:
            raise ValueError("max_single_tool_result_bytes must be at least 256")
        return cls(**data)


@dataclass(frozen=True)
class ToolApprovalRequest:
    call: ToolCall
    target: str | None
    arguments: dict[str, Any]
    expected_side_effect: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.call.id,
            "tool": self.call.function_name,
            "target": self.target,
            "arguments": deepcopy(self.arguments),
            "expected_side_effect": self.expected_side_effect,
        }


class ToolApprovalGateway(Protocol):
    def request(self, request: ToolApprovalRequest) -> bool:
        ...


class ToolSteeringInbox:
    def __init__(self) -> None:
        self._message: str | None = None
        self._lock = Lock()

    def queue(self, message: str) -> bool:
        if not message.strip():
            raise ValueError("steering message must not be empty")
        with self._lock:
            if self._message is not None:
                return False
            self._message = message
            return True

    def take(self) -> str | None:
        with self._lock:
            message = self._message
            self._message = None
            return message


@dataclass(frozen=True)
class QwenToolRunResult:
    status: str
    final_text: str
    turns: int
    tool_calls: int
    tool_results: tuple[ToolResult, ...]
    report: TaskReport
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_text": self.final_text,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_results": [result.as_dict() for result in self.tool_results],
            "report": self.report.as_dict(),
            "error": self.error,
            "metadata": deepcopy(self.metadata),
        }
