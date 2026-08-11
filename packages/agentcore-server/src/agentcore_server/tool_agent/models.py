from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from agentcore_server.generation import ToolCall, ToolResult
from agentcore_server.tasks import TaskReport


@dataclass(frozen=True)
class ToolAgentLimits:
    max_model_turns: int = 40
    completion_runway_turns: int = 12
    progress_window_turns: int = 8
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
    context_recovery_target_tokens: int = 2048
    preserve_recent_tool_results: int = 8
    max_edit_old_bytes: int = 64 * 1024
    max_edit_new_bytes: int = 64 * 1024
    max_write_file_bytes: int = 256 * 1024
    max_preview_bytes: int = 128 * 1024
    max_changed_lines: int = 1200
    max_rejected_side_effecting_calls: int = 5
    max_consecutive_rejected_side_effecting_calls: int = 5

    def __post_init__(self) -> None:
        if self.completion_runway_turns > 12:
            raise ValueError("completion_runway_turns must not exceed 12")
        if self.max_model_turns + self.completion_runway_turns > 52:
            raise ValueError(
                "max_model_turns plus completion_runway_turns must not exceed 52"
            )

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "ToolAgentLimits":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("tool_agent configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        protocol_keys = {"protocol", "reasoning_effort"}
        unknown = sorted(set(data) - allowed - protocol_keys)
        if unknown:
            raise ValueError("unknown tool_agent setting(s): " + ", ".join(unknown))
        limit_data = {name: value for name, value in data.items() if name in allowed}
        for name, value in limit_data.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"tool_agent setting '{name}' must be a positive integer")
        if limit_data.get("max_single_tool_result_bytes", cls.max_single_tool_result_bytes) < 256:
            raise ValueError("max_single_tool_result_bytes must be at least 256")
        return cls(**limit_data)


@dataclass(frozen=True)
class ToolEffectPreview:
    preview_id: str
    tool_call_id: str
    tool: str
    target: str | None
    effect_type: str
    content: str
    content_sha256: str
    digest: str
    source_sha256: str | None
    result_sha256: str | None
    source_exists: bool | None
    match_count: int | None
    changed_bytes: int
    changed_lines: int
    within_limits: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "preview_id": self.preview_id,
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "target": self.target,
            "effect_type": self.effect_type,
            "content_bytes": len(self.content.encode("utf-8")),
            "content_sha256": self.content_sha256,
            "digest": self.digest,
            "source_sha256": self.source_sha256,
            "result_sha256": self.result_sha256,
            "source_exists": self.source_exists,
            "match_count": self.match_count,
            "changed_bytes": self.changed_bytes,
            "changed_lines": self.changed_lines,
            "within_limits": self.within_limits,
            "metadata": deepcopy(self.metadata),
            "complete_content_available": True,
        }


@dataclass(frozen=True)
class ToolApprovalRequest:
    call: ToolCall
    target: str | None
    arguments: dict[str, Any]
    expected_side_effect: str
    preview: ToolEffectPreview
    preview_artifact: str | None = None

    def as_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        preview = self.preview.summary_dict()
        if include_content:
            preview["content"] = self.preview.content
        return {
            "tool_call_id": self.call.id,
            "tool": self.call.function_name,
            "target": self.target,
            "arguments": deepcopy(self.arguments),
            "expected_side_effect": self.expected_side_effect,
            "preview": preview,
            "preview_artifact": self.preview_artifact,
        }


@dataclass(frozen=True)
class ToolApprovalDecision:
    approved: bool
    tool_call_id: str
    preview_digest: str
    reason: str | None = None


class ToolApprovalGateway(Protocol):
    def request(self, request: ToolApprovalRequest) -> ToolApprovalDecision:
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
class ToolRunResult:
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


# Backward-compatible names for existing local integrations and serialized imports.
QwenToolAgentLimits = ToolAgentLimits
QwenToolRunResult = ToolRunResult
