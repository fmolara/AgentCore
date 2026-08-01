from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agentcore_server.generation.result import GenerationMetrics


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    function_name: str | None = None
    arguments_delta: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "function_name": self.function_name,
            "arguments_delta": self.arguments_delta,
        }


@dataclass(frozen=True)
class ToolCall:
    id: str
    index: int
    function_name: str
    argument_text: str
    arguments: dict[str, Any] | None
    parsing_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "function_name": self.function_name,
            "argument_text": self.argument_text,
            "arguments": deepcopy(self.arguments),
            "parsing_error": self.parsing_error,
        }

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function_name,
                # Qwen's current chat template consumes an argument mapping.
                "arguments": deepcopy(self.arguments) if self.arguments is not None else {},
            },
        }


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "content": self.content,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class AssistantTurn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    metrics: GenerationMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class ToolTurnChunk:
    chunk_type: str
    text_delta: str = ""
    tool_call_delta: ToolCallDelta | None = None
    turn: AssistantTurn | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def started(cls, *, metadata: dict[str, Any]) -> "ToolTurnChunk":
        return cls(chunk_type="started", metadata=deepcopy(metadata))

    @classmethod
    def text(cls, value: str) -> "ToolTurnChunk":
        return cls(chunk_type="text_delta", text_delta=value)

    @classmethod
    def tool_delta(cls, value: ToolCallDelta) -> "ToolTurnChunk":
        return cls(chunk_type="tool_call_delta", tool_call_delta=value)

    @classmethod
    def completed(cls, turn: AssistantTurn) -> "ToolTurnChunk":
        return cls(chunk_type="completed", turn=turn)

    @classmethod
    def failed(cls, error: str) -> "ToolTurnChunk":
        return cls(chunk_type="failed", error=error)
