from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agentcore_server.generation.tools import ToolCall, ToolResult


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_success: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.as_openai_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data


@dataclass
class Session:
    system_prompt: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.system_prompt:
            self.messages.append(Message(role="system", content=self.system_prompt))

    def add_user_message(self, text: str) -> None:
        self.messages.append(Message(role="user", content=text))
        self._touch()

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(Message(role="assistant", content=text))
        self._touch()

    def add_assistant_tool_message(self, text: str, tool_calls: tuple[ToolCall, ...]) -> None:
        self.messages.append(Message(role="assistant", content=text, tool_calls=tool_calls))
        self._touch()

    def add_tool_result(self, result: ToolResult) -> None:
        self.messages.append(
            Message(
                role="tool",
                content=result.content,
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                tool_success=result.success,
            )
        )
        self._touch()

    def transcript(self) -> list[dict[str, Any]]:
        return [message.as_dict() for message in self.messages]

    @property
    def turn_count(self) -> int:
        return sum(1 for message in self.messages if message.role == "assistant")

    def reset(self) -> None:
        self.messages.clear()
        if self.system_prompt:
            self.messages.append(Message(role="system", content=self.system_prompt))
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
