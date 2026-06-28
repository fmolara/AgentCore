from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


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

    def transcript(self) -> list[dict[str, str]]:
        return [message.as_dict() for message in self.messages]

    def reset(self) -> None:
        self.messages.clear()
        if self.system_prompt:
            self.messages.append(Message(role="system", content=self.system_prompt))
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

