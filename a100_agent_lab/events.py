from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentEvent:
    event_type: str
    summary: str
    task_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "payload": deepcopy(self.payload),
        }


class EventSink(Protocol):
    def emit(self, event: AgentEvent) -> None:
        ...


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]
