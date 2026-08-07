from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    event_type: str
    summary: str
    task_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        if not isinstance(data, dict):
            raise ValueError("AgentEvent must be a mapping")
        event_type = _required_str(data, "event_type")
        summary = _required_str(data, "summary")
        task_id = _optional_str(data.get("task_id"), "task_id")
        session_id = _optional_str(data.get("session_id"), "session_id")
        timestamp = _required_str(data, "timestamp")
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("AgentEvent payload must be a mapping")
        return cls(
            event_type=event_type,
            summary=summary,
            task_id=task_id,
            session_id=session_id,
            payload=deepcopy(payload),
            timestamp=timestamp,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "payload": deepcopy(self.payload),
        }


def _required_str(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required field '{field_name}' must be a non-empty string")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"field '{field_name}' must be a string or null")
    return value
