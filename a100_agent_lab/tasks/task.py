from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable
from uuid import uuid4


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    title: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    status: TaskStatus = TaskStatus.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None
    failure_reason: str | None = None
    _on_event: Callable[["Task", str], None] | None = field(default=None, repr=False, compare=False)

    def start(self) -> None:
        self._require_status({TaskStatus.CREATED})
        now = self._now()
        self.status = TaskStatus.RUNNING
        self.started_at = now
        self.updated_at = now
        self._emit("task_started")

    def complete(self) -> None:
        self._require_status({TaskStatus.CREATED, TaskStatus.RUNNING})
        now = self._now()
        self.status = TaskStatus.COMPLETED
        self.completed_at = now
        self.updated_at = now
        self._emit("task_completed")

    def fail(self, reason: str) -> None:
        self._require_status({TaskStatus.CREATED, TaskStatus.RUNNING})
        if not reason.strip():
            raise ValueError("failure reason must not be empty")
        now = self._now()
        self.status = TaskStatus.FAILED
        self.failure_reason = reason
        self.failed_at = now
        self.updated_at = now
        self._emit("task_failed")

    def cancel(self) -> None:
        self._require_status({TaskStatus.CREATED, TaskStatus.RUNNING})
        now = self._now()
        self.status = TaskStatus.CANCELLED
        self.cancelled_at = now
        self.updated_at = now
        self._emit("task_cancelled")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": None if self.started_at is None else self.started_at.isoformat(),
            "completed_at": None if self.completed_at is None else self.completed_at.isoformat(),
            "failed_at": None if self.failed_at is None else self.failed_at.isoformat(),
            "cancelled_at": None if self.cancelled_at is None else self.cancelled_at.isoformat(),
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
        }

    def _require_status(self, allowed: set[TaskStatus]) -> None:
        if self.status not in allowed:
            allowed_values = ", ".join(sorted(status.value for status in allowed))
            raise ValueError(f"cannot transition task from {self.status.value}; expected one of: {allowed_values}")

    def _emit(self, event_type: str) -> None:
        if self._on_event is not None:
            self._on_event(self, event_type)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
