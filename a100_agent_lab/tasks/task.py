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


@dataclass(frozen=True)
class TaskReport:
    id: str
    title: str
    description: str
    status: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    failed_at: str | None
    cancelled_at: str | None
    failure_reason: str | None
    git_branch: str | None = None
    git_status: str | None = None
    git_diff: str | None = None
    files_changed: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: "Task",
        *,
        git_branch: str | None = None,
        git_status: str | None = None,
        git_diff: str | None = None,
        files_changed: tuple[str, ...] = (),
    ) -> "TaskReport":
        return cls(
            id=task.id,
            title=task.title,
            description=task.description,
            status=task.status.value,
            created_at=task.created_at.isoformat(),
            updated_at=task.updated_at.isoformat(),
            started_at=None if task.started_at is None else task.started_at.isoformat(),
            completed_at=None if task.completed_at is None else task.completed_at.isoformat(),
            failed_at=None if task.failed_at is None else task.failed_at.isoformat(),
            cancelled_at=None if task.cancelled_at is None else task.cancelled_at.isoformat(),
            failure_reason=task.failure_reason,
            git_branch=git_branch,
            git_status=git_status,
            git_diff=git_diff,
            files_changed=files_changed,
            metadata=dict(task.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "cancelled_at": self.cancelled_at,
            "failure_reason": self.failure_reason,
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "files_changed": list(self.files_changed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskCheckpoint:
    id: str
    task_id: str
    timestamp: str
    label: str
    description: str | None = None
    git_branch: str | None = None
    git_status: str | None = None
    git_diff: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: "Task",
        *,
        label: str,
        description: str | None = None,
        git_branch: str | None = None,
        git_status: str | None = None,
        git_diff: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> "TaskCheckpoint":
        return cls(
            id=uuid4().hex,
            task_id=task.id,
            timestamp=(timestamp or task._now()).isoformat(),
            label=label,
            description=description,
            git_branch=git_branch,
            git_status=git_status,
            git_diff=git_diff,
            metadata=dict(metadata or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "label": self.label,
            "description": self.description,
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "metadata": dict(self.metadata),
        }


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
    _checkpoints: list[TaskCheckpoint] = field(default_factory=list, repr=False)
    _on_event: Callable[["Task", str], None] | None = field(default=None, repr=False, compare=False)
    _reporter: Callable[["Task"], TaskReport] | None = field(default=None, repr=False, compare=False)
    _checkpoint_builder: Callable[
        ["Task", str, str | None, dict[str, Any] | None],
        TaskCheckpoint,
    ] | None = field(default=None, repr=False, compare=False)

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
            "checkpoints": [checkpoint.as_dict() for checkpoint in self._checkpoints],
        }

    def report(self) -> TaskReport:
        if self._reporter is not None:
            return self._reporter(self)
        return TaskReport.from_task(self)

    def create_checkpoint(
        self,
        label: str,
        description: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TaskCheckpoint:
        if not label.strip():
            raise ValueError("checkpoint label must not be empty")
        checkpoint = self._build_checkpoint(label, description, metadata)
        self._checkpoints.append(checkpoint)
        self.updated_at = datetime.fromisoformat(checkpoint.timestamp)
        return checkpoint

    def checkpoints(self) -> list[TaskCheckpoint]:
        return list(self._checkpoints)

    def latest_checkpoint(self) -> TaskCheckpoint | None:
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def _require_status(self, allowed: set[TaskStatus]) -> None:
        if self.status not in allowed:
            allowed_values = ", ".join(sorted(status.value for status in allowed))
            raise ValueError(f"cannot transition task from {self.status.value}; expected one of: {allowed_values}")

    def _build_checkpoint(
        self,
        label: str,
        description: str | None,
        metadata: dict[str, Any] | None,
    ) -> TaskCheckpoint:
        if self._checkpoint_builder is not None:
            return self._checkpoint_builder(self, label, description, metadata)
        return TaskCheckpoint.from_task(self, label=label, description=description, metadata=metadata)

    def _emit(self, event_type: str) -> None:
        if self._on_event is not None:
            self._on_event(self, event_type)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
