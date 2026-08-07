from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from agentcore_server.workspace.checks import CheckExecutionError

if TYPE_CHECKING:
    from agentcore_server.executor.executor import TaskExecutionContext


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    action_type: str
    status: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, *, action_id: str, action_type: str, data: dict[str, Any] | None = None) -> "ActionResult":
        return cls(
            action_id=action_id,
            action_type=action_type,
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=deepcopy(data or {}),
        )

    @classmethod
    def failed(
        cls,
        *,
        action_id: str,
        action_type: str,
        error: str,
        data: dict[str, Any] | None = None,
    ) -> "ActionResult":
        return cls(
            action_id=action_id,
            action_type=action_type,
            status="failed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=deepcopy(data or {}),
            error=error,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "status": self.status,
            "timestamp": self.timestamp,
            "data": deepcopy(self.data),
            "error": self.error,
        }


class Action(Protocol):
    id: str

    @property
    def action_type(self) -> str:
        ...

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        ...


@dataclass(frozen=True)
class ReadFileAction:
    path: str | Path
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "read_file"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        text = context.agent.files.read_text(self.path)
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "path": str(self.path),
                "content": text,
                "bytes_read": len(text.encode("utf-8")),
                "lines_read": len(text.splitlines()),
            },
        )


@dataclass(frozen=True)
class WriteFileAction:
    path: str | Path
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "write_file"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        result = context.agent.files.write_text(self.path, self.content)
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "path": result.path,
                "bytes_written": result.bytes_written,
                "lines_written": result.lines_written,
                "files_changed": list(result.files_changed),
            },
        )


@dataclass(frozen=True)
class ReplaceTextAction:
    path: str | Path
    old: str
    new: str
    count: int = -1
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "replace_text"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        result = context.agent.files.replace_text(self.path, self.old, self.new, count=self.count)
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "path": result.path,
                "bytes_written": result.bytes_written,
                "lines_written": result.lines_written,
                "replacements": result.replacements,
                "files_changed": list(result.files_changed),
            },
        )


@dataclass(frozen=True)
class CreateCheckpointAction:
    label: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "create_checkpoint"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        checkpoint = context.task.create_checkpoint(
            self.label,
            self.description,
            metadata=self.metadata,
        )
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={"checkpoint": checkpoint.as_dict()},
        )


@dataclass(frozen=True)
class GitStatusAction:
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "git_status"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        result = context.agent.git.status()
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


@dataclass(frozen=True)
class GitDiffAction:
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "git_diff"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        result = context.agent.git.diff()
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


@dataclass(frozen=True)
class TaskReportAction:
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "task_report"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        report = context.task.report()
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={
                "report": report.as_dict(),
                "report_kind": "intermediate_snapshot",
            },
        )


@dataclass(frozen=True)
class RunCheckAction:
    check: str
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def action_type(self) -> str:
        return "run_check"

    def execute(self, context: TaskExecutionContext) -> ActionResult:
        if context.agent.workspace.read_only:
            raise PermissionError("run_check is not allowed in a read-only workspace")
        result = context.agent.workspace.checks.run(self.check)
        if not result.ok:
            raise CheckExecutionError(result)
        return ActionResult.ok(
            action_id=self.id,
            action_type=self.action_type,
            data={"check": result.as_dict()},
        )
