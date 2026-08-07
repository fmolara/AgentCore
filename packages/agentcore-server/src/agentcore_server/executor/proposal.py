from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import json

from agentcore_server.executor.executor import TaskExecutionResult, TaskExecutor
from agentcore_server.executor.plan import ActionPlan, ApprovalPolicy, ApprovalRequirement
from agentcore_server.tasks import Task


class PlanProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class PlanProposal:
    task_id: str
    title: str
    summary: str
    action_plan: ActionPlan
    approval_requirements: tuple[ApprovalRequirement, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    status: PlanProposalStatus = PlanProposalStatus.PROPOSED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    executed_at: datetime | None = None

    @classmethod
    def from_action_plan(
        cls,
        *,
        task_id: str,
        action_plan: ActionPlan,
        title: str | None = None,
        summary: str = "",
        approval_policy: ApprovalPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PlanProposal":
        return cls(
            task_id=task_id,
            title=title or action_plan.title,
            summary=summary,
            action_plan=action_plan,
            approval_requirements=action_plan.required_approvals(approval_policy),
            metadata=deepcopy(metadata or {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanProposal":
        if not isinstance(data, dict):
            raise ValueError("plan proposal must be a mapping")
        proposal_id = _required_str(data, "id")
        task_id = _required_str(data, "task_id")
        title = _required_str(data, "title")
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("plan proposal field 'summary' must be a string")
        action_plan_data = data.get("action_plan")
        if not isinstance(action_plan_data, dict):
            raise ValueError("plan proposal field 'action_plan' must be a mapping")
        requirements_data = data.get("approval_requirements", [])
        if not isinstance(requirements_data, list):
            raise ValueError("plan proposal field 'approval_requirements' must be a list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("plan proposal field 'metadata' must be a mapping")

        return cls(
            id=proposal_id,
            task_id=task_id,
            title=title,
            summary=summary,
            action_plan=ActionPlan.from_dict(action_plan_data),
            approval_requirements=tuple(_approval_requirement_from_dict(item) for item in requirements_data),
            status=_status_from_value(_required_str(data, "status")),
            created_at=_datetime_from_iso(_required_str(data, "created_at"), "created_at"),
            updated_at=_datetime_from_iso(_required_str(data, "updated_at"), "updated_at"),
            approved_at=_optional_datetime(data.get("approved_at"), "approved_at"),
            rejected_at=_optional_datetime(data.get("rejected_at"), "rejected_at"),
            executed_at=_optional_datetime(data.get("executed_at"), "executed_at"),
            metadata=deepcopy(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PlanProposal":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def approve(self) -> "PlanProposal":
        if self.status != PlanProposalStatus.PROPOSED:
            raise ValueError(f"cannot approve proposal with status: {self.status.value}")
        now = _now()
        self.status = PlanProposalStatus.APPROVED
        self.approved_at = now
        self.updated_at = now
        return self

    def reject(self, reason: str) -> "PlanProposal":
        if self.status != PlanProposalStatus.PROPOSED:
            raise ValueError(f"cannot reject proposal with status: {self.status.value}")
        if not reason:
            raise ValueError("rejection reason must be non-empty")
        now = _now()
        self.status = PlanProposalStatus.REJECTED
        self.rejected_at = now
        self.updated_at = now
        self.metadata["rejection_reason"] = reason
        return self

    def execute(self, executor: TaskExecutor, task: Task) -> TaskExecutionResult:
        if task.id != self.task_id:
            raise ValueError("proposal task_id does not match task")
        if self.status == PlanProposalStatus.REJECTED:
            raise ValueError("cannot execute rejected proposal")
        if self.status == PlanProposalStatus.EXECUTED:
            raise ValueError("cannot execute proposal more than once")

        approved = self.status == PlanProposalStatus.APPROVED
        if self.approval_requirements and not approved:
            reasons = "; ".join(
                f"action {requirement.action_index} ({requirement.action_type}): {requirement.reason}"
                for requirement in self.approval_requirements
            )
            return TaskExecutionResult(
                task_id=task.id,
                status="approval_required",
                actions=(),
                report=task.report(),
                error="plan proposal requires approval: " + reasons,
            )

        result = executor.execute_plan(task, self.action_plan, approved=approved)
        if result.status == "completed":
            now = _now()
            self.status = PlanProposalStatus.EXECUTED
            self.executed_at = now
            self.updated_at = now
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "title": self.title,
            "summary": self.summary,
            "action_plan": self.action_plan.as_dict(),
            "approval_requirements": [requirement.as_dict() for requirement in self.approval_requirements],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_at": None if self.approved_at is None else self.approved_at.isoformat(),
            "rejected_at": None if self.rejected_at is None else self.rejected_at.isoformat(),
            "executed_at": None if self.executed_at is None else self.executed_at.isoformat(),
            "metadata": deepcopy(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _required_str(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required field '{field_name}' must be a non-empty string")
    return value


def _status_from_value(value: str) -> PlanProposalStatus:
    try:
        return PlanProposalStatus(value)
    except ValueError as exc:
        raise ValueError(f"unknown plan proposal status: {value}") from exc


def _datetime_from_iso(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"field '{field_name}' must be an ISO datetime") from exc


def _optional_datetime(value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"field '{field_name}' must be an ISO datetime or null")
    return _datetime_from_iso(value, field_name)


def _approval_requirement_from_dict(data: Any) -> ApprovalRequirement:
    if not isinstance(data, dict):
        raise ValueError("approval requirement must be a mapping")
    action_index = data.get("action_index")
    if not isinstance(action_index, int):
        raise ValueError("approval requirement field 'action_index' must be an integer")
    action_type = _required_str(data, "action_type")
    reason = _required_str(data, "reason")
    return ApprovalRequirement(action_index=action_index, action_type=action_type, reason=reason)
