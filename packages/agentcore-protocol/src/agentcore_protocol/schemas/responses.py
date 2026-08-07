from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HealthResponse:
    status: str
    ready: bool
    api_version: str | None = None
    protocol_version: str | None = None
    schema_version: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthResponse":
        return cls(
            status=str(data.get("status", "")),
            ready=bool(data.get("ready", False)),
            api_version=_optional_str(data.get("api_version")),
            protocol_version=_optional_str(data.get("protocol_version")),
            schema_version=_optional_str(data.get("schema_version")),
            runtime=_dict(data.get("runtime")),
            counts=_dict(data.get("counts")),
            raw=deepcopy(data),
        )


@dataclass(frozen=True)
class AgentResponse:
    id: str
    session_id: str
    workspace: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponse":
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            workspace=_dict(data.get("workspace")),
            statistics=_dict(data.get("statistics")),
            raw=deepcopy(data),
        )


@dataclass(frozen=True)
class TaskResponse:
    task: dict[str, Any]
    agent_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResponse":
        return cls(
            task=_dict(data.get("task")),
            agent_id=_optional_str(data.get("agent_id")),
            raw=deepcopy(data),
        )

    @property
    def id(self) -> str:
        return str(self.task["id"])


@dataclass(frozen=True)
class ApprovalRequirement:
    action_index: int
    action_type: str
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequirement":
        return cls(
            action_index=int(data["action_index"]),
            action_type=str(data["action_type"]),
            reason=str(data["reason"]),
            raw=deepcopy(data),
        )


@dataclass(frozen=True)
class ProposalResponse:
    proposal: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposalResponse":
        return cls(proposal=_dict(data.get("proposal")), raw=deepcopy(data))

    @property
    def id(self) -> str:
        return str(self.proposal["id"])

    @property
    def approval_requirements(self) -> tuple[ApprovalRequirement, ...]:
        items = self.proposal.get("approval_requirements", [])
        if not isinstance(items, list):
            return ()
        return tuple(ApprovalRequirement.from_dict(item) for item in items if isinstance(item, dict))


@dataclass(frozen=True)
class PlannerResult:
    status: str
    proposal: ProposalResponse | None = None
    raw_text: str = ""
    error: str | None = None
    metrics: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannerResult":
        proposal_data = data.get("proposal")
        return cls(
            status=str(data.get("status", "")),
            proposal=None if proposal_data is None else ProposalResponse.from_dict({"proposal": proposal_data}),
            raw_text=str(data.get("raw_text", "")),
            error=_optional_str(data.get("error")),
            metrics=None if data.get("metrics") is None else _dict(data.get("metrics")),
            raw=deepcopy(data),
        )


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    status: str
    actions: tuple[dict[str, Any], ...] = ()
    report: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        report = data.get("report")
        return cls(
            task_id=str(data.get("task_id", "")),
            status=str(data.get("status", "")),
            actions=tuple(deepcopy(action) for action in actions if isinstance(action, dict)),
            report=None if report is None else _dict(report),
            error=_optional_str(data.get("error")),
            raw=deepcopy(data),
        )


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GitResult":
        return cls(
            returncode=int(data.get("returncode", 0)),
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            raw=deepcopy(data),
        )


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
