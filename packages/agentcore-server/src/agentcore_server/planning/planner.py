from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from agentcore_server.executor import ApprovalPolicy, PlanProposal
from agentcore_server.generation.result import GenerationMetrics
from agentcore_server.tasks import Task

if TYPE_CHECKING:
    from agentcore_server.agents import Agent


@dataclass(frozen=True)
class PlannerResult:
    status: str
    proposal: PlanProposal | None = None
    raw_text: str = ""
    error: str | None = None
    metrics: GenerationMetrics | None = None

    @classmethod
    def proposed(
        cls,
        *,
        proposal: PlanProposal,
        raw_text: str,
        metrics: GenerationMetrics | None,
    ) -> "PlannerResult":
        return cls(status="proposed", proposal=proposal, raw_text=raw_text, metrics=metrics)

    @classmethod
    def failed(
        cls,
        *,
        error: str,
        raw_text: str = "",
        metrics: GenerationMetrics | None = None,
    ) -> "PlannerResult":
        return cls(status="failed", raw_text=raw_text, error=error, metrics=metrics)

    @property
    def ok(self) -> bool:
        return self.status == "proposed" and self.proposal is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "proposal": None if self.proposal is None else self.proposal.as_dict(),
            "raw_text": self.raw_text,
            "error": self.error,
            "metrics": None if self.metrics is None else self.metrics.as_dict(),
        }


class Planner(Protocol):
    def propose(
        self,
        agent: Agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None = None,
        **generation_options: Any,
    ) -> PlannerResult:
        ...
