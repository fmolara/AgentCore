from __future__ import annotations

from typing import Protocol

from agentcore_server.executor import PlanProposal


class ApprovalGateway(Protocol):
    def approve(self, proposal: PlanProposal) -> bool:
        ...


class StaticApprovalGateway:
    """Explicit test/non-interactive approval decision."""

    def __init__(self, approved: bool) -> None:
        self.approved = approved

    def approve(self, proposal: PlanProposal) -> bool:
        del proposal
        return self.approved
