from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentcore_protocol.schemas.requests import (
    CancelTaskRequest as ProtocolCancelTaskRequest,
    CreateAgentRequest as ProtocolCreateAgentRequest,
    CreateProposalRequest as ProtocolCreateProposalRequest,
    CreateTaskRequest as ProtocolCreateTaskRequest,
    RejectProposalRequest as ProtocolRejectProposalRequest,
)


class CreateAgentRequest(BaseModel):
    system_prompt: str | None = None
    workspace_root: str | None = None
    workspace_mode: str = "read_write"
    workspace_metadata: dict[str, Any] = Field(default_factory=dict)
    generation_options: dict[str, Any] = Field(default_factory=dict)

    def to_protocol(self) -> ProtocolCreateAgentRequest:
        return ProtocolCreateAgentRequest(
            system_prompt=self.system_prompt,
            workspace_root=self.workspace_root,
            workspace_mode=self.workspace_mode,
            workspace_metadata=self.workspace_metadata,
            generation_options=self.generation_options,
        )


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_protocol(self) -> ProtocolCreateTaskRequest:
        return ProtocolCreateTaskRequest(
            title=self.title,
            description=self.description,
            metadata=self.metadata,
        )


class CreateProposalRequest(BaseModel):
    instruction: str
    max_tokens: int = 768
    temperature: float = 0.0

    def to_protocol(self) -> ProtocolCreateProposalRequest:
        return ProtocolCreateProposalRequest(
            instruction=self.instruction,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class RejectProposalRequest(BaseModel):
    reason: str = "rejected by user"

    def to_protocol(self) -> ProtocolRejectProposalRequest:
        return ProtocolRejectProposalRequest(reason=self.reason)


class CancelTaskRequest(BaseModel):
    reason: str = "cancel requested"

    def to_protocol(self) -> ProtocolCancelTaskRequest:
        return ProtocolCancelTaskRequest(reason=self.reason)


__all__ = [
    "CancelTaskRequest",
    "CreateAgentRequest",
    "CreateProposalRequest",
    "CreateTaskRequest",
    "RejectProposalRequest",
]
