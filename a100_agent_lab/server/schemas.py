from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateAgentRequest(BaseModel):
    system_prompt: str | None = None
    workspace_root: str | None = None
    workspace_mode: str = "read_write"
    workspace_metadata: dict[str, Any] = Field(default_factory=dict)
    generation_options: dict[str, Any] = Field(default_factory=dict)


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateProposalRequest(BaseModel):
    instruction: str
    max_tokens: int = 768
    temperature: float = 0.0


class RejectProposalRequest(BaseModel):
    reason: str = "rejected by user"
