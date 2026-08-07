from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreateAgentRequest:
    system_prompt: str | None = None
    workspace_root: str | None = None
    workspace_mode: str = "read_write"
    workspace_metadata: dict[str, Any] = field(default_factory=dict)
    generation_options: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "workspace_root": self.workspace_root,
            "workspace_mode": self.workspace_mode,
            "workspace_metadata": deepcopy(self.workspace_metadata),
            "generation_options": deepcopy(self.generation_options),
        }


@dataclass(frozen=True)
class CreateTaskRequest:
    title: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class CreateProposalRequest:
    instruction: str
    max_tokens: int = 768
    temperature: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class RejectProposalRequest:
    reason: str = "rejected by user"

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason}


@dataclass(frozen=True)
class CancelTaskRequest:
    reason: str = "cancel requested"

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason}
