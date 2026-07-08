from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from a100_agent_lab.executor.actions import (
    Action,
    CreateCheckpointAction,
    GitDiffAction,
    GitStatusAction,
    ReadFileAction,
    ReplaceTextAction,
    TaskReportAction,
    WriteFileAction,
)


@dataclass(frozen=True)
class ActionPlan:
    title: str
    description: str = ""
    actions: tuple[Action, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionPlan":
        if not isinstance(data, dict):
            raise ValueError("action plan must be a mapping")
        title = _required_str(data, "title")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise ValueError("action plan field 'description' must be a string")
        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list):
            raise ValueError("action plan field 'actions' must be a list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("action plan field 'metadata' must be a mapping")
        plan_id = data.get("id") or uuid4().hex
        if not isinstance(plan_id, str):
            raise ValueError("action plan field 'id' must be a string")
        return cls(
            id=plan_id,
            title=title,
            description=description,
            actions=tuple(action_from_dict(action) for action in raw_actions),
            metadata=deepcopy(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActionPlan":
        plan_path = Path(path)
        text = plan_path.read_text(encoding="utf-8")
        if plan_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "actions": [action_to_dict(action) for action in self.actions],
            "metadata": deepcopy(self.metadata),
        }

    def save(self, path: str | Path) -> None:
        plan_path = Path(path)
        data = self.as_dict()
        if plan_path.suffix.lower() in {".yaml", ".yml"}:
            plan_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            plan_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_from_dict(data: dict[str, Any]) -> Action:
    if not isinstance(data, dict):
        raise ValueError("action must be a mapping")
    action_type = _required_str(data, "type")
    if action_type == "read_file":
        _reject_unknown_fields(data, {"type", "id", "path"})
        return ReadFileAction(path=_required_str(data, "path"), id=_optional_id(data))
    if action_type == "write_file":
        _reject_unknown_fields(data, {"type", "id", "path", "content"})
        return WriteFileAction(
            path=_required_str(data, "path"),
            content=_required_str(data, "content"),
            id=_optional_id(data),
        )
    if action_type == "replace_text":
        _reject_unknown_fields(data, {"type", "id", "path", "old", "new", "count"})
        count = data.get("count", -1)
        if not isinstance(count, int):
            raise ValueError("replace_text field 'count' must be an integer")
        return ReplaceTextAction(
            path=_required_str(data, "path"),
            old=_required_str(data, "old"),
            new=_required_str(data, "new"),
            count=count,
            id=_optional_id(data),
        )
    if action_type == "create_checkpoint":
        _reject_unknown_fields(data, {"type", "id", "label", "description", "metadata"})
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("create_checkpoint field 'metadata' must be a mapping")
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("create_checkpoint field 'description' must be a string")
        return CreateCheckpointAction(
            label=_required_str(data, "label"),
            description=description,
            metadata=deepcopy(metadata),
            id=_optional_id(data),
        )
    if action_type == "git_status":
        _reject_unknown_fields(data, {"type", "id"})
        return GitStatusAction(id=_optional_id(data))
    if action_type == "git_diff":
        _reject_unknown_fields(data, {"type", "id"})
        return GitDiffAction(id=_optional_id(data))
    if action_type == "task_report":
        _reject_unknown_fields(data, {"type", "id"})
        return TaskReportAction(id=_optional_id(data))
    raise ValueError(f"unknown action type: {action_type}")


def action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, ReadFileAction):
        return {"type": action.action_type, "id": action.id, "path": str(action.path)}
    if isinstance(action, WriteFileAction):
        return {
            "type": action.action_type,
            "id": action.id,
            "path": str(action.path),
            "content": action.content,
        }
    if isinstance(action, ReplaceTextAction):
        return {
            "type": action.action_type,
            "id": action.id,
            "path": str(action.path),
            "old": action.old,
            "new": action.new,
            "count": action.count,
        }
    if isinstance(action, CreateCheckpointAction):
        data = {
            "type": action.action_type,
            "id": action.id,
            "label": action.label,
            "description": action.description,
        }
        if action.metadata is not None:
            data["metadata"] = deepcopy(action.metadata)
        return data
    if isinstance(action, (GitStatusAction, GitDiffAction, TaskReportAction)):
        return {"type": action.action_type, "id": action.id}
    raise ValueError(f"unsupported action instance: {type(action).__name__}")


def _required_str(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required field '{field_name}' must be a non-empty string")
    return value


def _optional_id(data: dict[str, Any]) -> str:
    value = data.get("id")
    if value is None:
        return uuid4().hex
    if not isinstance(value, str) or not value:
        raise ValueError("action field 'id' must be a non-empty string")
    return value


def _reject_unknown_fields(data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("unknown action field(s): " + ", ".join(unknown))
