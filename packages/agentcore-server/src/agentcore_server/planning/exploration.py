from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class PlanningPhase(str, Enum):
    EXPLORE = "explore"
    FINAL = "final"
    CANNOT_PLAN = "cannot_plan"


@dataclass(frozen=True)
class ExplorationLimits:
    max_rounds: int = 3
    max_actions_per_round: int = 8
    max_total_actions: int = 20
    max_directory_depth: int = 4
    max_files_returned: int = 100
    max_search_files_scanned: int = 1000
    max_search_bytes: int = 4 * 1024 * 1024
    max_single_file_bytes: int = 64 * 1024
    max_total_observation_bytes: int = 256 * 1024
    max_observation_text_per_action: int = 64 * 1024

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "ExplorationLimits":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("planner exploration configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("unknown exploration limit(s): " + ", ".join(unknown))
        values: dict[str, int] = {}
        for name, value in data.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"exploration limit '{name}' must be a positive integer")
            values[name] = value
        return cls(**values)

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ListDirectoryAction:
    path: str = "."
    max_depth: int = 1
    include_hidden: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)
    action_type: str = field(default="list_directory", init=False)


@dataclass(frozen=True)
class SearchFilesAction:
    root: str = "."
    name_pattern: str = "*"
    content_query: str | None = None
    max_results: int = 50
    id: str = field(default_factory=lambda: uuid4().hex)
    action_type: str = field(default="search_files", init=False)


@dataclass(frozen=True)
class ExploreReadFileAction:
    path: str = ""
    start_line: int = 1
    max_lines: int | None = None
    max_bytes: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    action_type: str = field(default="read_file", init=False)


ExplorationAction = ListDirectoryAction | SearchFilesAction | ExploreReadFileAction


@dataclass(frozen=True)
class ExplorationPlan:
    summary: str
    actions: tuple[ExplorationAction, ...]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        limits: ExplorationLimits,
    ) -> "ExplorationPlan":
        _require_exact_fields(data, required={"phase", "summary", "actions"})
        if data.get("phase") != PlanningPhase.EXPLORE.value:
            raise ValueError("exploration response phase must be 'explore'")
        summary = _required_str(data, "summary")
        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("exploration response actions must be a non-empty list")
        if len(raw_actions) > limits.max_actions_per_round:
            raise ValueError(
                f"exploration round exceeds max_actions_per_round={limits.max_actions_per_round}"
            )
        return cls(
            summary=summary,
            actions=tuple(exploration_action_from_dict(item, limits=limits) for item in raw_actions),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": PlanningPhase.EXPLORE.value,
            "summary": self.summary,
            "actions": [exploration_action_to_dict(action) for action in self.actions],
        }


@dataclass(frozen=True)
class ExplorationObservation:
    action_id: str
    action_type: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    truncated: bool = False

    @classmethod
    def ok(
        cls,
        action: ExplorationAction,
        *,
        data: dict[str, Any],
        truncated: bool = False,
    ) -> "ExplorationObservation":
        return cls(
            action_id=action.id,
            action_type=action.action_type,
            status="ok",
            data=deepcopy(data),
            truncated=truncated,
        )

    @classmethod
    def failed(
        cls,
        action: ExplorationAction,
        *,
        error: str,
        data: dict[str, Any] | None = None,
    ) -> "ExplorationObservation":
        return cls(
            action_id=action.id,
            action_type=action.action_type,
            status="failed",
            data=deepcopy(data or {}),
            error=error,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "status": self.status,
            "data": deepcopy(self.data),
            "error": self.error,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ExplorationRound:
    number: int
    plan: ExplorationPlan
    observations: tuple[ExplorationObservation, ...]
    observation_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "plan": self.plan.as_dict(),
            "observations": [observation.as_dict() for observation in self.observations],
            "observation_bytes": self.observation_bytes,
        }


@dataclass(frozen=True)
class PlanningDecision:
    phase: PlanningPhase
    summary: str = ""
    exploration: ExplorationPlan | None = None
    final_plan: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        limits: ExplorationLimits,
    ) -> "PlanningDecision":
        if not isinstance(data, dict):
            raise ValueError("planning response must be a mapping")
        raw_phase = data.get("phase")
        try:
            phase = PlanningPhase(raw_phase)
        except (TypeError, ValueError) as exc:
            raise ValueError("planning response phase must be explore, final, or cannot_plan") from exc
        if phase == PlanningPhase.EXPLORE:
            plan = ExplorationPlan.from_dict(data, limits=limits)
            return cls(phase=phase, summary=plan.summary, exploration=plan)
        if phase == PlanningPhase.FINAL:
            _require_exact_fields(data, required={"phase", "plan"})
            plan = data.get("plan")
            if not isinstance(plan, dict):
                raise ValueError("final planning response field 'plan' must be a mapping")
            return cls(phase=phase, final_plan=deepcopy(plan))
        _require_exact_fields(data, required={"phase", "reason"})
        return cls(phase=phase, reason=_required_str(data, "reason"))


def exploration_action_from_dict(
    data: dict[str, Any],
    *,
    limits: ExplorationLimits,
) -> ExplorationAction:
    if not isinstance(data, dict):
        raise ValueError("exploration action must be a mapping")
    action_type = _required_str(data, "type")
    if action_type == "list_directory":
        _require_exact_fields(
            data,
            required={"type"},
            optional={"id", "path", "max_depth", "include_hidden"},
        )
        path = data.get("path", ".")
        max_depth = data.get("max_depth", 1)
        include_hidden = data.get("include_hidden", False)
        _validate_relative_path(path, "list_directory path")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError("list_directory max_depth must be a non-negative integer")
        if max_depth > limits.max_directory_depth:
            raise ValueError(
                f"list_directory max_depth exceeds limit {limits.max_directory_depth}"
            )
        if not isinstance(include_hidden, bool):
            raise ValueError("list_directory include_hidden must be a boolean")
        return ListDirectoryAction(
            path=path,
            max_depth=max_depth,
            include_hidden=include_hidden,
            id=_optional_id(data),
        )
    if action_type == "search_files":
        _require_exact_fields(
            data,
            required={"type"},
            optional={"id", "root", "name_pattern", "content_query", "max_results"},
        )
        root = data.get("root", ".")
        name_pattern = data.get("name_pattern", "*")
        content_query = data.get("content_query")
        max_results = data.get("max_results", min(50, limits.max_files_returned))
        _validate_relative_path(root, "search_files root")
        if not isinstance(name_pattern, str) or not name_pattern:
            raise ValueError("search_files name_pattern must be a non-empty glob string")
        if content_query is not None and not isinstance(content_query, str):
            raise ValueError("search_files content_query must be a string")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
            raise ValueError("search_files max_results must be a positive integer")
        if max_results > limits.max_files_returned:
            raise ValueError(
                f"search_files max_results exceeds limit {limits.max_files_returned}"
            )
        return SearchFilesAction(
            root=root,
            name_pattern=name_pattern,
            content_query=content_query,
            max_results=max_results,
            id=_optional_id(data),
        )
    if action_type == "read_file":
        _require_exact_fields(
            data,
            required={"type", "path"},
            optional={"id", "start_line", "max_lines", "max_bytes"},
        )
        path = _required_str(data, "path")
        _validate_relative_path(path, "read_file path")
        start_line = data.get("start_line", 1)
        max_lines = data.get("max_lines")
        max_bytes = data.get("max_bytes")
        if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line <= 0:
            raise ValueError("read_file start_line must be a positive integer")
        if max_lines is not None and (
            not isinstance(max_lines, int) or isinstance(max_lines, bool) or max_lines <= 0
        ):
            raise ValueError("read_file max_lines must be a positive integer")
        if max_bytes is not None and (
            not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0
        ):
            raise ValueError("read_file max_bytes must be a positive integer")
        if max_bytes is not None and max_bytes > limits.max_single_file_bytes:
            raise ValueError(
                f"read_file max_bytes exceeds limit {limits.max_single_file_bytes}"
            )
        return ExploreReadFileAction(
            path=path,
            start_line=start_line,
            max_lines=max_lines,
            max_bytes=max_bytes,
            id=_optional_id(data),
        )
    raise ValueError(f"unknown exploration action type: {action_type}")


def exploration_action_to_dict(action: ExplorationAction) -> dict[str, Any]:
    if isinstance(action, ListDirectoryAction):
        return {
            "type": action.action_type,
            "id": action.id,
            "path": action.path,
            "max_depth": action.max_depth,
            "include_hidden": action.include_hidden,
        }
    if isinstance(action, SearchFilesAction):
        data: dict[str, Any] = {
            "type": action.action_type,
            "id": action.id,
            "root": action.root,
            "name_pattern": action.name_pattern,
            "max_results": action.max_results,
        }
        if action.content_query is not None:
            data["content_query"] = action.content_query
        return data
    return {
        "type": action.action_type,
        "id": action.id,
        "path": action.path,
        "start_line": action.start_line,
        "max_lines": action.max_lines,
        "max_bytes": action.max_bytes,
    }


def _required_str(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required field '{name}' must be a non-empty string")
    return value


def _optional_id(data: dict[str, Any]) -> str:
    value = data.get("id")
    if value is None:
        return uuid4().hex
    if not isinstance(value, str) or not value:
        raise ValueError("action field 'id' must be a non-empty string")
    return value


def _require_exact_fields(
    data: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError("missing required field(s): " + ", ".join(missing))
    allowed = required | (optional or set())
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError("unknown field(s): " + ", ".join(unknown))


def _validate_relative_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    from pathlib import PurePath

    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the workspace")
