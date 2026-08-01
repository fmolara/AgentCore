from __future__ import annotations

import json

from agentcore_server.planning.exploration import (
    ExplorationAction,
    ExplorationLimits,
    ExplorationObservation,
    ExplorationPlan,
    ExploreReadFileAction,
    ListDirectoryAction,
    SearchFilesAction,
)
from agentcore_server.workspace import Workspace
from agentcore_server.workspace.discovery import DiscoveryLimits, WorkspaceDiscovery


class ExplorationError(RuntimeError):
    pass


class ExplorationBudgetError(ExplorationError):
    def __init__(
        self,
        message: str,
        *,
        observations: tuple[ExplorationObservation, ...] = (),
    ) -> None:
        super().__init__(message)
        self.observations = observations


class WorkspaceExplorer:
    """Execute validated, bounded discovery without mutating the workspace."""

    def __init__(self, workspace: Workspace, *, limits: ExplorationLimits) -> None:
        self.workspace = workspace
        self.limits = limits
        self.total_actions = 0
        self.total_observation_bytes = 0
        self.discovery = WorkspaceDiscovery(
            workspace,
            limits=DiscoveryLimits(
                max_directory_depth=limits.max_directory_depth,
                max_files_returned=limits.max_files_returned,
                max_search_files_scanned=limits.max_search_files_scanned,
                max_search_bytes=limits.max_search_bytes,
                max_single_file_bytes=min(
                    limits.max_single_file_bytes,
                    limits.max_observation_text_per_action,
                ),
                max_read_lines=max(1, limits.max_observation_text_per_action),
            ),
        )

    def execute(self, plan: ExplorationPlan) -> tuple[tuple[ExplorationObservation, ...], int]:
        self.validate(plan)

        observations: list[ExplorationObservation] = []
        round_bytes = 0
        for action in plan.actions:
            observation = self.execute_action(action)
            size = self.observation_size(observation)
            self.validate_observation_budget(
                round_bytes=round_bytes,
                observation_bytes=size,
                observations=observations,
            )
            observations.append(observation)
            round_bytes += size

        self.commit_round(plan, observation_bytes=round_bytes)
        return tuple(observations), round_bytes

    def validate(self, plan: ExplorationPlan) -> None:
        if self.total_actions + len(plan.actions) > self.limits.max_total_actions:
            raise ExplorationBudgetError(
                f"exploration exceeds max_total_actions={self.limits.max_total_actions}"
            )

        # Resolve every path before any action runs so structurally unsafe rounds
        # cannot partially execute.
        for action in plan.actions:
            self._validate_action_path(action)

    def execute_action(self, action: ExplorationAction) -> ExplorationObservation:
        return self._execute_action(action)

    @staticmethod
    def observation_size(observation: ExplorationObservation) -> int:
        return _json_size(observation.as_dict())

    def validate_observation_budget(
        self,
        *,
        round_bytes: int,
        observation_bytes: int,
        observations: list[ExplorationObservation],
    ) -> None:
        if (
            self.total_observation_bytes + round_bytes + observation_bytes
            > self.limits.max_total_observation_bytes
        ):
            raise ExplorationBudgetError(
                "exploration exceeds "
                f"max_total_observation_bytes={self.limits.max_total_observation_bytes}",
                observations=tuple(observations),
            )

    def commit_round(self, plan: ExplorationPlan, *, observation_bytes: int) -> None:
        self.total_actions += len(plan.actions)
        self.total_observation_bytes += observation_bytes

    def _validate_action_path(self, action: ExplorationAction) -> None:
        if isinstance(action, ListDirectoryAction):
            self.workspace._resolve(action.path)
        elif isinstance(action, SearchFilesAction):
            self.workspace._resolve(action.root)
        else:
            self.workspace._resolve(action.path)

    def _execute_action(self, action: ExplorationAction) -> ExplorationObservation:
        if isinstance(action, ListDirectoryAction):
            return self._list_directory(action)
        if isinstance(action, SearchFilesAction):
            return self._search_files(action)
        return self._read_file(action)

    def _list_directory(self, action: ListDirectoryAction) -> ExplorationObservation:
        return self._observation(
            action,
            self.discovery.list_directory(
                action.path,
                max_depth=action.max_depth,
                include_hidden=action.include_hidden,
            ),
        )

    def _search_files(self, action: SearchFilesAction) -> ExplorationObservation:
        return self._observation(
            action,
            self.discovery.search_files(
                action.root,
                name_pattern=action.name_pattern,
                content_query=action.content_query,
                max_results=action.max_results,
            ),
        )

    def _read_file(self, action: ExploreReadFileAction) -> ExplorationObservation:
        result = self.discovery.read_file(
            action.path,
            start_line=action.start_line,
            max_lines=action.max_lines or self.discovery.limits.max_read_lines,
            max_bytes=action.max_bytes,
        )
        data = dict(result.data)
        if "content" in data:
            data["text"] = data.pop("content")
        return self._observation(action, result, data=data)

    @staticmethod
    def _observation(action, result, *, data=None) -> ExplorationObservation:
        payload = dict(result.data) if data is None else data
        if result.success:
            return ExplorationObservation.ok(
                action,
                data=payload,
                truncated=result.truncated,
            )
        return ExplorationObservation.failed(action, error=result.error or "discovery failed", data=payload)


def _json_size(data: dict[str, object]) -> int:
    return len(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))
