from __future__ import annotations

import fnmatch
import heapq
import json
import os
from pathlib import Path
from typing import Iterable

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
        root = self.workspace._resolve(action.path)
        if not root.exists():
            return ExplorationObservation.failed(action, error="path does not exist")
        if not root.is_dir():
            return ExplorationObservation.failed(action, error="path is not a directory")

        entries: list[dict[str, object]] = []
        truncated = False
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            directory, depth = stack.pop()
            remaining = self.limits.max_files_returned - len(entries)
            try:
                children = heapq.nsmallest(
                    remaining + 1,
                    (
                        child
                        for child in directory.iterdir()
                        if child.name != ".git"
                        and (action.include_hidden or not child.name.startswith("."))
                    ),
                    key=lambda path: path.name,
                )
            except OSError as exc:
                return ExplorationObservation.failed(
                    action,
                    error=f"unable to list directory: {exc}",
                    data={"path": action.path, "entries": entries},
                )
            if len(children) > remaining:
                children = children[:remaining]
                truncated = True
            descend: list[Path] = []
            for child in children:
                relative = child.relative_to(self.workspace.root).as_posix()
                is_symlink = child.is_symlink()
                if is_symlink:
                    kind = "symlink"
                    try:
                        target = child.resolve(strict=True)
                        inside = target == self.workspace.root or self.workspace.root in target.parents
                    except OSError:
                        inside = False
                    entry = {"path": relative, "kind": kind, "target_inside_workspace": inside}
                elif child.is_dir():
                    entry = {"path": relative, "kind": "directory"}
                    if depth < action.max_depth:
                        descend.append(child)
                else:
                    entry = {"path": relative, "kind": "file"}
                entries.append(entry)
                if len(entries) >= self.limits.max_files_returned:
                    truncated = True
                    break
            if truncated:
                break
            for child in reversed(descend):
                stack.append((child, depth + 1))

        return ExplorationObservation.ok(
            action,
            data={
                "path": action.path,
                "entries": entries,
                "max_depth": action.max_depth,
                "include_hidden": action.include_hidden,
                "skipped_git": True,
            },
            truncated=truncated,
        )

    def _search_files(self, action: SearchFilesAction) -> ExplorationObservation:
        root = self.workspace._resolve(action.root)
        if not root.exists():
            return ExplorationObservation.failed(action, error="root does not exist")
        if not root.is_dir():
            return ExplorationObservation.failed(action, error="root is not a directory")

        matches: list[dict[str, object]] = []
        skipped_binary: list[str] = []
        skipped_encoding: list[str] = []
        skipped_symlink: list[str] = []
        truncated = False
        for path in self._iter_search_files(root, skipped_symlink):
            relative = path.relative_to(self.workspace.root).as_posix()
            if not fnmatch.fnmatch(path.name, action.name_pattern):
                continue
            if action.content_query is not None:
                status, found, was_truncated = self._contains_text(path, action.content_query)
                if status == "binary":
                    skipped_binary.append(relative)
                    continue
                if status == "encoding":
                    skipped_encoding.append(relative)
                    continue
                if not found:
                    continue
            else:
                was_truncated = False
            matches.append({"path": relative, "content_scan_truncated": was_truncated})
            if len(matches) >= action.max_results:
                truncated = True
                break

        return ExplorationObservation.ok(
            action,
            data={
                "root": action.root,
                "name_pattern": action.name_pattern,
                "content_query": action.content_query,
                "matches": matches,
                "skipped_binary": skipped_binary,
                "skipped_encoding": skipped_encoding,
                "skipped_symlink": skipped_symlink,
                "pattern_semantics": "glob against basename",
                "hidden_files": "excluded",
                "max_directory_depth": self.limits.max_directory_depth,
            },
            truncated=truncated,
        )

    def _iter_search_files(
        self,
        root: Path,
        skipped_symlink: list[str],
    ) -> Iterable[Path]:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(directory)
            relative_parts = current.relative_to(root).parts
            if len(relative_parts) > self.limits.max_directory_depth:
                dirnames[:] = []
                continue
            retained_dirs: list[str] = []
            for name in sorted(dirnames):
                path = current / name
                if name == ".git" or name.startswith("."):
                    continue
                if path.is_symlink():
                    skipped_symlink.append(
                        path.relative_to(self.workspace.root).as_posix()
                    )
                    continue
                retained_dirs.append(name)
            dirnames[:] = retained_dirs
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = current / filename
                if path.is_symlink():
                    skipped_symlink.append(
                        path.relative_to(self.workspace.root).as_posix()
                    )
                    continue
                yield path

    def _contains_text(self, path: Path, query: str) -> tuple[str, bool, bool]:
        limit = self.limits.max_single_file_bytes
        try:
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
        except OSError:
            return "encoding", False, False
        truncated = len(raw) > limit
        raw = raw[:limit]
        if b"\0" in raw:
            return "binary", False, truncated
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "encoding", False, truncated
        return "text", query in text, truncated

    def _read_file(self, action: ExploreReadFileAction) -> ExplorationObservation:
        path = self.workspace._resolve(action.path)
        if not path.exists():
            return ExplorationObservation.failed(action, error="file does not exist")
        if path.is_dir():
            return ExplorationObservation.failed(action, error="read_file target is a directory")
        if not path.is_file():
            return ExplorationObservation.failed(action, error="path is not a regular file")

        byte_limit = min(
            action.max_bytes or self.limits.max_single_file_bytes,
            self.limits.max_single_file_bytes,
            self.limits.max_observation_text_per_action,
        )
        try:
            with path.open("rb") as stream:
                raw = stream.read(byte_limit + 1)
        except OSError as exc:
            return ExplorationObservation.failed(action, error=f"unable to read file: {exc}")
        truncated = len(raw) > byte_limit
        raw = raw[:byte_limit]
        if b"\0" in raw:
            return ExplorationObservation.failed(
                action,
                error="file appears to be binary",
                data={"encoding": None},
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            if truncated and exc.end == len(raw):
                text = raw[: exc.start].decode("utf-8")
            else:
                return ExplorationObservation.failed(
                    action,
                    error="file is not valid UTF-8",
                    data={"encoding": "utf-8"},
                )

        lines = text.splitlines(keepends=True)
        start = action.start_line - 1
        selected = lines[start:]
        if action.max_lines is not None and len(selected) > action.max_lines:
            selected = selected[: action.max_lines]
            truncated = True
        selected_text = "".join(selected)
        encoded = selected_text.encode("utf-8")
        if len(encoded) > self.limits.max_observation_text_per_action:
            encoded = encoded[: self.limits.max_observation_text_per_action]
            selected_text = encoded.decode("utf-8", errors="ignore")
            truncated = True

        return ExplorationObservation.ok(
            action,
            data={
                "path": action.path,
                "start_line": action.start_line,
                "lines_returned": len(selected_text.splitlines()),
                "bytes_returned": len(selected_text.encode("utf-8")),
                "encoding": "utf-8",
                "text": selected_text,
            },
            truncated=truncated,
        )


def _json_size(data: dict[str, object]) -> int:
    return len(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))
