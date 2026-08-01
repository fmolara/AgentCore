from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import heapq
import os
from pathlib import Path
from typing import Any, Iterable

from agentcore_server.workspace.workspace import Workspace


@dataclass(frozen=True)
class DiscoveryLimits:
    max_directory_depth: int = 4
    max_files_returned: int = 100
    max_search_files_scanned: int = 1000
    max_search_bytes: int = 4 * 1024 * 1024
    max_single_file_bytes: int = 64 * 1024
    max_read_lines: int = 500


@dataclass(frozen=True)
class DiscoveryResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            **self.data,
            "error": self.error,
            "truncated": self.truncated,
        }


class WorkspaceDiscovery:
    """Bounded deterministic workspace reads shared by planners and tool agents."""

    def __init__(self, workspace: Workspace, *, limits: DiscoveryLimits | None = None) -> None:
        self.workspace = workspace
        self.limits = limits or DiscoveryLimits()

    def list_directory(
        self,
        path: str = ".",
        *,
        max_depth: int = 1,
        include_hidden: bool = False,
    ) -> DiscoveryResult:
        if max_depth < 0 or max_depth > self.limits.max_directory_depth:
            raise ValueError(
                f"max_depth must be between 0 and {self.limits.max_directory_depth}"
            )
        root = self.workspace._resolve(path)
        if not root.exists():
            return DiscoveryResult(False, {"path": path, "entries": []}, "path does not exist")
        if not root.is_dir():
            return DiscoveryResult(False, {"path": path, "entries": []}, "path is not a directory")

        entries: list[dict[str, Any]] = []
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
                        and (include_hidden or not child.name.startswith("."))
                    ),
                    key=lambda item: item.name,
                )
            except OSError as exc:
                return DiscoveryResult(
                    False,
                    {"path": path, "entries": entries},
                    f"unable to list directory: {exc}",
                )
            if len(children) > remaining:
                children = children[:remaining]
                truncated = True
            descend: list[Path] = []
            for child in children:
                relative = child.relative_to(self.workspace.root).as_posix()
                if child.is_symlink():
                    try:
                        target = child.resolve(strict=True)
                        inside = target == self.workspace.root or self.workspace.root in target.parents
                    except OSError:
                        inside = False
                    entry = {
                        "path": relative,
                        "kind": "symlink",
                        "target_inside_workspace": inside,
                    }
                elif child.is_dir():
                    entry = {"path": relative, "kind": "directory"}
                    if depth < max_depth:
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
        return DiscoveryResult(
            True,
            {
                "path": path,
                "entries": entries,
                "max_depth": max_depth,
                "include_hidden": include_hidden,
                "skipped_git": True,
            },
            truncated=truncated,
        )

    def search_files(
        self,
        root: str = ".",
        *,
        name_pattern: str = "*",
        content_query: str | None = None,
        max_results: int = 50,
    ) -> DiscoveryResult:
        if not name_pattern:
            raise ValueError("name_pattern must not be empty")
        if max_results <= 0 or max_results > self.limits.max_files_returned:
            raise ValueError(
                f"max_results must be between 1 and {self.limits.max_files_returned}"
            )
        root_path = self.workspace._resolve(root)
        if not root_path.exists():
            return DiscoveryResult(False, {"root": root, "matches": []}, "root does not exist")
        if not root_path.is_dir():
            return DiscoveryResult(False, {"root": root, "matches": []}, "root is not a directory")

        matches: list[dict[str, Any]] = []
        skipped_binary: list[str] = []
        skipped_encoding: list[str] = []
        skipped_symlink: list[str] = []
        reasons: set[str] = set()
        scanned = 0
        scanned_bytes = 0
        truncated = False
        for candidate in self._iter_files(root_path, skipped_symlink):
            if scanned >= self.limits.max_search_files_scanned:
                reasons.add("max_search_files_scanned")
                truncated = True
                break
            scanned += 1
            relative = candidate.relative_to(self.workspace.root).as_posix()
            if not fnmatch.fnmatch(candidate.name, name_pattern):
                continue
            content_truncated = False
            if content_query is not None:
                remaining = self.limits.max_search_bytes - scanned_bytes
                if remaining <= 0:
                    reasons.add("max_search_bytes")
                    truncated = True
                    break
                status, found, content_truncated, read_bytes = self._contains_text(
                    candidate, content_query, max_bytes=remaining
                )
                scanned_bytes += read_bytes
                if status == "binary":
                    skipped_binary.append(relative)
                    continue
                if status == "encoding":
                    skipped_encoding.append(relative)
                    continue
                if not found:
                    if scanned_bytes >= self.limits.max_search_bytes:
                        reasons.add("max_search_bytes")
                        truncated = True
                        break
                    continue
            matches.append({"path": relative, "content_scan_truncated": content_truncated})
            if len(matches) >= max_results:
                reasons.add("max_results")
                truncated = True
                break
            if content_query is not None and scanned_bytes >= self.limits.max_search_bytes:
                reasons.add("max_search_bytes")
                truncated = True
                break
        return DiscoveryResult(
            True,
            {
                "root": root,
                "name_pattern": name_pattern,
                "content_query": content_query,
                "matches": matches,
                "pattern_semantics": "glob against basename",
                "files_scanned": scanned,
                "content_bytes_scanned": scanned_bytes,
                "skipped_binary": skipped_binary,
                "skipped_encoding": skipped_encoding,
                "skipped_symlink": skipped_symlink,
                "truncation_reasons": sorted(reasons),
            },
            truncated=truncated,
        )

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = 120,
        max_bytes: int | None = None,
    ) -> DiscoveryResult:
        if start_line <= 0:
            raise ValueError("start_line must be positive")
        if max_lines <= 0 or max_lines > self.limits.max_read_lines:
            raise ValueError(f"max_lines must be between 1 and {self.limits.max_read_lines}")
        resolved = self.workspace._resolve(path)
        if not resolved.exists():
            return DiscoveryResult(False, {"path": path}, "file does not exist")
        if resolved.is_dir():
            return DiscoveryResult(False, {"path": path}, "read_file target is a directory")
        if not resolved.is_file():
            return DiscoveryResult(False, {"path": path}, "path is not a regular file")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        limit = min(max_bytes or self.limits.max_single_file_bytes, self.limits.max_single_file_bytes)
        try:
            with resolved.open("rb") as stream:
                raw = stream.read(limit + 1)
        except OSError as exc:
            return DiscoveryResult(False, {"path": path}, f"unable to read file: {exc}")
        byte_truncated = len(raw) > limit
        raw = raw[:limit]
        if b"\0" in raw:
            return DiscoveryResult(False, {"path": path}, "file appears to be binary")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            if byte_truncated and exc.end == len(raw):
                text = raw[:exc.start].decode("utf-8")
            else:
                return DiscoveryResult(False, {"path": path}, "file is not valid UTF-8")
        lines = text.splitlines(keepends=True)
        start = start_line - 1
        selected = lines[start : start + max_lines]
        line_truncated = start + len(selected) < len(lines) or byte_truncated
        end_line = start_line + len(selected) - 1 if selected else start_line - 1
        total_lines = None if byte_truncated else len(lines)
        return DiscoveryResult(
            True,
            {
                "path": resolved.relative_to(self.workspace.root).as_posix(),
                "start_line": start_line,
                "end_line": end_line,
                "lines_returned": len(selected),
                "total_lines": total_lines,
                "continuation_start_line": end_line + 1 if line_truncated else None,
                "bytes_returned": len("".join(selected).encode("utf-8")),
                "encoding": "utf-8",
                "content": "".join(selected),
            },
            truncated=line_truncated,
        )

    def _iter_files(self, root: Path, skipped_symlink: list[str]) -> Iterable[Path]:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(directory)
            if len(current.relative_to(root).parts) > self.limits.max_directory_depth:
                dirnames[:] = []
                continue
            retained: list[str] = []
            for name in sorted(dirnames):
                candidate = current / name
                if name == ".git" or name.startswith("."):
                    continue
                if candidate.is_symlink():
                    skipped_symlink.append(candidate.relative_to(self.workspace.root).as_posix())
                    continue
                retained.append(name)
            dirnames[:] = retained
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                candidate = current / name
                if candidate.is_symlink():
                    skipped_symlink.append(candidate.relative_to(self.workspace.root).as_posix())
                    continue
                yield candidate

    def _contains_text(
        self, path: Path, query: str, *, max_bytes: int
    ) -> tuple[str, bool, bool, int]:
        limit = min(self.limits.max_single_file_bytes, max_bytes)
        try:
            with path.open("rb") as stream:
                raw = stream.read(limit + 1)
        except OSError:
            return "encoding", False, False, 0
        truncated = len(raw) > limit
        raw = raw[:limit]
        if b"\0" in raw:
            return "binary", False, truncated, len(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "encoding", False, truncated, len(raw)
        return "text", query in text, truncated, len(raw)
