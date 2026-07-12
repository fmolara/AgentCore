from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentcore_server.workspace.workspace import Workspace


@dataclass(frozen=True)
class FileEditResult:
    operation: str
    path: str | None
    bytes_written: int = 0
    lines_written: int = 0
    replacements: int = 0
    files_changed: tuple[str, ...] = ()


class FileWorkspace:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return self.workspace.read_text(path, encoding=encoding)

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> FileEditResult:
        self.workspace._require_writable()
        resolved = self.workspace._resolve(path)
        written = resolved.write_text(content, encoding=encoding)
        return FileEditResult(
            operation="write_text",
            path=self._display_path(resolved),
            bytes_written=written,
            lines_written=len(content.splitlines()),
            files_changed=(self._display_path(resolved),),
        )

    def append_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> FileEditResult:
        self.workspace._require_writable()
        resolved = self.workspace._resolve(path)
        with resolved.open("a", encoding=encoding) as f:
            written = f.write(content)
        return FileEditResult(
            operation="append_text",
            path=self._display_path(resolved),
            bytes_written=written,
            lines_written=len(content.splitlines()),
            files_changed=(self._display_path(resolved),),
        )

    def replace_text(
        self,
        path: str | Path,
        old: str,
        new: str,
        *,
        count: int = -1,
        encoding: str = "utf-8",
    ) -> FileEditResult:
        self.workspace._require_writable()
        if old == "":
            raise ValueError("old text must not be empty")
        resolved = self.workspace._resolve(path)
        original = resolved.read_text(encoding=encoding)
        replacements = original.count(old) if count < 0 else min(original.count(old), count)
        if replacements == 0:
            raise ValueError(f"text not found in {self._display_path(resolved)}")
        updated = original.replace(old, new, count)
        written = resolved.write_text(updated, encoding=encoding)
        return FileEditResult(
            operation="replace_text",
            path=self._display_path(resolved),
            bytes_written=written,
            lines_written=len(updated.splitlines()),
            replacements=replacements,
            files_changed=(self._display_path(resolved),),
        )

    def read_lines(
        self,
        path: str | Path,
        *,
        start: int | None = None,
        end: int | None = None,
        encoding: str = "utf-8",
    ) -> list[str]:
        lines = self.workspace.read_text(path, encoding=encoding).splitlines()
        return lines[start:end]

    def write_lines(
        self,
        path: str | Path,
        lines: list[str],
        *,
        encoding: str = "utf-8",
    ) -> FileEditResult:
        content = "\n".join(lines)
        if lines:
            content += "\n"
        result = self.write_text(path, content, encoding=encoding)
        return FileEditResult(
            operation="write_lines",
            path=result.path,
            bytes_written=result.bytes_written,
            lines_written=len(lines),
            files_changed=result.files_changed,
        )

    def apply_unified_diff(self, diff_text: str, *, encoding: str = "utf-8") -> FileEditResult:
        self.workspace._require_writable()
        patches = _parse_unified_diff(diff_text)
        if not patches:
            raise ValueError("diff contains no file patches")

        files_changed: list[str] = []
        total_bytes = 0
        total_lines = 0
        for patch in patches:
            resolved = self.workspace._resolve(patch.path)
            original = resolved.read_text(encoding=encoding).splitlines(keepends=True)
            updated, changed_lines = _apply_patch(original, patch)
            content = "".join(updated)
            written = resolved.write_text(content, encoding=encoding)
            files_changed.append(self._display_path(resolved))
            total_bytes += written
            total_lines += changed_lines

        return FileEditResult(
            operation="apply_unified_diff",
            path=None,
            bytes_written=total_bytes,
            lines_written=total_lines,
            files_changed=tuple(files_changed),
        )

    def _display_path(self, path: Path) -> str:
        return path.relative_to(self.workspace.root).as_posix()


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Patch:
    path: str
    hunks: tuple[_Hunk, ...]


_HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


def _parse_unified_diff(diff_text: str) -> list[_Patch]:
    lines = diff_text.splitlines(keepends=True)
    patches: list[_Patch] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("invalid unified diff: missing +++ header")
        path = _clean_diff_path(lines[index][4:].strip())
        index += 1
        hunks: list[_Hunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            header = lines[index]
            if not header.startswith("@@ "):
                index += 1
                continue
            match = _HUNK_RE.match(header)
            if not match:
                raise ValueError(f"invalid unified diff hunk header: {header.strip()}")
            index += 1
            hunk_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("@@ ") and not lines[index].startswith("--- "):
                line = lines[index]
                if line.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not line.startswith((" ", "+", "-")):
                    raise ValueError(f"invalid unified diff line: {line.rstrip()}")
                hunk_lines.append(line)
                index += 1
            hunks.append(
                _Hunk(
                    old_start=int(match.group("old_start")),
                    old_count=int(match.group("old_count") or 1),
                    new_start=int(match.group("new_start")),
                    new_count=int(match.group("new_count") or 1),
                    lines=tuple(hunk_lines),
                )
            )
        patches.append(_Patch(path=path, hunks=tuple(hunks)))
    return patches


def _clean_diff_path(raw_path: str) -> str:
    path = raw_path.split("\t", 1)[0]
    if path == "/dev/null":
        raise ValueError("creating or deleting files through unified diff is not supported yet")
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def _apply_patch(original: list[str], patch: _Patch) -> tuple[list[str], int]:
    output: list[str] = []
    cursor = 0
    changed_lines = 0
    for hunk in patch.hunks:
        hunk_start = max(hunk.old_start - 1, 0)
        if hunk_start < cursor:
            raise ValueError(f"overlapping hunks for {patch.path}")
        output.extend(original[cursor:hunk_start])
        cursor = hunk_start
        for line in hunk.lines:
            marker = line[0]
            content = line[1:]
            if marker == " ":
                if cursor >= len(original) or original[cursor] != content:
                    raise ValueError(f"diff context does not match {patch.path}")
                output.append(original[cursor])
                cursor += 1
            elif marker == "-":
                if cursor >= len(original) or original[cursor] != content:
                    raise ValueError(f"diff removal does not match {patch.path}")
                cursor += 1
                changed_lines += 1
            elif marker == "+":
                output.append(content)
                changed_lines += 1
        if hunk.new_count != sum(1 for line in hunk.lines if line.startswith((" ", "+"))):
            raise ValueError(f"diff added line count mismatch for {patch.path}")
        if hunk.old_count != sum(1 for line in hunk.lines if line.startswith((" ", "-"))):
            raise ValueError(f"diff removed line count mismatch for {patch.path}")
    output.extend(original[cursor:])
    return output, changed_lines
