from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentcore_server.workspace.workspace import Workspace


@dataclass(frozen=True)
class GitResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class GitWorkspace:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def is_repo(self) -> bool:
        result = self._run(["rev-parse", "--show-toplevel"], check=False)
        if not result.ok:
            return False
        try:
            top_level = Path(result.stdout.strip()).resolve()
        except OSError:
            return False
        return top_level == self.workspace.root

    def init(self) -> GitResult:
        self.workspace._require_writable()
        return self._run(["init"])

    def status(self) -> GitResult:
        return self._run(["status", "--short"])

    def diff(self, path: str | Path | None = None) -> GitResult:
        args = ["diff", "--"]
        if path is not None:
            args.append(self._relative_path(path))
        tracked = self._run(args)
        untracked = self._untracked_file_diffs(path)
        if not untracked:
            return tracked
        return GitResult(
            command=tracked.command,
            returncode=tracked.returncode,
            stdout=tracked.stdout + "".join(untracked),
            stderr=tracked.stderr,
        )

    def _untracked_file_diffs(self, path: str | Path | None) -> list[str]:
        status_args = [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
            "--",
        ]
        if path is not None:
            status_args.append(self._relative_path(path))
        status = self._run(status_args)
        diffs: list[str] = []
        for entry in status.stdout.split("\0"):
            if not entry.startswith("?? "):
                continue
            relative = entry[3:]
            # Git condenses untracked directories. Check/build artifacts in those
            # directories are not workspace edits and should not flood the diff.
            if not relative or relative.endswith("/"):
                continue
            resolved = self.workspace._resolve(relative)
            if not resolved.is_file():
                continue
            result = self._run(
                ["diff", "--no-index", "--", "/dev/null", relative],
                check=False,
            )
            if result.returncode not in {0, 1}:
                raise RuntimeError(
                    f"git command failed ({result.returncode}): "
                    f"{' '.join(result.command)}\n{result.stderr}"
                )
            diffs.append(result.stdout)
        return diffs

    def add(self, paths: str | Path | Sequence[str | Path]) -> GitResult:
        self.workspace._require_writable()
        path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not path_list:
            raise ValueError("at least one path is required")
        relative_paths = [self._relative_path(path) for path in path_list]
        return self._run(["add", "--", *relative_paths])

    def commit(self, message: str) -> GitResult:
        self.workspace._require_writable()
        if not message.strip():
            raise ValueError("commit message must not be empty")
        return self._run(["commit", "-m", message])

    def log(self, limit: int = 10) -> GitResult:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        return self._run(["log", f"--max-count={limit}", "--oneline"])

    def current_branch(self) -> str | None:
        result = self._run(["branch", "--show-current"], check=False)
        if not result.ok:
            return None
        branch = result.stdout.strip()
        return branch or None

    def current_commit(self) -> str | None:
        result = self._run(["rev-parse", "HEAD"], check=False)
        if not result.ok:
            return None
        commit = result.stdout.strip()
        return commit or None

    def _relative_path(self, path: str | Path) -> str:
        resolved = self.workspace._resolve(path)
        return resolved.relative_to(self.workspace.root).as_posix()

    def _run(self, args: list[str], *, check: bool = True) -> GitResult:
        command = ["git", *args]
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            text=True,
            capture_output=True,
            check=False,
        )
        result = GitResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and not result.ok:
            raise RuntimeError(
                f"git command failed ({result.returncode}): {' '.join(result.command)}\n{result.stderr}"
            )
        return result
