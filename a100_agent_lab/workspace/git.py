from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from a100_agent_lab.workspace.workspace import Workspace


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

    def diff(self) -> GitResult:
        return self._run(["diff", "--"])

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
