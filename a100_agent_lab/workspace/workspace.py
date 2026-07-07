from __future__ import annotations

from pathlib import Path
from typing import Any


class Workspace:
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"

    def __init__(
        self,
        root: str | Path,
        *,
        cwd: str | Path | None = None,
        mode: str = READ_WRITE,
        metadata: dict[str, Any] | None = None,
        create: bool = True,
    ):
        if mode not in {self.READ_ONLY, self.READ_WRITE}:
            raise ValueError(f"invalid workspace mode: {mode}")

        self.root = Path(root).expanduser().resolve()
        self.mode = mode
        self.metadata = dict(metadata or {})

        if not self.root.exists():
            if self.read_only or not create:
                raise FileNotFoundError(f"workspace root does not exist: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {self.root}")

        self.cwd = self.root
        self.cwd = self._resolve(cwd or ".")
        if not self.cwd.exists():
            if self.read_only or not create:
                raise FileNotFoundError(f"workspace cwd does not exist: {self.cwd}")
            self.cwd.mkdir(parents=True, exist_ok=True)
        if not self.cwd.is_dir():
            raise NotADirectoryError(f"workspace cwd is not a directory: {self.cwd}")
        from a100_agent_lab.workspace.git import GitWorkspace

        self.git = GitWorkspace(self)

    @property
    def read_only(self) -> bool:
        return self.mode == self.READ_ONLY

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return self._resolve(path).read_text(encoding=encoding)

    def write_text(
        self,
        path: str | Path,
        text: str,
        *,
        encoding: str = "utf-8",
        create_parents: bool = False,
    ) -> int:
        self._require_writable()
        resolved = self._resolve(path)
        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved.write_text(text, encoding=encoding)

    def list(self, path: str | Path = ".") -> list[str]:
        resolved = self._resolve(path)
        return sorted(child.name for child in resolved.iterdir())

    def exists(self, path: str | Path) -> bool:
        return self._resolve(path).exists()

    def mkdir(self, path: str | Path, *, parents: bool = True, exist_ok: bool = True) -> Path:
        self._require_writable()
        resolved = self._resolve(path)
        resolved.mkdir(parents=parents, exist_ok=exist_ok)
        return resolved

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "cwd": str(self.cwd.relative_to(self.root)),
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.cwd / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"path escapes workspace root: {path}")
        return resolved

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("workspace is read-only")
