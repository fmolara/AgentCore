from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
from threading import Thread
from time import monotonic
from typing import Any


_INHERITED_ENV_NAMES = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "CC",
    "CXX",
    "CPPFLAGS",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "MAKEFLAGS",
    "PKG_CONFIG_PATH",
)


@dataclass(frozen=True)
class CheckDefinition:
    argv: tuple[str, ...]
    timeout_sec: float = 60.0
    max_output_bytes: int = 64 * 1024
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "CheckDefinition":
        if not isinstance(data, dict):
            raise ValueError(f"workspace check '{name}' must be a mapping")
        unknown = sorted(set(data) - {"argv", "timeout_sec", "max_output_bytes", "env"})
        if unknown:
            raise ValueError(
                f"unknown workspace check field(s) for '{name}': " + ", ".join(unknown)
            )
        argv = data.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError(f"workspace check '{name}' argv must be a non-empty string list")
        timeout_sec = data.get("timeout_sec", 60.0)
        if (
            not isinstance(timeout_sec, (int, float))
            or isinstance(timeout_sec, bool)
            or timeout_sec <= 0
        ):
            raise ValueError(f"workspace check '{name}' timeout_sec must be positive")
        max_output_bytes = data.get("max_output_bytes", 64 * 1024)
        if (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes <= 0
        ):
            raise ValueError(f"workspace check '{name}' max_output_bytes must be positive")
        env = data.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError(f"workspace check '{name}' env must be a string mapping")
        return cls(
            argv=tuple(argv),
            timeout_sec=float(timeout_sec),
            max_output_bytes=max_output_bytes,
            env=deepcopy(env),
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    argv: tuple[str, ...]
    status: str
    returncode: int | None
    wall_sec: float
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "exited" and self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "status": self.status,
            "returncode": self.returncode,
            "wall_sec": self.wall_sec,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "error": self.error,
        }


class CheckExecutionError(RuntimeError):
    def __init__(self, result: CheckResult) -> None:
        if result.status == "timeout":
            message = f"configured check '{result.name}' timed out"
        elif result.status == "launch_failed":
            message = f"configured check '{result.name}' could not start: {result.error}"
        else:
            message = (
                f"configured check '{result.name}' failed with exit code "
                f"{result.returncode}"
            )
        super().__init__(message)
        self.action_data = {"check": result.as_dict()}


class WorkspaceCheckRunner:
    def __init__(
        self,
        root: Path,
        definitions: dict[str, CheckDefinition] | None = None,
    ) -> None:
        self.root = root
        self.definitions = dict(definitions or {})

    @classmethod
    def from_config(
        cls,
        root: Path,
        data: dict[str, Any] | None,
    ) -> "WorkspaceCheckRunner":
        if data is None:
            return cls(root)
        if not isinstance(data, dict):
            raise ValueError("workspace checks configuration must be a mapping")
        return cls(
            root,
            {
                name: CheckDefinition.from_dict(name, definition)
                for name, definition in data.items()
            },
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.definitions))

    def run(self, name: str) -> CheckResult:
        try:
            definition = self.definitions[name]
        except KeyError as exc:
            raise ValueError(f"unknown configured check: {name}") from exc

        environment = {
            key: value
            for key in _INHERITED_ENV_NAMES
            if (value := os.environ.get(key)) is not None
        }
        environment.update(definition.env)
        started = monotonic()
        use_process_group = os.name == "posix"
        try:
            process = subprocess.Popen(
                list(definition.argv),
                cwd=self.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=use_process_group,
            )
        except OSError as exc:
            return CheckResult(
                name=name,
                argv=definition.argv,
                status="launch_failed",
                returncode=None,
                wall_sec=monotonic() - started,
                stdout="",
                stderr="",
                error=str(exc),
            )

        timed_out = False
        assert process.stdout is not None and process.stderr is not None
        stdout_collector = _OutputCollector(process.stdout, definition.max_output_bytes)
        stderr_collector = _OutputCollector(process.stderr, definition.max_output_bytes)
        stdout_collector.start()
        stderr_collector.start()
        try:
            process.wait(timeout=definition.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(process, use_process_group=use_process_group)
            process.wait()
        stdout_collector.join()
        stderr_collector.join()
        return CheckResult(
            name=name,
            argv=definition.argv,
            status="timeout" if timed_out else "exited",
            returncode=process.returncode,
            wall_sec=monotonic() - started,
            stdout=stdout_collector.text,
            stderr=stderr_collector.text,
            timed_out=timed_out,
            stdout_truncated=stdout_collector.truncated,
            stderr_truncated=stderr_collector.truncated,
        )


def _terminate_process(process: subprocess.Popen[bytes], *, use_process_group: bool) -> None:
    if use_process_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1.0)
                return
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


class _OutputCollector:
    def __init__(self, stream, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.thread = Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join()

    @property
    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")

    def _read(self) -> None:
        while True:
            chunk = self.stream.read(8192)
            if not chunk:
                return
            remaining = self.limit - len(self.data)
            if remaining > 0:
                self.data.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                self.truncated = True
