from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class ServerProcess:
    def __init__(
        self,
        *,
        runtime_name: str,
        api_base: str,
        project_root: Path,
        log_dir: Path,
        log_prefix: str,
        ready_path: str = "/v1/models",
    ):
        self.runtime_name = runtime_name
        self.api_base = api_base
        self.project_root = project_root
        self.log_dir = log_dir
        self.log_prefix = log_prefix
        self.ready_path = ready_path
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_path: Path | None = None
        self.ready_sec: float | None = None
        self.last_error: str | None = None

    def start(
        self,
        command: list[str],
        *,
        timeout: float,
        env_updates: dict[str, str] | None = None,
        path_prefix: str | None = None,
    ) -> None:
        env = os.environ.copy()
        if path_prefix:
            env["PATH"] = f"{path_prefix}:{env.get('PATH', '')}"
        if env_updates:
            env.update(env_updates)

        self.log_path = self._new_log_path()
        self.log_file = self.log_path.open("a", encoding="utf-8")

        start = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self.wait_ready(timeout)
            self.ready_sec = time.perf_counter() - start
            self.last_error = None
        except Exception as exc:
            self.last_error = repr(exc)
            raise

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def ready(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            return False
        try:
            self.get(self.ready_path, timeout=2)
            return True
        except Exception:
            return False

    def wait_ready(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"{self.runtime_name} server exited with code {self.process.returncode}")
            try:
                self.get(self.ready_path, timeout=2)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(f"{self.runtime_name} server did not become ready within {timeout}s: {last_error}")

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready(),
            "runtime": self.runtime_name,
            "api_base": self.api_base,
            "ready_sec": self.ready_sec,
            "server_log_path": str(self.log_path) if self.log_path else None,
            "process_pid": self.process.pid if self.process else None,
            "last_error": self.last_error,
        }

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        for obj in self.stream_chat_events(payload):
            choice = obj.get("choices", [{}])[0] if obj.get("choices") else {}
            delta = choice.get("delta") or {}
            yield delta.get("content") or ""

    def stream_chat_events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.api_base}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            self.last_error = f"HTTP {exc.code}: {body[:1000]}"
            raise RuntimeError(f"{self.runtime_name} HTTP {exc.code}: {body[:1000]}") from exc

    def get(self, path: str, *, timeout: float) -> str:
        with urllib.request.urlopen(f"{self.api_base}{path}", timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")

    def _new_log_path(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.log_dir / f"{self.log_prefix}-{stamp}.log"
