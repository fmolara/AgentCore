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

from transformers import AutoTokenizer

from a100_agent_lab.generation.config import GenerationConfig
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.logging.events import generation_event
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions.session import Session


class SGLangRuntime(Runtime):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: Path,
        log_writer: JsonlWriter | None = None,
    ):
        self.config = config
        self.project_root = project_root
        self.log_writer = log_writer
        self.process: subprocess.Popen | None = None
        self.server_log = None
        self.server_log_path: Path | None = None
        self.tokenizer = None
        self.ready_sec: float | None = None
        self.api_base = self._api_base()

    def load(self) -> None:
        if self.ready():
            return

        model_cfg = self.config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

        server_cfg = self.config.get("server", {})
        env = os.environ.copy()
        path_prefix = server_cfg.get("path_prefix")
        if path_prefix:
            env["PATH"] = f"{path_prefix}:{env.get('PATH', '')}"
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.get("gpu", {}).get("device", 0))

        self.server_log_path = self._server_log_path()
        self.server_log = self.server_log_path.open("a", encoding="utf-8")
        cmd = self._launch_command()
        start = time.perf_counter()
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            env=env,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        timeout = float(server_cfg.get("startup_timeout_sec", 180))
        self._wait_ready(timeout)
        self.ready_sec = time.perf_counter() - start

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.process = None
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

    def ready(self) -> bool:
        if self.process is not None and self.process.poll() is not None:
            return False
        try:
            self._get("/v1/models", timeout=2)
            return True
        except Exception:
            return False

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready(),
            "runtime": "sglang",
            "api_base": self.api_base,
            "ready_sec": self.ready_sec,
            "server_log_path": str(self.server_log_path) if self.server_log_path else None,
            "process_pid": self.process.pid if self.process else None,
        }

    def warmup(self) -> GenerationResult:
        session = self.create_session()
        return self.generate(session, "Say ready.", max_tokens=4)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        if not self.ready():
            raise RuntimeError("SGLang server is not ready")

        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        session.add_user_message(prompt)
        messages = session.transcript()
        prompt_tokens = self.tokenize(messages, generation=generation)

        payload = {
            "model": self.config.get("model", {}).get("name", self.config["model"]["path"]),
            "messages": messages,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": generation.enable_thinking},
        }

        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        for chunk in self._stream_chat(payload):
            if chunk and first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk)
        end = time.perf_counter()

        text = "".join(chunks)
        generated_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        ttft = None if first_token_at is None else first_token_at - start
        decode_sec = None if first_token_at is None else max(end - first_token_at, 1e-9)
        tokens_per_sec = 0.0 if decode_sec is None else generated_tokens / decode_sec
        session.add_assistant_message(text)

        result = GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                ttft_sec=ttft,
                tokens_per_sec=tokens_per_sec,
                wall_sec=end - start,
            ),
        )

        if self.log_writer is not None:
            self.log_writer.write(generation_event("sglang", session, result, self.health()))

        return result

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[str]:
        result = self.generate(session, prompt, **kwargs)
        yield result.text

    def tokenize(
        self,
        text_or_messages: Any,
        *,
        generation: GenerationConfig | None = None,
    ) -> int:
        if self.tokenizer is None:
            raise RuntimeError("runtime is not loaded")
        if isinstance(text_or_messages, list):
            if generation is None:
                generation = GenerationConfig.from_dict(self.config.get("generation", {}))
            try:
                text = self.tokenizer.apply_chat_template(
                    text_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=generation.enable_thinking,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    text_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(self.tokenizer.encode(str(text_or_messages), add_special_tokens=False))

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def _launch_command(self) -> list[str]:
        server_cfg = self.config.get("server", {})
        model_cfg = self.config["model"]
        context_cfg = self.config.get("context", {})
        gpu_cfg = self.config.get("gpu", {})

        cmd = [
            server_cfg.get("python", "/home/sglang/venv/bin/python"),
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_cfg["path"],
            "--host",
            str(server_cfg.get("host", "127.0.0.1")),
            "--port",
            str(server_cfg.get("port", 31000)),
            "--dtype",
            str(model_cfg.get("dtype", "bfloat16")),
            "--context-length",
            str(context_cfg.get("max_context_tokens", 4096)),
            "--mem-fraction-static",
            str(gpu_cfg.get("memory_fraction", 0.80)),
            "--log-level",
            str(server_cfg.get("log_level", "info")),
        ]
        if model_cfg.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")
        reasoning_parser = server_cfg.get("reasoning_parser")
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", str(reasoning_parser)])
        model_name = model_cfg.get("name")
        if model_name:
            cmd.extend(["--served-model-name", str(model_name)])
        return cmd

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"SGLang server exited with code {self.process.returncode}")
            try:
                self._get("/v1/models", timeout=2)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(f"SGLang server did not become ready within {timeout}s: {last_error}")

    def _stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
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
                    obj = json.loads(data)
                    choice = obj.get("choices", [{}])[0]
                    delta = choice.get("delta") or {}
                    yield delta.get("content") or ""
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"SGLang HTTP {exc.code}: {body[:1000]}") from exc

    def _get(self, path: str, *, timeout: float) -> str:
        with urllib.request.urlopen(f"{self.api_base}{path}", timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")

    def _api_base(self) -> str:
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 31000)
        return f"http://{host}:{port}"

    def _server_log_path(self) -> Path:
        log_cfg = self.config.get("logging", {})
        log_dir = Path(log_cfg.get("path", "experiments/logs"))
        if not log_dir.is_absolute():
            log_dir = self.project_root / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return log_dir / f"sglang-server-{stamp}.log"

