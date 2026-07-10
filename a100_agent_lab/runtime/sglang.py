from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from a100_agent_lab.generation.config import GenerationConfig
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.generation.stream import StreamChunk
from a100_agent_lab.logging.events import generation_event
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.runtime.health import normalized_health, query_gpu
from a100_agent_lab.runtime.server_process import ServerProcess
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
        self.tokenizer = None
        self.api_base = self._api_base()
        self.last_warmup_wall_sec: float | None = None
        self.server = ServerProcess(
            runtime_name="sglang",
            api_base=self.api_base,
            project_root=project_root,
            log_dir=self._log_dir(),
            log_prefix="sglang-server",
        )

    def load(self) -> None:
        if self.ready():
            return

        model_cfg = self.config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

        server_cfg = self.config.get("server", {})
        self.server.start(
            self._launch_command(),
            timeout=float(server_cfg.get("startup_timeout_sec", 180)),
            env_updates={"CUDA_VISIBLE_DEVICES": str(self.config.get("gpu", {}).get("device", 0))},
            path_prefix=server_cfg.get("path_prefix"),
        )

    def shutdown(self) -> None:
        self.server.shutdown()

    def ready(self) -> bool:
        return self.server.ready()

    def health(self) -> dict[str, Any]:
        server_health = self.server.health()
        return normalized_health(
            runtime_name="sglang",
            backend_type="server",
            model_path=self.config.get("model", {}).get("path"),
            ready=server_health["ready"],
            server_ready_time_sec=server_health["ready_sec"],
            warmup_wall_sec=self.last_warmup_wall_sec,
            process_pid=server_health["process_pid"],
            endpoint=server_health["api_base"],
            last_error=server_health["last_error"],
            gpu=query_gpu(self.config.get("gpu", {}).get("device", 0)),
            extra={"server_log_path": server_health["server_log_path"]},
        )

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        result = self._collect_stream(
            session,
            prompt or "Say ready.",
            event_type="warmup",
            max_tokens=max_tokens,
        )
        self.last_warmup_wall_sec = result.metrics.wall_sec
        return result

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        return self._collect_stream(session, prompt, event_type="generation", **kwargs)

    def _collect_stream(
        self,
        session: Session,
        prompt: str,
        *,
        event_type: str,
        **kwargs: Any,
    ) -> GenerationResult:
        completed: GenerationResult | None = None
        for chunk in self._stream(session, prompt, event_type=event_type, **kwargs):
            if chunk.chunk_type == "failed":
                raise RuntimeError(chunk.error or "generation failed")
            if chunk.chunk_type == "completed" and chunk.metrics is not None:
                completed = GenerationResult(text=chunk.text, metrics=chunk.metrics)
        if completed is None:
            raise RuntimeError("generation did not complete")
        return completed

    def _stream(
        self,
        session: Session,
        prompt: str,
        *,
        event_type: str,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        if not self.ready():
            raise RuntimeError("SGLang server is not ready")

        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        session.add_user_message(prompt)
        messages = session.transcript()
        prompt_tokens = self.tokenize(messages, generation=generation)
        yield StreamChunk.started(metadata={"prompt_tokens": prompt_tokens, "runtime": "sglang"})

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
        for chunk in self.server.stream_chat(payload):
            if chunk and first_token_at is None:
                first_token_at = time.perf_counter()
            if chunk:
                yield StreamChunk.delta(chunk)
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
            self.log_writer.write(generation_event("sglang", session, result, self.health(), event_type=event_type))

        yield StreamChunk.completed(text=text, metrics=result.metrics, metadata={"runtime": "sglang"})

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        yield from self._stream(session, prompt, event_type="generation", **kwargs)

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
            server_cfg.get("python", "python"),
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

    def _api_base(self) -> str:
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 31000)
        return f"http://{host}:{port}"

    def _log_dir(self) -> Path:
        log_cfg = self.config.get("logging", {})
        log_dir = Path(log_cfg.get("path", "experiments/logs"))
        if not log_dir.is_absolute():
            log_dir = self.project_root / log_dir
        return log_dir
