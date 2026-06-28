from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from a100_agent_lab.generation.config import GenerationConfig
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.logging.events import generation_event
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.runtime.server_process import ServerProcess
from a100_agent_lab.sessions.session import Session


class LMDeployRuntime(Runtime):
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
        self.server = ServerProcess(
            runtime_name="lmdeploy",
            api_base=self.api_base,
            project_root=project_root,
            log_dir=self._log_dir(),
            log_prefix="lmdeploy-server",
        )

    def load(self) -> None:
        if self.ready():
            return

        model_cfg = self.config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

        self.server.start(
            self._launch_command(),
            timeout=float(self.config.get("server", {}).get("startup_timeout_sec", 120)),
            env_updates={"CUDA_VISIBLE_DEVICES": str(self.config.get("gpu", {}).get("device", 0))},
        )

    def shutdown(self) -> None:
        self.server.shutdown()

    def ready(self) -> bool:
        return self.server.ready()

    def health(self) -> dict[str, Any]:
        health = self.server.health()
        health["gpu"] = self._gpu_info()
        return health

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        return self._generate(
            session,
            prompt or "Say ready.",
            event_type="warmup",
            max_tokens=max_tokens,
        )

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        return self._generate(session, prompt, event_type="generation", **kwargs)

    def _generate(
        self,
        session: Session,
        prompt: str,
        *,
        event_type: str,
        **kwargs: Any,
    ) -> GenerationResult:
        if not self.ready():
            raise RuntimeError("LMDeploy server is not ready")

        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        session.add_user_message(prompt)
        messages = session.transcript()
        prompt_tokens = self.tokenize(messages, generation=generation)

        payload = {
            "model": self.config["model"].get("name", self.config["model"]["path"]),
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
            self.log_writer.write(
                generation_event("lmdeploy", session, result, self.health(), event_type=event_type)
            )

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
        return [
            server_cfg.get("executable", "/opt/lmdeploy/bin/lmdeploy"),
            "serve",
            "api_server",
            model_cfg["path"],
            "--server-name",
            str(server_cfg.get("host", "127.0.0.1")),
            "--server-port",
            str(server_cfg.get("port", 32000)),
            "--backend",
            str(server_cfg.get("backend", "turbomind")),
            "--model-format",
            str(server_cfg.get("model_format", "hf")),
            "--dtype",
            str(model_cfg.get("dtype", "bfloat16")),
            "--session-len",
            str(context_cfg.get("max_context_tokens", 4096)),
            "--max-batch-size",
            str(server_cfg.get("max_batch_size", 1)),
            "--cache-max-entry-count",
            str(server_cfg.get("cache_max_entry_count", 0.2)),
            "--model-name",
            str(model_cfg.get("name", "model")),
            "--log-level",
            str(server_cfg.get("log_level", "INFO")),
            "--max-log-len",
            str(server_cfg.get("max_log_len", 0)),
            "--trust-remote-code",
        ]

    def _api_base(self) -> str:
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 32000)
        return f"http://{host}:{port}"

    def _log_dir(self) -> Path:
        log_cfg = self.config.get("logging", {})
        log_dir = Path(log_cfg.get("path", "experiments/logs"))
        if not log_dir.is_absolute():
            log_dir = self.project_root / log_dir
        return log_dir

    def _gpu_info(self) -> dict[str, Any]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            used, total, util, power = [item.strip() for item in output.splitlines()[0].split(",")]
            return {
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_gpu_pct": int(util),
                "power_w": float(power),
            }
        except Exception as exc:
            return {"error": repr(exc)}
