from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from a100_agent_lab.generation.result import GenerationResult
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.runtime.lmdeploy import LMDeployRuntime
from a100_agent_lab.runtime.sglang import SGLangRuntime
from a100_agent_lab.runtime.transformers import TransformersRuntime
from a100_agent_lab.sessions.session import Session


class AgentLab:
    def __init__(self, config: dict[str, Any], *, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.runtime = self._build_runtime()

    @classmethod
    def from_config(cls, path: str | Path) -> "AgentLab":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        project_root = config_path.parent.parent
        return cls(config, project_root=project_root)

    def start(self) -> None:
        self.runtime.load()

    def shutdown(self) -> None:
        self.runtime.shutdown()

    def ready(self) -> bool:
        return self.runtime.ready()

    def health(self) -> dict[str, Any]:
        return self.runtime.health()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        return self.runtime.warmup(prompt=prompt, max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return self.runtime.create_session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        return self.runtime.generate(session, prompt, **kwargs)

    def stream(self, session: Session, prompt: str, **kwargs: Any):
        return self.runtime.stream(session, prompt, **kwargs)

    def _build_runtime(self) -> Runtime:
        runtime_name = self.config.get("runtime")
        writer = self._build_log_writer()
        if runtime_name == "transformers":
            return TransformersRuntime(self.config, log_writer=writer)
        if runtime_name == "sglang":
            return SGLangRuntime(self.config, project_root=self.project_root, log_writer=writer)
        if runtime_name == "lmdeploy":
            return LMDeployRuntime(self.config, project_root=self.project_root, log_writer=writer)
        raise ValueError(f"unknown runtime: {runtime_name}")

    def _build_log_writer(self) -> JsonlWriter | None:
        log_cfg = self.config.get("logging", {})
        if log_cfg.get("format", "jsonl") != "jsonl":
            return None
        log_dir = Path(log_cfg.get("path", "experiments/logs"))
        if not log_dir.is_absolute():
            log_dir = self.project_root / log_dir
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return JsonlWriter(log_dir / f"generation-{stamp}.jsonl")
