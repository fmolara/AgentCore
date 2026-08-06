from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentcore_server.agents import Agent
from agentcore_server.generation.result import GenerationResult
from agentcore_server.logging.writer import JsonlWriter
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions.session import Session
from agentcore_server.sessions.store import SessionStore
from agentcore_server.workspace import Workspace


class AgentLab:
    def __init__(self, config: dict[str, Any], *, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.runtime = self._build_runtime()
        self.sessions = SessionStore(self.runtime.create_session)

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

    def statistics(self) -> dict[str, Any]:
        return self.runtime.statistics()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        return self.runtime.warmup(prompt=prompt, max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return self.sessions.create(system_prompt=system_prompt)

    def get_session(self, session_id: str) -> Session:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        return self.sessions.list()

    def reset_session(self, session_id: str) -> Session:
        return self.sessions.reset(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.sessions.delete(session_id)

    def create_agent(
        self,
        *,
        system_prompt: str | None = None,
        session: Session | None = None,
        workspace: Workspace | None = None,
        workspace_root: str | Path | None = None,
        workspace_mode: str = Workspace.READ_WRITE,
        workspace_metadata: dict[str, Any] | None = None,
        workspace_checks: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
        event_sink: Any | None = None,
        **kwargs: Any,
    ) -> Agent:
        return Agent(
            self,
            system_prompt=system_prompt,
            session=session,
            workspace=workspace,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            workspace_metadata=workspace_metadata,
            workspace_checks=(
                self.workspace_checks()
                if workspace_checks is None
                else workspace_checks
            ),
            generation_options=generation_options,
            event_sink=event_sink,
            **kwargs,
        )

    def default_workspace_root(self) -> Path:
        workspace_cfg = self.config.get("workspace", {})
        root = Path(workspace_cfg.get("root", "workspace"))
        if not root.is_absolute():
            root = self.project_root / root
        return root

    def workspace_checks(self) -> dict[str, Any]:
        workspace_cfg = self.config.get("workspace", {})
        if not isinstance(workspace_cfg, dict):
            raise ValueError("workspace configuration must be a mapping")
        checks = workspace_cfg.get("checks", {})
        if not isinstance(checks, dict):
            raise ValueError("workspace checks configuration must be a mapping")
        return checks

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        return self.runtime.generate(session, prompt, **kwargs)

    def stream(self, session: Session, prompt: str, **kwargs: Any):
        return self.runtime.stream(session, prompt, **kwargs)

    def _build_runtime(self) -> Runtime:
        runtime_name = self.config.get("runtime")
        writer = self._build_log_writer()
        if runtime_name == "transformers":
            from agentcore_server.runtime.transformers import TransformersRuntime

            return TransformersRuntime(self.config, log_writer=writer)
        if runtime_name == "sglang":
            from agentcore_server.runtime.sglang import SGLangRuntime

            return SGLangRuntime(self.config, project_root=self.project_root, log_writer=writer)
        if runtime_name == "vllm":
            from agentcore_server.runtime.vllm import VLLMRuntime

            return VLLMRuntime(self.config, project_root=self.project_root, log_writer=writer)
        if runtime_name == "lmdeploy":
            from agentcore_server.runtime.lmdeploy import LMDeployRuntime

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
