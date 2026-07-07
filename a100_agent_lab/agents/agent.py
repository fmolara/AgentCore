from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.sessions.session import Session
from a100_agent_lab.workspace import Workspace

if TYPE_CHECKING:
    from a100_agent_lab.api.client import AgentLab


class Agent:
    def __init__(
        self,
        lab: AgentLab,
        *,
        system_prompt: str | None = None,
        session: Session | None = None,
        workspace: Workspace | None = None,
        workspace_root: str | Path | None = None,
        workspace_mode: str = Workspace.READ_WRITE,
        workspace_metadata: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.lab = lab
        self.session = session or lab.create_session(system_prompt=system_prompt)
        self.workspace = workspace or Workspace(
            workspace_root or lab.default_workspace_root(),
            mode=workspace_mode,
            metadata=workspace_metadata,
        )
        self.generation_options = dict(generation_options or {})
        self.generation_options.update({key: value for key, value in kwargs.items() if value is not None})
        self._last_result: GenerationResult | None = None
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0

    @property
    def runtime(self):
        return self.lab.runtime

    @property
    def git(self):
        return self.workspace.git

    @property
    def last_metrics(self) -> GenerationMetrics | None:
        return None if self._last_result is None else self._last_result.metrics

    def ask(self, prompt: str, **kwargs: Any) -> GenerationResult:
        options = self._merged_options(kwargs)
        result = self.lab.generate(self.session, prompt, **options)
        self._record_result(result)
        return result

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        options = self._merged_options(kwargs)
        yield from self.lab.stream(self.session, prompt, **options)

    def reset(self) -> None:
        self.lab.reset_session(self.session.id)
        self._last_result = None
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0

    def statistics(self) -> dict[str, Any]:
        metrics = self.last_metrics
        return {
            "session_id": self.session.id,
            "conversation_turns": self.session.turn_count,
            "prompt_tokens": self._total_prompt_tokens,
            "generated_tokens": self._total_generated_tokens,
            "last_ttft_sec": None if metrics is None else metrics.ttft_sec,
            "last_tokens_per_sec": None if metrics is None else metrics.tokens_per_sec,
            "last_wall_sec": None if metrics is None else metrics.wall_sec,
            "generation_options": dict(self.generation_options),
            "workspace": self.workspace.as_dict(),
        }

    def _merged_options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        options = dict(self.generation_options)
        options.update({key: value for key, value in overrides.items() if value is not None})
        return options

    def _record_result(self, result: GenerationResult) -> None:
        self._last_result = result
        self._total_prompt_tokens += result.metrics.prompt_tokens
        self._total_generated_tokens += result.metrics.generated_tokens
