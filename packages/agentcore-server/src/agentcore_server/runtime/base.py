from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Iterator

from agentcore_server.generation.result import GenerationResult
from agentcore_server.generation.stream import StreamChunk
from agentcore_server.sessions.session import Session


class Runtime(ABC):
    def context_capabilities(self) -> dict[str, int | None]:
        """Return reliable runtime capacity separately from model metadata."""
        return {}

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def create_session(self, *, system_prompt: str | None = None) -> Session:
        raise NotImplementedError

    @abstractmethod
    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text_or_messages: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def statistics(self) -> dict[str, Any]:
        raise NotImplementedError


def model_declared_context_limit(config: dict[str, Any]) -> int | None:
    """Read informational architecture metadata without treating it as capacity."""
    model_path = config.get("model", {}).get("path")
    if not isinstance(model_path, str):
        return None
    try:
        data = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    text_config = data.get("text_config", {})
    candidates = [
        text_config.get("max_position_embeddings") if isinstance(text_config, dict) else None,
        data.get("max_position_embeddings"),
    ]
    return next(
        (
            value
            for value in candidates
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ),
        None,
    )
