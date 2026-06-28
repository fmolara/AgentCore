from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from a100_agent_lab.generation.result import GenerationResult
from a100_agent_lab.sessions.session import Session


class Runtime(ABC):
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
    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text_or_messages: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def statistics(self) -> dict[str, Any]:
        raise NotImplementedError
