from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentcore_server.generation.result import GenerationMetrics


@dataclass(frozen=True)
class StreamChunk:
    chunk_type: str
    text_delta: str = ""
    text: str = ""
    metrics: GenerationMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    @classmethod
    def started(cls, *, metadata: dict[str, Any] | None = None) -> "StreamChunk":
        return cls(chunk_type="started", metadata=deepcopy(metadata or {}))

    @classmethod
    def delta(cls, text_delta: str, *, metadata: dict[str, Any] | None = None) -> "StreamChunk":
        return cls(chunk_type="delta", text_delta=text_delta, metadata=deepcopy(metadata or {}))

    @classmethod
    def completed(
        cls,
        *,
        text: str,
        metrics: GenerationMetrics,
        metadata: dict[str, Any] | None = None,
    ) -> "StreamChunk":
        return cls(chunk_type="completed", text=text, metrics=metrics, metadata=deepcopy(metadata or {}))

    @classmethod
    def failed(cls, error: str, *, metadata: dict[str, Any] | None = None) -> "StreamChunk":
        return cls(chunk_type="failed", error=error, metadata=deepcopy(metadata or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "chunk_type": self.chunk_type,
            "text_delta": self.text_delta,
            "text": self.text,
            "metrics": None if self.metrics is None else self.metrics.as_dict(),
            "metadata": deepcopy(self.metadata),
            "error": self.error,
        }
