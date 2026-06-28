from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float | None
    tokens_per_sec: float
    wall_sec: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metrics: GenerationMetrics

