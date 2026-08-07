from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 64
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        return cls(
            temperature=float(data.get("temperature", cls.temperature)),
            max_tokens=int(data.get("max_tokens", cls.max_tokens)),
            top_p=float(data.get("top_p", cls.top_p)),
            top_k=(
                None if data.get("top_k") is None else int(data["top_k"])
            ),
            repetition_penalty=(
                None
                if data.get("repetition_penalty") is None
                else float(data["repetition_penalty"])
            ),
            enable_thinking=bool(data.get("enable_thinking", cls.enable_thinking)),
        )

    def override(self, **kwargs: Any) -> "GenerationConfig":
        allowed = {k: v for k, v in kwargs.items() if v is not None and hasattr(self, k)}
        return replace(self, **allowed)
