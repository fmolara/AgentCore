from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


COMPATIBILITY_CONTEXT_LIMIT = 4096


@dataclass(frozen=True)
class ContextCapabilities:
    configured_context_limit: int | None = None
    runtime_reported_context_limit: int | None = None
    model_declared_context_limit: int | None = None
    effective_context_limit: int = COMPATIBILITY_CONTEXT_LIMIT
    context_limit_source: str = "compatibility_fallback"

    @classmethod
    def discover(cls, config: dict[str, Any], runtime: Any) -> "ContextCapabilities":
        context = config.get("context", {})
        configured = _positive_int(
            context.get("max_context_tokens") if isinstance(context, dict) else None
        )
        reported: int | None = None
        declared: int | None = None
        capability = runtime.context_capabilities()
        if isinstance(capability, dict):
            reported = _positive_int(capability.get("runtime_context_limit"))
            declared = _positive_int(capability.get("model_declared_context_limit"))
        if configured is not None and reported is not None:
            effective = min(configured, reported)
            source = "configured_and_runtime_minimum"
        elif configured is not None:
            effective = configured
            source = "trusted_configuration"
        elif reported is not None:
            effective = reported
            source = "runtime_reported"
        else:
            effective = COMPATIBILITY_CONTEXT_LIMIT
            source = "compatibility_fallback"
        return cls(configured, reported, declared, effective, source)

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured_context_limit": self.configured_context_limit,
            "runtime_reported_context_limit": self.runtime_reported_context_limit,
            "model_declared_context_limit": self.model_declared_context_limit,
            "effective_context_limit": self.effective_context_limit,
            "context_limit_source": self.context_limit_source,
        }


@dataclass(frozen=True)
class PromptSection:
    name: str
    text: str


@dataclass(frozen=True)
class RenderedPrompt:
    sections: tuple[PromptSection, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections if section.text)

    def section_tokens(self, tokenize: Callable[[str], int]) -> dict[str, int]:
        return {
            section.name: tokenize(section.text)
            for section in self.sections
            if section.text
        }


@dataclass(frozen=True)
class ContextPolicy:
    safety_margin_tokens: int = 128
    phase_output_tokens: dict[str, int] = field(default_factory=dict)
    minimum_output_tokens: dict[str, int] = field(default_factory=dict)
    malformed_prefix_chars: int = 240
    malformed_suffix_chars: int = 240

    @classmethod
    def from_config(
        cls,
        data: dict[str, Any] | None,
        *,
        legacy_max_tokens: int,
    ) -> "ContextPolicy":
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("planner context configuration must be a mapping")
        outputs = {
            "exploration": legacy_max_tokens,
            "final_candidate": max(legacy_max_tokens, 1600),
            "format_recovery": max(legacy_max_tokens, 1600),
            "review": min(max(legacy_max_tokens, 512), 768),
            "revision": max(legacy_max_tokens, 1600),
            "final_review": min(max(legacy_max_tokens, 512), 768),
        }
        outputs.update(_phase_mapping(data.get("output_tokens"), outputs))
        minimums = {
            "exploration": min(outputs["exploration"], 384),
            "final_candidate": min(outputs["final_candidate"], 1500),
            "format_recovery": min(outputs["format_recovery"], 1500),
            "review": min(outputs["review"], 384),
            "revision": min(outputs["revision"], 1500),
            "final_review": min(outputs["final_review"], 384),
        }
        minimums.update(_phase_mapping(data.get("minimum_output_tokens"), minimums))
        malformed = data.get("malformed_output", {})
        if not isinstance(malformed, dict):
            raise ValueError("planner context malformed_output must be a mapping")
        policy = cls(
            safety_margin_tokens=_positive_int(data.get("safety_margin_tokens")) or 128,
            phase_output_tokens=outputs,
            minimum_output_tokens=minimums,
            malformed_prefix_chars=_positive_int(malformed.get("prefix_chars")) or 240,
            malformed_suffix_chars=_positive_int(malformed.get("suffix_chars")) or 240,
        )
        for phase, minimum in policy.minimum_output_tokens.items():
            if minimum > policy.phase_output_tokens[phase]:
                raise ValueError(f"minimum output exceeds requested output for {phase}")
        return policy


@dataclass(frozen=True)
class ContextPreflight:
    phase: str
    compaction_level: int
    capabilities: ContextCapabilities
    section_tokens: dict[str, int]
    total_prompt_tokens: int
    safety_margin_tokens: int
    requested_output_tokens: int
    configured_minimum_output_tokens: int
    effective_output_tokens: int

    @property
    def sufficient(self) -> bool:
        return self.effective_output_tokens >= self.configured_minimum_output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "compaction_level": self.compaction_level,
            "section_tokens": dict(self.section_tokens),
            "total_prompt_tokens": self.total_prompt_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "configured_minimum_output_tokens": self.configured_minimum_output_tokens,
            "effective_output_tokens": self.effective_output_tokens,
            **self.capabilities.as_dict(),
        }


def _phase_mapping(value: Any, defaults: dict[str, int]) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("planner phase token configuration must be a mapping")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError("unknown planner phase(s): " + ", ".join(sorted(unknown)))
    result: dict[str, int] = {}
    for key, item in value.items():
        parsed = _positive_int(item)
        if parsed is None:
            raise ValueError(f"planner token value for {key} must be positive")
        result[key] = parsed
    return result


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
