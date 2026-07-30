from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from uuid import uuid4

from agentcore_server.executor import ActionPlan


@dataclass(frozen=True)
class PlanDiagnostic:
    code: str
    message: str
    severity: str = "warning"
    action_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "action_index": self.action_index,
        }


@dataclass(frozen=True)
class FinalCandidate:
    action_plan: ActionPlan
    diagnostics: tuple[PlanDiagnostic, ...] = ()
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", uuid4().hex)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_plan": self.action_plan.as_dict(),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class ReviewFinding:
    severity: str
    requirement: str
    problem: str
    required_change: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewFinding":
        _require_fields(
            data,
            required={"severity", "requirement", "problem", "required_change"},
        )
        severity = _required_string(data, "severity")
        if severity not in {"minor", "major", "critical"}:
            raise ValueError("review finding severity must be minor, major, or critical")
        return cls(
            severity=severity,
            requirement=_required_string(data, "requirement"),
            problem=_required_string(data, "problem"),
            required_change=_required_string(data, "required_change"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "requirement": self.requirement,
            "problem": self.problem,
            "required_change": self.required_change,
        }


@dataclass(frozen=True)
class CandidateReview:
    verdict: str
    summary: str = ""
    findings: tuple[ReviewFinding, ...] = ()
    reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateReview":
        if not isinstance(data, dict):
            raise ValueError("candidate review must be a mapping")
        verdict = _required_string(data, "verdict")
        if verdict == "accept":
            _require_fields(data, required={"verdict", "summary", "findings"})
            findings = _findings(data["findings"])
            if findings:
                raise ValueError("accepted candidate review must not contain findings")
            return cls(verdict=verdict, summary=_required_string(data, "summary"))
        if verdict == "revise":
            _require_fields(data, required={"verdict", "summary", "findings"})
            findings = _findings(data["findings"])
            if not findings:
                raise ValueError("revision review must contain at least one finding")
            return cls(
                verdict=verdict,
                summary=_required_string(data, "summary"),
                findings=findings,
            )
        if verdict == "cannot_verify":
            _require_fields(data, required={"verdict", "reason"})
            return cls(verdict=verdict, reason=_required_string(data, "reason"))
        raise ValueError("review verdict must be accept, revise, or cannot_verify")

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"verdict": self.verdict}
        if self.verdict == "cannot_verify":
            data["reason"] = self.reason
        else:
            data["summary"] = self.summary
            data["findings"] = [item.as_dict() for item in self.findings]
        return data


def build_recovery_prompt(
    *,
    expected_schema: str,
    task_context: dict[str, Any],
    observations: list[dict[str, Any]],
    malformed_output: str,
) -> str:
    context = {
        "task": task_context,
        "observations": observations,
        "malformed_output": malformed_output,
    }
    return (
        "Return one complete replacement JSON object only. Do not continue or "
        "repair the partial text. Do not return Markdown or prose. Do not reveal "
        "hidden chain-of-thought.\n\nRequired schema:\n"
        + expected_schema
        + "\n\nContext:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )


def build_review_prompt(
    *,
    task_context: dict[str, Any],
    observations: list[dict[str, Any]],
    candidate: FinalCandidate,
    check_names: tuple[str, ...],
) -> str:
    context = {
        "task": task_context,
        "observations": observations,
        "candidate": candidate.as_dict(),
        "configured_checks": list(check_names),
    }
    return (
        "You are an independent AgentCore candidate-plan reviewer. Review only "
        "the concrete evidence supplied. Do not reveal hidden chain-of-thought. "
        "Check task coverage, consistency with observed source and tests, exact "
        "old/new edit content, language-level correctness where inferable, "
        "explicit portability and undefined-behavior requirements, preservation "
        "requirements, malformed-input coverage, configured checks, unrelated "
        "changes, arbitrary commands, and automatic commits.\n\n"
        "Return exactly one JSON object using one of these forms:\n"
        '{"verdict":"accept","summary":"concise evidence-based summary","findings":[]}\n'
        '{"verdict":"revise","summary":"concise summary","findings":'
        '[{"severity":"major","requirement":"...","problem":"...",'
        '"required_change":"..."}]}\n'
        '{"verdict":"cannot_verify","reason":"..."}\n\n'
        "An accept verdict means no material defect was identified from the "
        "available evidence; it is not a formal proof.\n\nContext:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )


def build_revision_prompt(
    *,
    task_context: dict[str, Any],
    observations: list[dict[str, Any]],
    candidate: FinalCandidate,
    review: CandidateReview,
    final_schema: str,
    check_names: tuple[str, ...],
) -> str:
    context = {
        "task": task_context,
        "observations": observations,
        "rejected_candidate": candidate.as_dict(),
        "review": review.as_dict(),
        "configured_checks": list(check_names),
    }
    return (
        "Return one complete revised FINAL planning JSON object. Apply every "
        "required review change. Return neither a JSON patch nor prose. Prefer "
        "compact replace_text actions for localized edits to existing files. "
        "Do not reveal hidden chain-of-thought.\n\nRequired schema:\n"
        + final_schema
        + "\n\nContext:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )


def _findings(value: Any) -> tuple[ReviewFinding, ...]:
    if not isinstance(value, list):
        raise ValueError("review findings must be a list")
    return tuple(ReviewFinding.from_dict(item) for item in value)


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review field {field!r} must be a non-empty string")
    return value


def _require_fields(data: dict[str, Any], *, required: set[str]) -> None:
    unknown = set(data) - required
    missing = required - set(data)
    if missing:
        raise ValueError("missing review field(s): " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("unknown review field(s): " + ", ".join(sorted(unknown)))
