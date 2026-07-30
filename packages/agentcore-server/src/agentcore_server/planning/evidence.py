from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable

from agentcore_server.planning.exploration import ExplorationRound


_STOP_WORDS = {
    "about", "after", "again", "all", "and", "another", "before", "both",
    "but", "can", "change", "complete", "could",
    "continue", "existing", "from", "have", "including", "integer", "into",
    "least", "must", "not", "only", "parser", "should", "support", "that",
    "the", "their", "these", "this", "through", "with", "without", "would",
    "actionplan", "approval", "based", "behavior", "character", "check", "code", "commit",
    "configured", "create", "current", "editing", "execution", "final",
    "implement", "implementation", "include", "inspect", "modify", "plan",
    "produce", "propose", "report", "required", "run", "task", "test", "tests",
    "verification",
}


@dataclass(frozen=True)
class EvidenceBudget:
    max_total_tokens: int = 1100
    max_tokens_per_file: int = 320
    max_files: int = 8
    context_lines: int = 2

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "EvidenceBudget":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("planner evidence configuration must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("unknown evidence setting(s): " + ", ".join(sorted(unknown)))
        values = {}
        for key, value in data.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"evidence setting {key} must be positive")
            values[key] = value
        return cls(**values)


@dataclass(frozen=True)
class EvidenceSpan:
    start_line: int
    end_line: int
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }


@dataclass(frozen=True)
class EvidenceItem:
    path: str
    observation_id: str
    evidence_type: str
    spans: tuple[EvidenceSpan, ...]
    digest: str
    source_truncated: bool
    omitted_ranges: tuple[tuple[int, int], ...]
    selection_reason: str

    def as_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        spans = [span.as_dict() for span in self.spans]
        if not include_text:
            spans = [
                {"start_line": item["start_line"], "end_line": item["end_line"]}
                for item in spans
            ]
        return {
            "path": self.path,
            "observation_id": self.observation_id,
            "evidence_type": self.evidence_type,
            "spans": spans,
            "digest": self.digest,
            "source_truncated": self.source_truncated,
            "omitted_ranges": [list(item) for item in self.omitted_ranges],
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class EvidencePack:
    items: tuple[EvidenceItem, ...]
    compaction_level: int
    omitted_observation_ids: tuple[str, ...] = ()

    def as_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "compaction_level": self.compaction_level,
            "items": [item.as_dict(include_text=include_text) for item in self.items],
            "omitted_observation_ids": list(self.omitted_observation_ids),
        }


def build_evidence_pack(
    rounds: tuple[ExplorationRound, ...] | list[ExplorationRound],
    *,
    instruction: str,
    budget: EvidenceBudget,
    tokenize: Callable[[str], int],
    compaction_level: int,
) -> EvidencePack:
    terms = _task_terms(instruction)
    candidates: list[tuple[int, str, Any]] = []
    read_paths = {
        observation.data.get("path")
        for round_ in rounds
        for observation in round_.observations
        if observation.status == "ok"
        and isinstance(observation.data.get("text"), str)
        and isinstance(observation.data.get("path"), str)
    }
    latest = max((round_.number for round_ in rounds), default=0)
    for round_ in rounds:
        for observation in round_.observations:
            path = observation.data.get("path")
            text = observation.data.get("text")
            matches = observation.data.get("matches")
            if observation.status == "ok" and isinstance(matches, list):
                for match in matches:
                    match_path = match.get("path") if isinstance(match, dict) else None
                    if isinstance(match_path, str):
                        candidates.append(
                            (
                                -55,
                                match_path,
                                ("search", observation, match_path),
                            )
                        )
            entries = observation.data.get("entries")
            if observation.status == "ok" and isinstance(entries, list):
                for entry in entries:
                    entry_path = entry.get("path") if isinstance(entry, dict) else None
                    if isinstance(entry_path, str) and entry_path not in read_paths:
                        candidates.append(
                            (
                                -25,
                                entry_path,
                                ("listing", observation, entry_path),
                            )
                        )
            if observation.status != "ok" or not isinstance(path, str) or not isinstance(text, str):
                continue
            role = _role(path)
            score = role + (20 if round_.number == latest else 0)
            lowered = (path + "\n" + text).lower()
            score += min(20, sum(2 for term in terms if term in lowered))
            candidates.append((-score, path, observation))
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2][1].action_id if isinstance(item[2], tuple) else item[2].action_id,
        )
    )
    max_files = max(1, budget.max_files - compaction_level * 2)
    per_file = max(120, budget.max_tokens_per_file - compaction_level * 100)
    total_limit = max(400, budget.max_total_tokens - compaction_level * 220)
    items: list[EvidenceItem] = []
    omitted: list[str] = []
    total = 0
    for _, path, observation_value in candidates:
        if isinstance(observation_value, tuple):
            source_kind, observation, match_path = observation_value
            item = EvidenceItem(
                match_path,
                observation.action_id,
                "search_match" if source_kind == "search" else "directory_entry",
                (),
                hashlib.sha256(match_path.encode("utf-8")).hexdigest(),
                observation.truncated,
                (),
                "successful_search_match" if source_kind == "search" else "directory_listing",
            )
        else:
            observation = observation_value
            item = _item_from_text(
                path,
                observation.action_id,
                observation.data["text"],
                terms=terms,
                per_file_tokens=per_file,
                tokenize=tokenize,
                source_truncated=observation.truncated,
                context_lines=max(0, budget.context_lines - compaction_level),
            )
        if len(items) >= max_files:
            omitted.append(observation.action_id)
            continue
        serialized_tokens = tokenize(json.dumps(item.as_dict(), separators=(",", ":")))
        if items and total + serialized_tokens > total_limit:
            omitted.append(observation.action_id)
            continue
        items.append(item)
        total += serialized_tokens
    return EvidencePack(tuple(items), compaction_level, tuple(sorted(set(omitted))))


def _item_from_text(
    path: str,
    observation_id: str,
    text: str,
    *,
    terms: tuple[str, ...],
    per_file_tokens: int,
    tokenize: Callable[[str], int],
    source_truncated: bool,
    context_lines: int,
) -> EvidenceItem:
    lines = text.splitlines()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if tokenize(text) <= per_file_tokens:
        spans = (EvidenceSpan(1, len(lines), _numbered(lines, 1)),)
        complete = EvidenceItem(
            path, observation_id, _role_name(path), spans, digest,
            source_truncated, (), "small_relevant_file",
        )
        if tokenize(json.dumps(complete.as_dict(), separators=(",", ":"))) <= per_file_tokens:
            return complete
    scored_matches = [
        (sum(term in line.lower() for term in terms), index)
        for index, line in enumerate(lines)
    ]
    matches = [
        index
        for score, index in sorted(scored_matches, key=lambda item: (-item[0], item[1]))
        if score
    ][:8]
    if not matches:
        matches = list(range(min(len(lines), 12)))
    ranges = _merge_ranges(
        (max(0, index - context_lines), min(len(lines) - 1, index + context_lines))
        for index in matches
    )
    selected: list[EvidenceSpan] = []
    for start, end in ranges:
        selected_end = end
        while True:
            rendered = _numbered(lines[start : selected_end + 1], start + 1)
            proposed_spans = tuple(
                selected + [EvidenceSpan(start + 1, selected_end + 1, rendered)]
            )
            provisional = EvidenceItem(
                path,
                observation_id,
                _role_name(path),
                proposed_spans,
                digest,
                source_truncated,
                (),
                "task_term_and_file_role",
            )
            provisional_tokens = tokenize(
                json.dumps(provisional.as_dict(), separators=(",", ":"))
            )
            if provisional_tokens <= per_file_tokens or selected_end == start:
                break
            selected_end -= 1
        if provisional_tokens > per_file_tokens:
            break
        selected.append(EvidenceSpan(start + 1, selected_end + 1, rendered))
    covered = [(span.start_line, span.end_line) for span in selected]
    return EvidenceItem(
        path,
        observation_id,
        _role_name(path),
        tuple(selected),
        digest,
        source_truncated,
        tuple(_omitted_ranges(len(lines), covered)),
        "task_term_and_file_role",
    )


def _task_terms(instruction: str) -> tuple[str, ...]:
    terms = {
        token.lower().strip("./-")
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", instruction)
        if token.lower().strip("./-") not in _STOP_WORDS
    }
    return tuple(sorted(terms))


def _role(path: str) -> int:
    lowered = path.lower()
    if "/test" in lowered or lowered.startswith("test"):
        return 35
    if lowered.endswith((".c", ".cc", ".cpp", ".py", ".rs")):
        return 40
    if lowered.endswith((".h", ".hpp")):
        return 30
    return 10


def _role_name(path: str) -> str:
    lowered = path.lower()
    if "/test" in lowered or lowered.startswith("test"):
        return "test"
    if lowered.endswith((".h", ".hpp")):
        return "public_declaration"
    return "implementation" if _role(path) >= 40 else "file"


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{number}: {line}" for number, line in enumerate(lines, start=start))


def _merge_ranges(ranges) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _omitted_ranges(total: int, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    omitted: list[tuple[int, int]] = []
    cursor = 1
    for start, end in covered:
        if cursor < start:
            omitted.append((cursor, start - 1))
        cursor = end + 1
    if cursor <= total:
        omitted.append((cursor, total))
    return omitted
