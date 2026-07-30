from __future__ import annotations

import json

from agentcore_server.planning.context import ContextCapabilities, ContextPolicy
from agentcore_server.planning.evidence import EvidenceBudget, build_evidence_pack
from agentcore_server.planning.exploration import (
    ExplorationObservation,
    ExplorationPlan,
    ExplorationRound,
    ExploreReadFileAction,
)


class CapabilityRuntime:
    def __init__(self, reported=None, declared=None):
        self.reported = reported
        self.declared = declared

    def context_capabilities(self):
        return {
            "runtime_context_limit": self.reported,
            "model_declared_context_limit": self.declared,
        }


def test_context_limit_precedence_and_model_metadata_is_informational() -> None:
    both = ContextCapabilities.discover(
        {"context": {"max_context_tokens": 4096}},
        CapabilityRuntime(reported=8192, declared=262144),
    )
    assert both.effective_context_limit == 4096
    assert both.context_limit_source == "configured_and_runtime_minimum"
    assert both.model_declared_context_limit == 262144

    runtime_only = ContextCapabilities.discover({}, CapabilityRuntime(reported=8192))
    assert runtime_only.effective_context_limit == 8192
    assert runtime_only.context_limit_source == "runtime_reported"

    metadata_only = ContextCapabilities.discover({}, CapabilityRuntime(declared=262144))
    assert metadata_only.effective_context_limit == 4096
    assert metadata_only.context_limit_source == "compatibility_fallback"


def test_context_policy_has_phase_specific_safe_minimums() -> None:
    policy = ContextPolicy.from_config(None, legacy_max_tokens=1024)
    assert policy.minimum_output_tokens["final_candidate"] == 1500
    assert policy.minimum_output_tokens["format_recovery"] == 1500
    assert policy.minimum_output_tokens["review"] == 384
    assert policy.phase_output_tokens["final_candidate"] == 1600


def test_evidence_pack_preserves_exact_numbered_spans_and_merges_matches() -> None:
    action = ExploreReadFileAction(path="src/parser.c")
    text = "\n".join(
        [
            "boilerplate",
            "long parse_value;",
            "overflow guard",
            "negative limit",
            "tail",
            "unrelated",
        ]
    )
    observation = ExplorationObservation.ok(
        action,
        data={"path": "src/parser.c", "text": text},
    )
    round_ = ExplorationRound(
        number=1,
        plan=ExplorationPlan("inspect", (action,)),
        observations=(observation,),
        observation_bytes=len(text),
    )
    pack = build_evidence_pack(
        [round_],
        instruction="Handle long negative overflow in parser.c",
        budget=EvidenceBudget(max_total_tokens=100, max_tokens_per_file=30, context_lines=1),
        tokenize=lambda value: max(1, len(value) // 4),
        compaction_level=0,
    )

    assert len(pack.items) == 1
    item = pack.items[0]
    assert item.path == "src/parser.c"
    assert item.observation_id == action.id
    assert item.spans
    assert item.spans[0].start_line <= item.spans[0].end_line
    assert all(": " in line for span in item.spans for line in span.text.splitlines())
    assert item.digest


def test_evidence_pack_is_deterministic_and_does_not_duplicate_observation() -> None:
    action = ExploreReadFileAction(path="tests/test_parser.c")
    observation = ExplorationObservation.ok(
        action,
        data={"path": action.path, "text": "test signed parser\n" * 30},
    )
    round_ = ExplorationRound(
        number=1,
        plan=ExplorationPlan("inspect", (action,)),
        observations=(observation,),
        observation_bytes=600,
    )
    kwargs = {
        "instruction": "test signed parser",
        "budget": EvidenceBudget(max_total_tokens=80, max_tokens_per_file=40),
        "tokenize": lambda value: max(1, len(value) // 4),
        "compaction_level": 2,
    }
    first = build_evidence_pack([round_], **kwargs)
    second = build_evidence_pack([round_], **kwargs)
    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(
        second.as_dict(), sort_keys=True
    )
    assert [item.observation_id for item in first.items] == [action.id]
