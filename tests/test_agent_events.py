from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from agentcore_server import AgentLab, ListEventSink, PlanProposalStatus
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self, response: str):
        self.response = response

    def load(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def ready(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "fake", "ready": True}

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        return self.generate(session, prompt or "warmup", max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        session.add_user_message(prompt)
        session.add_assistant_message(self.response)
        return GenerationResult(
            text=self.response,
            metrics=GenerationMetrics(
                prompt_tokens=1,
                generated_tokens=1,
                ttft_sec=0.01,
                tokens_per_sec=10.0,
                wall_sec=0.1,
            ),
        )

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self.generate(session, prompt, **kwargs).text

    def tokenize(self, text_or_messages: Any) -> int:
        return 1


def make_lab(workspace_root: Path, *, response: str) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = FakeRuntime(response)
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def plan_json(*, old: str = "return 0;", new: str = "return 1;") -> str:
    return json.dumps(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": old, "new": new},
            ],
            "metadata": {"source": "fake-model"},
        }
    )


def prepare_agent(tmp_path, *, response: str):
    sink = ListEventSink()
    lab = make_lab(tmp_path / "workspace", response=response)
    agent = lab.create_agent(
        system_prompt="You are a concise coding assistant.",
        event_sink=sink,
    )
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    task = agent.create_task(title="Edit parser", description="Replace parser return value.")
    return agent, task, sink


def event_types(sink: ListEventSink) -> list[str]:
    return [event.event_type for event in sink.events]


def test_plan_proposal_event_is_emitted_before_execution(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())

    proposal_result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert proposal_result.proposal is not None
    assert "plan.proposed" in event_types(sink)
    assert "execution.started" not in event_types(sink)


def test_mutating_plan_waits_for_approval(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())
    proposal = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c").proposal

    assert proposal is not None
    result = agent.execute_proposal(task, proposal)

    assert result.status == "approval_required"
    assert proposal.status == PlanProposalStatus.PROPOSED
    assert "action.started" not in event_types(sink)
    assert "return 0;" in agent.files.read_text("parser.c")


def test_rejected_proposal_cannot_execute(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())
    proposal = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c").proposal

    assert proposal is not None
    agent.reject_proposal(task, proposal, "not acceptable")
    with pytest.raises(ValueError, match="cannot execute rejected proposal"):
        agent.execute_proposal(task, proposal, approved=True)

    assert "plan.rejected" in event_types(sink)
    assert "return 0;" in agent.files.read_text("parser.c")


def test_approved_proposal_executes_and_emits_ordered_events(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())
    proposal = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c").proposal

    assert proposal is not None
    result = agent.execute_proposal(task, proposal, approved=True)

    assert result.status == "completed"
    assert "return 1;" in agent.files.read_text("parser.c")
    types = event_types(sink)
    expected = [
        "task.created",
        "plan.proposed",
        "plan.approved",
        "task.started",
        "execution.started",
        "action.started",
        "action.completed",
        "workspace.modified",
        "task.completed",
        "execution.completed",
    ]
    positions = [types.index(event_type) for event_type in expected]
    assert positions == sorted(positions)


def test_action_failure_appears_in_trace(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json(old="not present"))
    proposal = agent.propose_plan(task, instruction="Replace text in parser.c").proposal

    assert proposal is not None
    result = agent.execute_proposal(task, proposal, approved=True)

    assert result.status == "failed"
    types = event_types(sink)
    assert "action.failed" in types
    assert "task.failed" in types


def test_event_schema_does_not_expose_chain_of_thought_fields(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())
    proposal = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c").proposal

    assert proposal is not None
    agent.execute_proposal(task, proposal, approved=True)
    forbidden = {"chain_of_thought", "reasoning", "thoughts", "hidden_reasoning"}

    def walk(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for event in sink.as_dicts():
        assert set(event) == {"timestamp", "event_type", "task_id", "session_id", "summary", "payload"}
        walk(event)
