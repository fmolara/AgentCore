from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from a100_agent_lab import AgentLab, PlanProposalStatus
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

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
        self.last_prompt = prompt
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
    runtime = FakeRuntime(response)
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = runtime
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def valid_mutating_plan_json() -> str:
    return json.dumps(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
                {"type": "git_diff"},
            ],
            "metadata": {"source": "fake-model"},
        }
    )


def executable_mutating_plan_json() -> str:
    return json.dumps(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
            ],
            "metadata": {"source": "fake-model"},
        }
    )


def prepare_agent(tmp_path, *, response: str):
    lab = make_lab(tmp_path / "workspace", response=response)
    agent = lab.create_agent(system_prompt="You are a concise coding assistant.")
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    task = agent.create_task(title="Edit parser", description="Replace parser return value.")
    return agent, task


def test_valid_llm_plan_produces_plan_proposal(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response=valid_mutating_plan_json())

    result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.status == PlanProposalStatus.PROPOSED
    assert result.proposal.action_plan.title == "Edit parser"
    assert result.proposal.metadata["planner"] == "simple_llm"
    assert "shell commands" in agent.runtime.last_prompt


def test_invalid_json_is_reported_as_failed_planner_result(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response="this is not json")

    result = agent.propose_plan(task, instruction="Replace return value.")

    assert result.status == "failed"
    assert result.proposal is None
    assert result.error is not None
    assert "valid JSON" in result.error


def test_unknown_action_is_rejected_cleanly(tmp_path) -> None:
    agent, task = prepare_agent(
        tmp_path,
        response=json.dumps(
            {
                "title": "Bad plan",
                "actions": [{"type": "shell", "command": "echo no"}],
            }
        ),
    )

    result = agent.propose_plan(task, instruction="Run a command.")

    assert result.status == "failed"
    assert result.proposal is None
    assert result.error == "unknown action type: shell"


def test_mutating_llm_proposal_requires_approval(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response=valid_mutating_plan_json())

    result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert result.proposal is not None
    requirements = result.proposal.approval_requirements
    assert [requirement.action_type for requirement in requirements] == ["replace_text"]
    assert requirements[0].reason == "mutating action requires approval"
    assert "return 0;" in agent.files.read_text("parser.c")


def test_approved_llm_proposal_can_be_executed(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response=executable_mutating_plan_json())
    proposal_result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert proposal_result.proposal is not None
    execution = agent.execute_proposal(task, proposal_result.proposal, approved=True)

    assert execution.status == "completed"
    assert proposal_result.proposal.status == PlanProposalStatus.EXECUTED
    assert task.status == "completed"
    assert "return 1;" in agent.files.read_text("parser.c")


def test_unapproved_mutating_llm_proposal_is_refused(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response=executable_mutating_plan_json())
    proposal_result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert proposal_result.proposal is not None
    execution = agent.execute_proposal(task, proposal_result.proposal)

    assert execution.status == "approval_required"
    assert proposal_result.proposal.status == PlanProposalStatus.PROPOSED
    assert task.status == "created"
    assert "return 0;" in agent.files.read_text("parser.c")


def test_rejected_llm_proposal_cannot_execute(tmp_path) -> None:
    agent, task = prepare_agent(tmp_path, response=executable_mutating_plan_json())
    proposal_result = agent.propose_plan(task, instruction="Replace return 0 with return 1 in parser.c")

    assert proposal_result.proposal is not None
    proposal_result.proposal.reject("not the requested edit")
    with pytest.raises(ValueError, match="cannot execute rejected proposal"):
        agent.execute_proposal(task, proposal_result.proposal, approved=True)

    assert "return 0;" in agent.files.read_text("parser.c")
