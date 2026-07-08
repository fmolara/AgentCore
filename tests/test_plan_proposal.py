from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from a100_agent_lab import ActionPlan, AgentLab, PlanProposal, PlanProposalStatus, TaskExecutor
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions import Session, SessionStore


class FakeRuntime(Runtime):
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
        session.add_assistant_message("ok")
        return GenerationResult(
            text="ok",
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


def make_lab(workspace_root: Path) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = FakeRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def set_test_git_identity(monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "agentcore-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "agentcore-test@example.invalid")


def prepare_agent(tmp_path, monkeypatch):
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")
    return agent


def readonly_plan() -> ActionPlan:
    return ActionPlan.from_dict(
        {
            "title": "Inspect parser",
            "actions": [
                {"type": "read_file", "path": "parser.c"},
                {"type": "git_status"},
                {"type": "task_report"},
            ],
        }
    )


def mutating_plan() -> ActionPlan:
    return ActionPlan.from_dict(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
                {"type": "git_diff"},
            ],
        }
    )


def test_plan_proposal_creation_and_json_serialization(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Edit parser")
    plan = mutating_plan()

    proposal = PlanProposal.from_action_plan(
        task_id=task.id,
        action_plan=plan,
        summary="Replace the parser return value.",
        metadata={"source": "test"},
    )
    path = tmp_path / "proposal.json"
    proposal.save(path)
    loaded = PlanProposal.load(path)

    assert proposal.status == PlanProposalStatus.PROPOSED
    assert [requirement.action_type for requirement in proposal.approval_requirements] == ["replace_text"]
    assert loaded.as_dict() == proposal.as_dict()
    assert loaded.metadata == {"source": "test"}


def test_plan_proposal_approve_and_reject_transitions(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Edit parser")
    plan = mutating_plan()

    approved = PlanProposal.from_action_plan(task_id=task.id, action_plan=plan).approve()
    rejected = PlanProposal.from_action_plan(task_id=task.id, action_plan=plan).reject("not acceptable")

    assert approved.status == PlanProposalStatus.APPROVED
    assert approved.approved_at is not None
    assert rejected.status == PlanProposalStatus.REJECTED
    assert rejected.metadata["rejection_reason"] == "not acceptable"

    with pytest.raises(ValueError, match="cannot approve"):
        approved.approve()
    with pytest.raises(ValueError, match="cannot reject"):
        rejected.reject("again")


def test_plan_proposal_cannot_execute_rejected(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Edit parser")
    proposal = PlanProposal.from_action_plan(task_id=task.id, action_plan=mutating_plan()).reject("no")

    with pytest.raises(ValueError, match="cannot execute rejected proposal"):
        proposal.execute(TaskExecutor(agent), task)


def test_mutating_plan_proposal_requires_approval(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Edit parser")
    proposal = PlanProposal.from_action_plan(task_id=task.id, action_plan=mutating_plan())

    result = proposal.execute(TaskExecutor(agent), task)

    assert result.status == "approval_required"
    assert "plan proposal requires approval" in result.error
    assert proposal.status == PlanProposalStatus.PROPOSED
    assert task.status == "created"
    assert "return 0;" in agent.files.read_text("parser.c")


def test_readonly_plan_proposal_executes_without_explicit_approval(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Inspect parser")
    proposal = PlanProposal.from_action_plan(task_id=task.id, action_plan=readonly_plan())

    result = proposal.execute(TaskExecutor(agent), task)

    assert proposal.approval_requirements == ()
    assert result.status == "completed"
    assert proposal.status == PlanProposalStatus.EXECUTED
    assert proposal.executed_at is not None
    assert task.status == "completed"


def test_approved_mutating_plan_proposal_executes_and_updates_status(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Edit parser")
    proposal = PlanProposal.from_action_plan(task_id=task.id, action_plan=mutating_plan())

    result = proposal.approve().execute(TaskExecutor(agent), task)

    assert result.status == "completed"
    assert proposal.status == PlanProposalStatus.EXECUTED
    assert proposal.executed_at is not None
    assert task.status == "completed"
    assert "return 1;" in agent.files.read_text("parser.c")
