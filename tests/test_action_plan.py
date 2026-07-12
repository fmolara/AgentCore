from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from agentcore_server import ActionPlan, AgentLab, ApprovalPolicy, TaskExecutor
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


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


def plan_data() -> dict[str, Any]:
    return {
        "id": "test-plan",
        "title": "Edit parser",
        "description": "Replace parser return value.",
        "metadata": {"source": "test"},
        "actions": [
            {"type": "create_checkpoint", "label": "before"},
            {"type": "read_file", "path": "parser.c"},
            {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
            {"type": "git_diff"},
            {"type": "task_report"},
        ],
    }


def test_action_plan_json_and_yaml_serialization(tmp_path) -> None:
    plan = ActionPlan.from_dict(plan_data())
    json_path = tmp_path / "plan.json"
    yaml_path = tmp_path / "plan.yaml"

    plan.save(json_path)
    plan.save(yaml_path)

    loaded_json = ActionPlan.load(json_path)
    loaded_yaml = ActionPlan.load(yaml_path)

    assert loaded_json.as_dict() == plan.as_dict()
    assert loaded_yaml.as_dict() == plan.as_dict()
    assert loaded_json.actions[2].action_type == "replace_text"


def test_action_plan_rejects_unknown_action_type() -> None:
    data = plan_data()
    data["actions"] = [{"type": "shell", "command": "echo no"}]

    with pytest.raises(ValueError, match="unknown action type"):
        ActionPlan.from_dict(data)


def test_action_plan_rejects_missing_required_field() -> None:
    data = plan_data()
    data["actions"] = [{"type": "replace_text", "path": "parser.c", "old": "return 0;"}]

    with pytest.raises(ValueError, match="required field 'new'"):
        ActionPlan.from_dict(data)


def test_task_executor_executes_valid_action_plan(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(plan_data())
    task = agent.create_task(title=plan.title, description=plan.description)

    result = TaskExecutor(agent).execute_plan(task, plan, approved=True)

    assert result.status == "completed"
    assert task.status == "completed"
    assert "return 1;" in agent.files.read_text("parser.c")
    assert "+    return 1;" in agent.git.diff().stdout
    assert len(task.metadata["actions"]) == len(plan.actions)


def test_action_plan_path_traversal_is_rejected_during_execution(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(
        {
            "title": "Bad path",
            "actions": [
                {"type": "read_file", "path": "../outside.txt"},
            ],
        }
    )
    task = agent.create_task(title=plan.title)

    result = TaskExecutor(agent).execute_plan(task, plan)

    assert result.status == "failed"
    assert task.status == "failed"
    assert "path escapes workspace root" in result.error


def test_readonly_action_plan_runs_without_approval(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(
        {
            "title": "Inspect parser",
            "actions": [
                {"type": "read_file", "path": "parser.c"},
                {"type": "git_status"},
                {"type": "git_diff"},
                {"type": "task_report"},
            ],
        }
    )
    task = agent.create_task(title=plan.title)

    result = TaskExecutor(agent).execute_plan(task, plan)

    assert plan.required_approvals() == ()
    assert result.status == "completed"
    assert task.status == "completed"


def test_mutating_action_plan_requires_approval_by_default(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(plan_data())
    task = agent.create_task(title=plan.title)

    requirements = plan.required_approvals()
    result = TaskExecutor(agent).execute_plan(task, plan)

    assert [requirement.action_type for requirement in requirements] == ["create_checkpoint", "replace_text"]
    assert result.status == "approval_required"
    assert "mutating action requires approval" in result.error
    assert task.status == "created"
    assert "return 0;" in agent.files.read_text("parser.c")


def test_approved_mutating_action_plan_runs(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(plan_data())
    task = agent.create_task(title=plan.title)

    result = TaskExecutor(agent).execute_plan(task, plan, approved=True)

    assert result.status == "completed"
    assert task.status == "completed"
    assert "return 1;" in agent.files.read_text("parser.c")


def test_denied_action_type_is_rejected_by_policy(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict({"title": "Inspect parser", "actions": [{"type": "read_file", "path": "parser.c"}]})
    policy = ApprovalPolicy(denied_action_types=frozenset({"read_file"}))
    task = agent.create_task(title=plan.title)

    requirements = plan.required_approvals(policy)
    result = TaskExecutor(agent).execute_plan(task, plan, approval_policy=policy)

    assert requirements[0].as_dict() == {
        "action_index": 0,
        "action_type": "read_file",
        "reason": "action type is denied by policy",
    }
    assert result.status == "approval_required"
    assert task.status == "created"


def test_allowed_action_type_allowlist_is_enforced(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    plan = ActionPlan.from_dict(
        {
            "title": "Inspect parser",
            "actions": [
                {"type": "read_file", "path": "parser.c"},
                {"type": "git_diff"},
            ],
        }
    )
    policy = ApprovalPolicy(allowed_action_types=frozenset({"read_file"}))
    task = agent.create_task(title=plan.title)

    requirements = plan.required_approvals(policy)
    result = TaskExecutor(agent).execute_plan(task, plan, approval_policy=policy)

    assert len(requirements) == 1
    assert requirements[0].action_type == "git_diff"
    assert requirements[0].reason == "action type is not allowed by policy"
    assert result.status == "approval_required"
    assert task.status == "created"
