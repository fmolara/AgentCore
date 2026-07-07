from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from a100_agent_lab import (
    AgentLab,
    Task,
    TaskCheckpoint,
    TaskCheckpointComparison,
    TaskCheckpointRestorePlan,
    TaskReport,
    TaskStatus,
)
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self, log_writer: JsonlWriter | None = None) -> None:
        self.loaded = False
        self.log_writer = log_writer

    def load(self) -> None:
        self.loaded = True

    def shutdown(self) -> None:
        self.loaded = False

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "fake", "ready": self.ready()}

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


def make_lab(workspace_root: Path, *, log_writer: JsonlWriter | None = None) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = FakeRuntime(log_writer=log_writer)
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def set_test_git_identity(monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "agentcore-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "agentcore-test@example.invalid")


def test_task_lifecycle() -> None:
    task = Task(title="Refactor parser", description="Replace strtok.")

    assert task.status == TaskStatus.CREATED
    assert task.started_at is None

    task.start()

    assert task.status == TaskStatus.RUNNING
    assert task.started_at is not None

    task.complete()

    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None


def test_task_fail_cancel_and_invalid_transitions() -> None:
    failed = Task(title="Failing task")
    failed.start()
    failed.fail("parser regression")

    assert failed.status == TaskStatus.FAILED
    assert failed.failure_reason == "parser regression"

    cancelled = Task(title="Cancelled task")
    cancelled.cancel()
    assert cancelled.status == TaskStatus.CANCELLED

    with pytest.raises(ValueError):
        cancelled.start()

    with pytest.raises(ValueError):
        Task(title="bad").fail(" ")


def test_task_json_serialization() -> None:
    task = Task(title="Refactor parser", description="Replace strtok.", metadata={"area": "parser"})
    task.start()

    encoded = json.dumps(task.as_dict())
    decoded = json.loads(encoded)

    assert decoded["id"] == task.id
    assert decoded["title"] == "Refactor parser"
    assert decoded["status"] == "running"
    assert decoded["metadata"] == {"area": "parser"}
    assert decoded["started_at"] is not None


def test_task_report_json_serialization() -> None:
    task = Task(title="Refactor parser", description="Replace strtok.", metadata={"area": "parser"})
    task.start()

    report = task.report()
    encoded = json.dumps(report.as_dict())
    decoded = json.loads(encoded)

    assert isinstance(report, TaskReport)
    assert decoded["id"] == task.id
    assert decoded["title"] == "Refactor parser"
    assert decoded["status"] == "running"
    assert decoded["git_branch"] is None
    assert decoded["files_changed"] == []
    assert decoded["metadata"] == {"area": "parser"}


def test_task_checkpoint_json_serialization() -> None:
    task = Task(title="Refactor parser")

    checkpoint = task.create_checkpoint(
        "after parser edit",
        description="Parser returns one token.",
        metadata={"phase": "edit"},
    )
    encoded = json.dumps(checkpoint.as_dict())
    decoded = json.loads(encoded)

    assert isinstance(checkpoint, TaskCheckpoint)
    assert decoded["task_id"] == task.id
    assert decoded["label"] == "after parser edit"
    assert decoded["description"] == "Parser returns one token."
    assert decoded["git_branch"] is None
    assert decoded["metadata"] == {"phase": "edit"}
    assert task.checkpoints() == [checkpoint]
    assert task.latest_checkpoint() is checkpoint
    assert task.as_dict()["checkpoints"][0]["id"] == checkpoint.id


def test_task_checkpoint_requires_label() -> None:
    task = Task(title="Refactor parser")

    with pytest.raises(ValueError):
        task.create_checkpoint(" ")


def test_agent_owns_tasks_and_tracks_current_task(tmp_path) -> None:
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()

    task = agent.create_task(title="Refactor parser", description="Replace strtok.")

    assert agent.tasks() == [task]
    assert agent.current_task() is task
    assert task.metadata["workspace"]["root"] == str(tmp_path / "workspace")
    assert task.metadata["git_commit_before"] is None

    task.start()
    assert agent.current_task() is task

    task.complete()
    assert agent.current_task() is None
    assert agent.statistics()["tasks"]["count"] == 1


def test_agent_current_task_report(tmp_path) -> None:
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    task = agent.create_task(title="Edit file")

    report = agent.current_task_report()

    assert report is not None
    assert report.id == task.id
    assert report.status == "created"

    task.complete()
    assert agent.current_task_report() is None


def test_task_git_commit_metadata(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("note.txt", "before\n")
    agent.git.add(["note.txt"])
    agent.git.commit("Before task")
    before = agent.git.current_commit()

    task = agent.create_task(title="Update note")

    agent.files.write_text("note.txt", "after\n")
    agent.git.add(["note.txt"])
    agent.git.commit("After task")
    after = agent.git.current_commit()
    task.complete()

    assert task.metadata["git_commit_before"] == before
    assert task.metadata["git_commit_after"] == after


def test_task_report_includes_git_workspace_state(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")

    task = agent.create_task(title="Refactor parser")
    agent.files.replace_text("parser.c", "return 0;", "return 1;")

    report = task.report()
    encoded = json.dumps(report.as_dict())
    decoded = json.loads(encoded)

    assert report.git_branch is not None
    assert "parser.c" in report.git_status
    assert "+    return 1;" in report.git_diff
    assert report.files_changed == ("parser.c",)
    assert decoded["files_changed"] == ["parser.c"]
    assert decoded["metadata"]["git_commit_before"] is not None


def test_task_checkpoint_includes_git_workspace_snapshot(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")

    task = agent.create_task(title="Refactor parser")
    agent.files.replace_text("parser.c", "return 0;", "return 1;")
    first = task.create_checkpoint("after first edit")
    agent.files.replace_text("parser.c", "return 1;", "return 2;")
    second = task.create_checkpoint("after second edit")

    assert first.task_id == task.id
    assert first.git_branch is not None
    assert "+    return 1;" in first.git_diff
    assert "+    return 2;" not in first.git_diff
    assert second.git_branch == first.git_branch
    assert "+    return 2;" in second.git_diff
    assert task.checkpoints() == [first, second]
    assert task.latest_checkpoint() is second


def test_task_compare_checkpoints_json_serialization(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.files.write_text("lexer.c", "int lex(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c", "lexer.c"])
    agent.git.commit("Initial parser")

    task = agent.create_task(title="Refactor parser")
    agent.files.replace_text("parser.c", "return 0;", "return 1;")
    first = task.create_checkpoint("first edit")
    agent.files.replace_text("lexer.c", "return 0;", "return 1;")
    second = task.create_checkpoint("second edit")

    comparison = task.compare_checkpoints(first, second)
    encoded = json.dumps(comparison.as_dict())
    decoded = json.loads(encoded)

    assert isinstance(comparison, TaskCheckpointComparison)
    assert decoded["checkpoint_a"]["id"] == first.id
    assert decoded["checkpoint_b"]["id"] == second.id
    assert decoded["changed_files_a"] == ["parser.c"]
    assert decoded["changed_files_b"] == ["lexer.c", "parser.c"]
    assert decoded["files_added"] == ["lexer.c"]
    assert decoded["files_removed"] == []
    assert decoded["files_changed"] == ["parser.c"]
    assert "+    return 1;" in decoded["diff_a"]
    assert "lexer.c" in decoded["diff_b"]


def test_task_compare_latest_checkpoint_and_checkpoint_ids(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")

    task = agent.create_task(title="Refactor parser")
    agent.files.replace_text("parser.c", "return 0;", "return 1;")
    first = task.create_checkpoint("first edit")
    agent.files.replace_text("parser.c", "return 1;", "return 2;")
    second = task.create_checkpoint("second edit")

    latest = task.compare_latest_checkpoint()
    by_id = task.compare_checkpoints(first.id, second.id)

    assert latest.checkpoint_a_id == first.id
    assert latest.checkpoint_b_id == second.id
    assert by_id.checkpoint_a_label == "first edit"
    assert by_id.checkpoint_b_label == "second edit"


def test_task_compare_checkpoint_validation() -> None:
    task = Task(title="Refactor parser")
    other = Task(title="Other")
    checkpoint = other.create_checkpoint("other checkpoint")

    with pytest.raises(ValueError):
        task.compare_latest_checkpoint()

    with pytest.raises(ValueError):
        task.compare_checkpoints(checkpoint, checkpoint)

    with pytest.raises(ValueError):
        task.compare_checkpoints("missing", "also-missing")


def test_task_restore_plan_without_git_state_is_not_safe() -> None:
    task = Task(title="Refactor parser")
    checkpoint = task.create_checkpoint("manual checkpoint")

    plan = task.plan_restore_checkpoint(checkpoint)
    encoded = json.dumps(plan.as_dict())
    decoded = json.loads(encoded)

    assert isinstance(plan, TaskCheckpointRestorePlan)
    assert decoded["target_checkpoint"]["id"] == checkpoint.id
    assert decoded["current_git_status"] is None
    assert decoded["current_git_diff"] is None
    assert decoded["safe_to_restore"] is False
    assert "current git state is unavailable" in decoded["warnings"]


def test_task_restore_plan_detects_overwritten_current_changes(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace")
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")

    task = agent.create_task(title="Refactor parser")
    agent.files.replace_text("parser.c", "return 0;", "return 1;")
    checkpoint = task.create_checkpoint("first edit")
    agent.files.replace_text("parser.c", "return 1;", "return 2;")
    before = agent.files.read_text("parser.c")

    plan = task.plan_restore_checkpoint(checkpoint)
    decoded = json.loads(json.dumps(plan.as_dict()))
    after = agent.files.read_text("parser.c")

    assert plan.target_checkpoint_id == checkpoint.id
    assert plan.files_would_be_modified == ("parser.c",)
    assert plan.files_would_be_overwritten == ("parser.c",)
    assert plan.safe_to_restore is False
    assert "+    return 1;" in plan.checkpoint_diff
    assert "+    return 2;" in plan.current_git_diff
    assert decoded["files_would_be_modified"] == ["parser.c"]
    assert decoded["files_would_be_overwritten"] == ["parser.c"]
    assert before == after


def test_task_events_are_written_to_jsonl(tmp_path) -> None:
    writer = JsonlWriter(tmp_path / "events.jsonl")
    lab = make_lab(tmp_path / "workspace", log_writer=writer)
    agent = lab.create_agent()

    task = agent.create_task(title="Edit file")
    task.start()
    task.complete()

    events = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]

    assert [event["event_type"] for event in events] == [
        "task_created",
        "task_started",
        "task_completed",
    ]
    assert all(event["task"]["id"] == task.id for event in events)
    assert events[-1]["task"]["status"] == "completed"
