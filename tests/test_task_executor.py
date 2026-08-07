from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from agentcore_server import (
    AgentLab,
    CreateCheckpointAction,
    GitDiffAction,
    ReadFileAction,
    ReplaceTextAction,
    TaskExecutor,
    TaskReportAction,
    WriteFileAction,
)
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.logging.writer import JsonlWriter
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


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


def prepare_agent(tmp_path, monkeypatch, *, log_writer: JsonlWriter | None = None):
    set_test_git_identity(monkeypatch)
    lab = make_lab(tmp_path / "workspace", log_writer=log_writer)
    agent = lab.create_agent()
    agent.git.init()
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    agent.git.add(["parser.c"])
    agent.git.commit("Initial parser")
    return agent


def test_task_executor_runs_actions_and_completes_task(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Refactor parser")
    executor = TaskExecutor(agent)
    commit_before = agent.git.current_commit()

    result = executor.execute(
        task,
        [
            CreateCheckpointAction("before edit"),
            ReadFileAction("parser.c"),
            ReplaceTextAction("parser.c", "return 0;", "return 1;"),
            CreateCheckpointAction("after edit"),
            GitDiffAction(),
            TaskReportAction(),
        ],
    )

    assert result.status == "completed"
    assert task.status == "completed"
    assert len(result.actions) == 6
    assert len(task.checkpoints()) == 2
    assert len(task.metadata["actions"]) == 6
    assert "return 1;" in agent.files.read_text("parser.c")
    assert "+    return 1;" in agent.git.diff().stdout
    assert result.report is not None
    assert result.report.status == "completed"
    snapshot = result.actions[-1].data
    assert snapshot["report_kind"] == "intermediate_snapshot"
    assert snapshot["report"]["status"] == "running"
    assert snapshot["report"]["final"] is False
    assert snapshot["report"]["lifecycle_phase"] == "running"
    assert result.report.final is True
    assert result.report.lifecycle_phase == "completed"
    assert agent.git.current_commit() == commit_before


def test_task_executor_stops_on_failure_and_marks_task_failed(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Broken edit")
    executor = TaskExecutor(agent)

    result = executor.execute(
        task,
        [
            ReplaceTextAction("parser.c", "not present", "replacement"),
            WriteFileAction("after.txt", "must not be written\n"),
        ],
    )

    assert result.status == "failed"
    assert task.status == "failed"
    assert result.error is not None
    assert result.actions[0].status == "failed"
    assert len(task.metadata["actions"]) == 1
    assert agent.workspace.exists("after.txt") is False


def test_task_executor_emits_jsonl_action_events(tmp_path, monkeypatch) -> None:
    writer = JsonlWriter(tmp_path / "events.jsonl")
    agent = prepare_agent(tmp_path, monkeypatch, log_writer=writer)
    task = agent.create_task(title="Read parser")
    executor = TaskExecutor(agent)

    executor.execute(task, [ReadFileAction("parser.c")])
    events = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    action_events = [event for event in events if event["event_type"] == "task_action"]

    assert len(action_events) == 1
    assert action_events[0]["task_id"] == task.id
    assert action_events[0]["action"]["action_type"] == "read_file"


def test_task_executor_rejects_completed_task(tmp_path, monkeypatch) -> None:
    agent = prepare_agent(tmp_path, monkeypatch)
    task = agent.create_task(title="Done")
    task.complete()
    executor = TaskExecutor(agent)

    try:
        executor.execute(task, [ReadFileAction("parser.c")])
    except ValueError as exc:
        assert "cannot execute task with status" in str(exc)
    else:
        raise AssertionError("executor accepted completed task")
