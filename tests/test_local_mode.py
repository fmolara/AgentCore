from __future__ import annotations

import ast
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from typing import Any, Iterator
from uuid import uuid4

import httpx
import pytest

from agentcore_server import (
    ActionPlan,
    AgentEvent,
    AgentLab,
    PlanProposal,
    StreamChunk,
    TaskExecutor,
    WriteFileAction,
)
from agentcore_server.executor.actions import ActionResult
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.local import InvalidProposalError, LocalAgentCoreApp, LocalEventSink
from agentcore_server.local.cli import (
    CliUsageError,
    LocalExitCode,
    _resolve_config,
    build_parser,
    run_cli,
)
from agentcore_server.planning import SimpleLLMPlanner
from agentcore_server.planning.llm import PLANNER_SYSTEM_PROMPT
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self, response: str):
        self.response = response
        self.loaded = False
        self.load_calls = 0
        self.shutdown_calls = 0
        self.last_prompt: str | None = None

    def load(self) -> None:
        self.loaded = True
        self.load_calls += 1

    def shutdown(self) -> None:
        self.loaded = False
        self.shutdown_calls += 1

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "fake", "ready": self.ready()}

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        del prompt, max_tokens
        return GenerationResult(text="", metrics=_metrics())

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        completed = None
        for chunk in self.stream(session, prompt, **kwargs):
            if chunk.chunk_type == "completed":
                completed = chunk
        assert completed is not None and completed.metrics is not None
        return GenerationResult(text=completed.text, metrics=completed.metrics)

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        del kwargs
        self.last_prompt = prompt
        session.add_user_message(prompt)
        yield StreamChunk.started(metadata={"runtime": "fake"})
        yield StreamChunk.delta(self.response)
        session.add_assistant_message(self.response)
        yield StreamChunk.completed(text=self.response, metrics=_metrics(), metadata={"runtime": "fake"})

    def tokenize(self, text_or_messages: Any) -> int:
        del text_or_messages
        return 1


class WarmupFailureRuntime(FakeRuntime):
    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        del prompt, max_tokens
        raise RuntimeError("warmup failed")


def _metrics() -> GenerationMetrics:
    return GenerationMetrics(
        prompt_tokens=1,
        generated_tokens=1,
        ttft_sec=0.01,
        tokens_per_sec=100.0,
        wall_sec=0.02,
    )


def _plan(*actions: dict[str, Any]) -> str:
    return json.dumps(
        {
            "title": "Local test plan",
            "description": "Exercise local orchestration.",
            "actions": list(actions),
            "metadata": {"source": "fake"},
        }
    )


def _make_lab(workspace: Path, response: str) -> AgentLab:
    runtime = FakeRuntime(response)
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace)}}
    lab.project_root = workspace.parent
    lab.runtime = runtime
    lab.sessions = SessionStore(runtime.create_session)
    return lab


def _factory(lab: AgentLab):
    def create(_config: str) -> AgentLab:
        return lab

    return create


def _init_repo(workspace: Path) -> str:
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / "parser.c").write_text("int parse(void) { return 0; }\n", encoding="utf-8")
    subprocess.run(["git", "add", "parser.c"], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AgentCore Test",
            "-c",
            "user.email=agentcore@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=workspace,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()


def test_tool_loop_defaults_to_installed_fast_profile(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    fast = profile_dir / "fast.yaml"
    fast.write_text("runtime: sglang\n", encoding="utf-8")
    monkeypatch.setenv("AGENTCORE_PROFILE_DIR", str(profile_dir))
    args = build_parser().parse_args(
        ["--agent", "tool-loop", "--workspace", str(workspace)]
    )

    path, profile = _resolve_config(args)

    assert path == str(fast.resolve())
    assert profile == "fast"


def test_strong_profile_is_explicit_and_legacy_mode_still_requires_config(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    strong = profile_dir / "strong.yaml"
    strong.write_text("runtime: vllm\n", encoding="utf-8")
    monkeypatch.setenv("AGENTCORE_PROFILE_DIR", str(profile_dir))
    args = build_parser().parse_args(
        ["--agent", "tool-loop", "--profile", "strong", "--workspace", str(workspace)]
    )

    path, profile = _resolve_config(args)

    assert path == str(strong.resolve())
    assert profile == "strong"
    planner_args = build_parser().parse_args(["--workspace", str(workspace)])
    with pytest.raises(CliUsageError, match="legacy planner mode"):
        _resolve_config(planner_args)


def test_local_event_sink_writes_one_passive_tool_run_metric(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    metrics_file = tmp_path / "metrics.jsonl"
    sink = LocalEventSink(
        trace_file=trace,
        metrics_file=metrics_file,
        metrics_context={"profile": "fast", "runtime": "sglang", "model": "Qwen3.6-27B"},
    )

    def emit(event_type: str, payload: dict | None = None) -> None:
        sink.emit(
            AgentEvent(
                event_type=event_type,
                summary=event_type,
                task_id="task-1",
                session_id="session-1",
                payload=payload or {},
            )
        )

    emit("agent.loop.started", {"protocol": "qwen"})
    emit("agent.turn.completed", {"metrics": {
        "prompt_tokens": 100,
        "generated_tokens": 20,
        "ttft_sec": 0.2,
        "tokens_per_sec": 40.0,
        "wall_sec": 0.7,
    }})
    emit("tool.call.received", {"tool": "run_check"})
    emit("tool.approved", {"tool": "run_check"})
    emit("tool.completed", {"tool": "run_check", "success": True})
    emit("agent.turn_runway.granted", {
        "base_limit": 1,
        "runway_turns": 2,
        "absolute_limit": 3,
        "current_turn": 1,
        "recent_progress_events": [],
    })
    emit("agent.turn.completed", {"metrics": {
        "prompt_tokens": 80,
        "generated_tokens": 10,
        "ttft_sec": 0.1,
        "tokens_per_sec": 50.0,
        "wall_sec": 0.3,
    }})
    emit("agent.final", {"text": "Done."})
    emit("task.report", {"report": {"status": "completed", "files_changed": ["a.c"]}})
    emit("git.diff", {"diff": "+change\n"})

    records = [json.loads(line) for line in metrics_file.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["profile"] == "fast"
    assert record["protocol"] == "qwen"
    assert record["status"] == "completed"
    assert record["model_turns"] == 2
    assert record["tool_calls"] == 1
    assert record["approvals"] == 1
    assert record["checks_completed"] == 1
    assert record["prompt_tokens"] == 180
    assert record["generated_tokens"] == 30
    assert record["final_response_present"] is True
    assert record["runway_granted"] is True
    assert record["runway_turns"] == 2
    assert record["runway_turns_used"] == 1
    assert record["normal_turn_limit"] == 1
    assert record["absolute_turn_limit"] == 3
    assert record["files_changed"] == ["a.c"]
    assert record["trace_file"] == str(trace.resolve())
    assert len(trace.read_text().splitlines()) == 10


def test_passive_metrics_failure_does_not_interrupt_event_delivery(tmp_path) -> None:
    invalid_metrics_target = tmp_path / "metrics-directory"
    invalid_metrics_target.mkdir()
    rendered = []
    sink = LocalEventSink(
        renderer=rendered.append,
        metrics_file=invalid_metrics_target,
    )

    for event_type, payload in (
        ("agent.loop.started", {"protocol": "qwen"}),
        ("task.report", {"report": {"status": "completed"}}),
        ("git.diff", {"diff": ""}),
    ):
        sink.emit(AgentEvent(event_type, event_type, "task-1", "session-1", payload))

    assert len(rendered) == 3
    assert sink.metrics_errors


def test_local_import_has_no_fastapi_or_agentclient_use() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; "
        f"sys.path[:0] = {[str(root / 'packages' / 'agentcore-server' / 'src'), str(root / 'packages' / 'agentcore-protocol' / 'src')]!r}; "
        "import agentcore_server.local; "
        "assert 'fastapi' not in sys.modules; "
        "assert 'agentclient' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    local_root = (
        root / "packages" / "agentcore-server" / "src" / "agentcore_server" / "local"
    )
    for path in local_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imports.isdisjoint({"fastapi", "httpx", "agentclient", "agentcore_protocol"})


def test_build_prompt_preserves_previous_prompt_text(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("int parse(void);\n", encoding="utf-8")
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "parser.c"}))
    app = LocalAgentCoreApp(lab, workspace=str(workspace))
    task = app.create_task("Inspect parser.c")

    expected = (
        PLANNER_SYSTEM_PROMPT
        + "\nTask:\n"
        + f"- id: {task.id}\n"
        + f"- title: {task.title}\n"
        + f"- description: {task.description}\n"
        + "\nWorkspace:\n"
        + "- files: parser.c\n"
        + "\nGit status:\n"
        + "(not a git repository or clean status)"
        + "\n\nUser instruction:\n"
        + "Inspect parser.c"
    )

    assert app.planner.build_prompt(app.agent, task, "Inspect parser.c") == expected
    assert app.planner._prompt(app.agent, task, "Inspect parser.c") == expected


def test_local_workflow_constructs_no_http_client_or_socket_connection(tmp_path, monkeypatch) -> None:
    import socket

    monkeypatch.setattr(
        httpx.Client,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HTTP client constructed")),
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket connection attempted")),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("return 0;\n", encoding="utf-8")
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "parser.c"}))
    app = LocalAgentCoreApp(lab, workspace=str(workspace))
    app.start(warmup=False)
    task = app.create_task("Inspect parser")

    result = app.propose(task, "Inspect parser")

    assert result.ok
    app.shutdown()


def test_local_app_uses_existing_planner_and_task_executor(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("return 0;\n", encoding="utf-8")
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "0", "new": "1"}),
    )
    app = LocalAgentCoreApp(lab, workspace=str(workspace))
    assert type(app.planner) is SimpleLLMPlanner
    task = app.create_task("Edit parser")
    result = app.propose(task, "Edit parser", stream=True)
    assert result.proposal is not None
    app.approve(task, result.proposal)

    called = False
    original = TaskExecutor.execute_plan

    def spy(self, *args, **kwargs):
        nonlocal called
        called = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TaskExecutor, "execute_plan", spy)
    execution = app.execute(task, result.proposal)

    assert called is True
    assert execution.status == "completed"
    assert (workspace / "parser.c").read_text(encoding="utf-8") == "return 1;\n"


def test_proposal_only_accepts_structurally_valid_poor_plan_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    baseline = _init_repo(workspace)
    trace = tmp_path / "trace.jsonl"
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "src"}))

    code = run_cli(
        [
            "--config",
            "unused.yaml",
            "--workspace",
            str(workspace),
            "--prompt",
            "Fix the parser",
            "--proposal-only",
            "--trace-file",
            str(trace),
            "--no-color",
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.SUCCESS
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() == baseline
    assert subprocess.check_output(["git", "status", "--short"], cwd=workspace, text=True) == ""
    event_types = [json.loads(line)["event_type"] for line in trace.read_text().splitlines()]
    assert event_types[:4] == [
        "task.created",
        "planner.prompt",
        "assistant.started",
        "assistant.delta",
    ]
    assert event_types.count("plan.proposed") == 1


def test_seeded_interactive_execution_requires_explicit_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )

    code = run_cli(
        ["--config", "unused", "--workspace", str(workspace), "--prompt", "Edit parser"],
        stdin=StringIO(""),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.APPROVAL_REQUIRED
    assert "return 0" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert lab.runtime.shutdown_calls == 1


def test_prompt_file_preserves_multiline_content_then_accepts_approval(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    baseline = _init_repo(workspace)
    prompt = "Fix the parser.\n\n  Preserve indentation.\nAccettare A-F.\n"
    prompt_file = tmp_path / "task.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt_file),
            "--no-color",
        ],
        stdin=StringIO("/approve\n"),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.SUCCESS
    assert lab.runtime.last_prompt is not None
    assert lab.runtime.last_prompt.endswith(prompt)
    assert "return 1" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() == baseline


def test_prompt_content_is_preserved_then_accepts_rejection(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    prompt = "Fix parser\nwithout changing tests"
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )

    code = run_cli(
        ["--config", "unused", "--workspace", str(workspace), "--prompt", prompt, "--no-color"],
        stdin=StringIO("/reject not approved\n"),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.PROPOSAL_REJECTED
    assert lab.runtime.last_prompt is not None
    assert lab.runtime.last_prompt.endswith(prompt)
    assert "return 0" in (workspace / "parser.c").read_text(encoding="utf-8")


def test_prompt_file_eof_after_proposal_requires_approval_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    prompt_file = tmp_path / "task.txt"
    prompt_file.write_text("Edit parser\n", encoding="utf-8")
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt_file),
            "--no-color",
        ],
        stdin=StringIO(""),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.APPROVAL_REQUIRED
    assert "return 0" in (workspace / "parser.c").read_text(encoding="utf-8")


def test_eof_before_interactive_task_exits_without_creating_task(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = tmp_path / "trace.jsonl"
    lab = _make_lab(workspace, _plan({"type": "git_diff"}))

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--trace-file",
            str(trace),
            "--no-color",
        ],
        stdin=StringIO(""),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.SUCCESS
    assert trace.read_text(encoding="utf-8") == ""


def test_prompt_and_prompt_file_are_mutually_exclusive(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt_file = tmp_path / "task.txt"
    prompt_file.write_text("Task\n", encoding="utf-8")
    lab = _make_lab(workspace, _plan({"type": "git_diff"}))

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "Task",
            "--prompt-file",
            str(prompt_file),
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.CLI_OR_CONFIG_ERROR
    assert lab.runtime.load_calls == 0


def test_explicit_approval_executes_and_never_commits(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    baseline = _init_repo(workspace)
    trace = tmp_path / "approved.jsonl"
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "Edit parser",
            "--approve",
            "--trace-file",
            str(trace),
            "--no-color",
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.SUCCESS
    assert "return 1" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() == baseline
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    event_types = [record["event_type"] for record in records]
    assert event_types.count("plan.approved") == 1
    assert event_types.count("task.completed") == 1
    assert event_types.count("execution.completed") == 1
    assert [record["sequence"] for record in records] == list(range(1, len(records) + 1))
    report = next(record["payload"]["report"] for record in records if record["event_type"] == "task.report")
    assert report["metadata"]["local_approval"]["approved"] is True
    assert report["final"] is True
    assert report["lifecycle_phase"] == "completed"


def test_interactive_rejection_returns_rejected_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )
    code = run_cli(
        ["--config", "unused", "--workspace", str(workspace), "--no-color"],
        stdin=StringIO("Edit parser\n/reject insufficient plan\n"),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.PROPOSAL_REJECTED
    assert "return 0" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert lab.runtime.shutdown_calls == 1


def test_interactive_abort_cancels_before_execution(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )
    code = run_cli(
        ["--config", "unused", "--workspace", str(workspace), "--no-color"],
        stdin=StringIO("Edit parser\n/abort stop now\n"),
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.TASK_CANCELLED
    assert "return 0" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert lab.runtime.shutdown_calls == 1


def test_local_cancellation_stops_between_atomic_actions(tmp_path) -> None:
    class BlockingAction:
        id = uuid4().hex
        action_type = "write_file"

        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()

        def execute(self, context) -> ActionResult:
            del context
            self.started.set()
            assert self.release.wait(timeout=2)
            return ActionResult.ok(action_id=self.id, action_type=self.action_type)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "unused"}))
    app = LocalAgentCoreApp(lab, workspace=str(workspace))
    app.start(warmup=False)
    task = app.create_task("Cancel between actions")
    blocking = BlockingAction()
    plan = ActionPlan(
        title="Cancellation plan",
        actions=(blocking, WriteFileAction("after.txt", "must not exist\n")),
    )
    proposal = PlanProposal.from_action_plan(task_id=task.id, action_plan=plan)
    app.approve(task, proposal)

    handle = app.execute_async(task, proposal)
    assert blocking.started.wait(timeout=2)
    app.cancel(task, "test abort")
    blocking.release.set()
    handle.wait(timeout=2)

    assert handle.running is False
    assert handle.error is None
    assert handle.result is not None
    assert handle.result.status == "cancelled"
    assert not (workspace / "after.txt").exists()
    app.shutdown()
    assert lab.runtime.shutdown_calls == 1


def test_failed_action_returns_task_failed_and_shuts_down(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "missing.c"}))

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "Read a missing file",
            "--approve",
            "--no-color",
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.TASK_FAILED
    assert lab.runtime.shutdown_calls == 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("not json", LocalExitCode.INVALID_PROPOSAL),
        (_plan(), LocalExitCode.INVALID_PROPOSAL),
        (_plan({"type": "unknown"}), LocalExitCode.INVALID_PROPOSAL),
    ],
)
def test_structurally_invalid_proposal_returns_invalid(tmp_path, response, expected) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _make_lab(workspace, response)

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "Bad plan",
            "--proposal-only",
            "--no-color",
        ],
        lab_factory=_factory(lab),
    )

    assert code == expected
    assert lab.runtime.shutdown_calls == 1


def test_local_report_and_diff_match_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    baseline = _init_repo(workspace)
    sink = LocalEventSink()
    lab = _make_lab(
        workspace,
        _plan({"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}),
    )
    app = LocalAgentCoreApp(lab, workspace=str(workspace), event_sink=sink)
    task = app.create_task("Edit parser")
    result = app.propose(task, "Edit parser")
    assert result.proposal is not None
    app.approve(task, result.proposal)
    app.execute(task, result.proposal)

    report = app.report(task)
    assert "parser.c" in report.git_diff
    assert "return 1" in app.diff()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() == baseline


def test_cli_usage_error_does_not_start_runtime(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _make_lab(workspace, _plan({"type": "read_file", "path": "x"}))

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "x",
            "--proposal-only",
            "--approve",
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.CLI_OR_CONFIG_ERROR
    assert lab.runtime.load_calls == 0
    assert lab.runtime.shutdown_calls == 0


def test_runtime_shutdown_happens_when_warmup_fails(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = WarmupFailureRuntime(_plan({"type": "read_file", "path": "x"}))
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace)}}
    lab.project_root = tmp_path
    lab.runtime = runtime
    lab.sessions = SessionStore(runtime.create_session)

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "x",
            "--proposal-only",
        ],
        lab_factory=_factory(lab),
    )

    assert code == LocalExitCode.RUNTIME_UNAVAILABLE
    assert runtime.load_calls == 1
    assert runtime.shutdown_calls == 1


def test_local_app_raises_invalid_proposal_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _make_lab(workspace, _plan())
    app = LocalAgentCoreApp(lab, workspace=str(workspace))
    task = app.create_task("Poor plan")

    with pytest.raises(InvalidProposalError, match="at least one action"):
        app.propose(task, "Poor plan")

    assert list(workspace.iterdir()) == []
