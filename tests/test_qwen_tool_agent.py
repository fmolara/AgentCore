from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator
from time import monotonic, sleep

import pytest

from agentcore_server import AgentLab, ListEventSink
from agentcore_server.generation import AssistantTurn, ToolCall, ToolResult, ToolTurnChunk
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.generation.stream import StreamChunk
from agentcore_server.runtime.base import Runtime
from agentcore_server.runtime.server_process import RuntimeStreamError
from agentcore_server.sessions import Session, SessionStore
from agentcore_server.tool_agent import QwenToolAgent, QwenToolAgentLimits, ToolSteeringInbox
from agentcore_server.tool_agent.tools import encode_tool_result


def metrics() -> GenerationMetrics:
    return GenerationMetrics(10, 5, 0.01, 50.0, 0.1)


def call(call_id: str, index: int, name: str, arguments: dict[str, Any]) -> ToolCall:
    raw = json.dumps(arguments)
    return ToolCall(call_id, index, name, raw, arguments)


def malformed_call(call_id: str, name: str, raw: str) -> ToolCall:
    return ToolCall(call_id, 0, name, raw, None, "invalid tool arguments")


def turn(*calls: ToolCall, text: str = "", finish: str | None = None) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=tuple(calls),
        finish_reason=finish or ("tool_calls" if calls else "stop"),
        metrics=metrics(),
    )


class ScriptedToolRuntime(Runtime):
    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = deque(turns)
        self.loaded = True
        self.requests: list[list[dict[str, Any]]] = []
        self.schemas: list[list[dict[str, Any]]] = []

    def load(self) -> None:
        self.loaded = True

    def shutdown(self) -> None:
        self.loaded = False

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "scripted-qwen", "ready": self.loaded}

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        del prompt, max_tokens
        return GenerationResult("", metrics())

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        del session, prompt, kwargs
        raise AssertionError("plain generation must not be used by QwenToolAgent")

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        del session, prompt, kwargs
        raise AssertionError("plain streaming must not be used by QwenToolAgent")

    def stream_tool_turn(
        self,
        session: Session,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ToolTurnChunk]:
        del kwargs
        self.requests.append(session.transcript())
        self.schemas.append(tools)
        current = self.turns.popleft()
        yield ToolTurnChunk.started(metadata={"runtime": "scripted-qwen"})
        if current.text:
            yield ToolTurnChunk.text(current.text)
        if current.tool_calls:
            session.add_assistant_tool_message(current.text, current.tool_calls)
        else:
            session.add_assistant_message(current.text)
        yield ToolTurnChunk.completed(current)

    def tokenize(self, text_or_messages: Any) -> int:
        return len(json.dumps(text_or_messages))

    def statistics(self) -> dict[str, Any]:
        return self.health()


@dataclass
class ScriptedApproval:
    decisions: deque[bool]

    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = deque(decisions)
        self.requests = []

    def request(self, request) -> bool:
        self.requests.append(request)
        return self.decisions.popleft()


def make_lab(workspace: Path, runtime: ScriptedToolRuntime, checks=None) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {
        "runtime": "fake",
        "workspace": {"root": str(workspace), "checks": checks or {}},
    }
    lab.project_root = workspace.parent
    lab.runtime = runtime
    lab.sessions = SessionStore(runtime.create_session)
    return lab


def prepare(tmp_path, turns, *, decisions=None, checks=None, limits=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = ScriptedToolRuntime(turns)
    lab = make_lab(workspace, runtime, checks=checks)
    sink = ListEventSink()
    agent = lab.create_agent(event_sink=sink)
    approval = ScriptedApproval(decisions or [])
    tool_agent = QwenToolAgent(
        agent,
        approval_gateway=approval,
        limits=limits,
    )
    task = agent.create_task(title="Tool task", description="test")
    return workspace, runtime, agent, tool_agent, task, approval, sink


def test_session_serializes_native_assistant_and_tool_messages() -> None:
    session = Session(system_prompt="system")
    native_call = call("call_1", 0, "read_file", {"path": "src/a.c"})
    session.add_user_message("inspect")
    session.add_assistant_tool_message("", (native_call,))
    session.add_tool_result(ToolResult("call_1", "read_file", True, '{"success":true}'))

    transcript = session.transcript()

    assert transcript[2]["tool_calls"][0]["id"] == "call_1"
    assert transcript[2]["tool_calls"][0]["function"]["arguments"] == {"path": "src/a.c"}
    assert transcript[3] == {
        "role": "tool",
        "content": '{"success":true}',
        "tool_call_id": "call_1",
    }


def test_tool_result_byte_limit_is_strict_and_preserves_digest() -> None:
    content, truncated = encode_tool_result(
        {"success": True, "content": "x" * 10000},
        max_bytes=512,
    )

    assert truncated is True
    assert len(content.encode("utf-8")) <= 512
    assert json.loads(content)["result_truncated"] is True


def test_read_result_is_appended_and_model_continues(tmp_path) -> None:
    turns = [
        turn(call("read_1", 0, "read_file", {"path": "main.c"})),
        turn(text="Inspected main.c; no change needed."),
    ]
    workspace, runtime, _, tool_agent, task, _, _ = prepare(tmp_path, turns)
    (workspace / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")

    result = tool_agent.run(task, "Inspect main.c")

    assert result.status == "completed"
    assert result.turns == 2
    assert runtime.requests[1][-1]["role"] == "tool"
    assert runtime.requests[1][-1]["tool_call_id"] == "read_1"
    payload = json.loads(runtime.requests[1][-1]["content"])
    assert payload["path"] == "main.c"
    assert "int main" in payload["content"]


def test_multiple_calls_preserve_index_order_and_result_ids(tmp_path) -> None:
    turns = [
        turn(
            call("read_b", 1, "read_file", {"path": "b.c"}),
            call("read_a", 0, "read_file", {"path": "a.c"}),
        ),
        turn(text="Done."),
    ]
    workspace, runtime, _, tool_agent, task, _, _ = prepare(tmp_path, turns)
    (workspace / "a.c").write_text("a\n", encoding="utf-8")
    (workspace / "b.c").write_text("b\n", encoding="utf-8")

    result = tool_agent.run(task, "Read both")

    assert [item.tool_call_id for item in result.tool_results] == ["read_a", "read_b"]
    assert [message["tool_call_id"] for message in runtime.requests[1][-2:]] == [
        "read_a",
        "read_b",
    ]


def test_read_continuation_metadata_is_returned(tmp_path) -> None:
    turns = [
        turn(call("read_1", 0, "read_file", {"path": "main.c", "max_lines": 2})),
        turn(text="Read the first page."),
    ]
    workspace, runtime, _, tool_agent, task, _, _ = prepare(tmp_path, turns)
    (workspace / "main.c").write_text("one\ntwo\nthree\n", encoding="utf-8")

    tool_agent.run(task, "Read")

    payload = json.loads(runtime.requests[1][-1]["content"])
    assert payload["start_line"] == 1
    assert payload["end_line"] == 2
    assert payload["continuation_start_line"] == 3
    assert payload["truncated"] is True


def test_malformed_arguments_never_execute_and_are_returned(tmp_path) -> None:
    turns = [
        turn(malformed_call("bad_1", "write_file", '{"path":')),
        turn(text="The malformed call was not executed."),
    ]
    workspace, runtime, _, tool_agent, task, approval, sink = prepare(
        tmp_path, turns, decisions=[True]
    )

    result = tool_agent.run(task, "Write a file")

    assert result.status == "completed"
    assert list(workspace.iterdir()) == []
    assert approval.requests == []
    assert json.loads(runtime.requests[1][-1]["content"])["kind"] == "validation"
    assert "tool.validation.failed" in [event.event_type for event in sink.events]


def test_workspace_escape_is_fatal_and_no_tool_executes(tmp_path) -> None:
    turns = [turn(
        call("escape", 0, "read_file", {"path": "../secret"}),
        call("write", 1, "write_file", {"path": "created.txt", "content": "no"}),
    )]
    workspace, _, _, tool_agent, task, approval, sink = prepare(
        tmp_path, turns, decisions=[True]
    )

    result = tool_agent.run(task, "Escape")

    assert result.status == "failed"
    assert "escapes workspace root" in (result.error or "")
    assert not (workspace / "created.txt").exists()
    assert approval.requests == []
    failed = [event for event in sink.events if event.event_type == "tool.validation.failed"]
    assert failed and failed[0].payload["fatal"] is True


def test_exact_edit_requires_one_approval_and_mutates_once(tmp_path) -> None:
    turns = [
        turn(call("edit_1", 0, "edit", {"path": "main.c", "old": "return 0;", "new": "return 1;"})),
        turn(call("diff_1", 0, "git_diff", {})),
        turn(text="Changed the return value and inspected the diff."),
    ]
    workspace, _, _, tool_agent, task, approval, _ = prepare(
        tmp_path, turns, decisions=[True]
    )
    (workspace / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "main.c"], cwd=workspace, check=True)

    result = tool_agent.run(task, "Change return value")

    assert result.status == "completed"
    assert (workspace / "main.c").read_text() == "int main(void) { return 1; }\n"
    assert [request.call.id for request in approval.requests] == ["edit_1"]
    assert result.report.final is True


def test_rejected_edit_returns_result_without_mutation(tmp_path) -> None:
    turns = [
        turn(call("edit_1", 0, "edit", {"path": "main.c", "old": "0", "new": "1"})),
        turn(text="The operator rejected the edit; no changes were made."),
    ]
    workspace, runtime, _, tool_agent, task, approval, sink = prepare(
        tmp_path, turns, decisions=[False]
    )
    (workspace / "main.c").write_text("0\n", encoding="utf-8")

    result = tool_agent.run(task, "Edit")

    assert result.status == "completed"
    assert (workspace / "main.c").read_text() == "0\n"
    assert len(approval.requests) == 1
    assert json.loads(runtime.requests[1][-1]["content"])["rejected"] is True
    assert "tool.rejected" in [event.event_type for event in sink.events]


def test_write_file_requires_its_own_approval(tmp_path) -> None:
    turns = [
        turn(call("write_1", 0, "write_file", {"path": "new.txt", "content": "created\n"})),
        turn(text="Created the new file."),
    ]
    workspace, _, _, tool_agent, task, approval, _ = prepare(
        tmp_path, turns, decisions=[True]
    )

    result = tool_agent.run(task, "Create a file")

    assert result.status == "completed"
    assert (workspace / "new.txt").read_text() == "created\n"
    assert [request.call.id for request in approval.requests] == ["write_1"]


@pytest.mark.parametrize("content,error", [
    ("no match\n", "not found"),
    ("target target\n", "ambiguous"),
])
def test_failed_exact_edit_never_mutates_and_model_can_recover(tmp_path, content, error) -> None:
    turns = [
        turn(call("edit_1", 0, "edit", {"path": "main.c", "old": "target", "new": "changed"})),
        turn(text="I received the edit failure and stopped safely."),
    ]
    workspace, runtime, _, tool_agent, task, _, _ = prepare(
        tmp_path, turns, decisions=[True]
    )
    (workspace / "main.c").write_text(content, encoding="utf-8")

    result = tool_agent.run(task, "Edit")

    assert result.status == "completed"
    assert (workspace / "main.c").read_text() == content
    assert error in json.loads(runtime.requests[1][-1]["content"])["error"]


def test_approval_is_not_reused_for_second_edit(tmp_path) -> None:
    turns = [
        turn(call("edit_1", 0, "edit", {"path": "main.c", "old": "a", "new": "b"})),
        turn(call("edit_2", 0, "edit", {"path": "main.c", "old": "b", "new": "c"})),
        turn(text="One edit was accepted and one rejected."),
    ]
    workspace, _, _, tool_agent, task, approval, _ = prepare(
        tmp_path, turns, decisions=[True, False]
    )
    (workspace / "main.c").write_text("a\n", encoding="utf-8")

    tool_agent.run(task, "Two changes")

    assert (workspace / "main.c").read_text() == "b\n"
    assert [request.call.id for request in approval.requests] == ["edit_1", "edit_2"]


def test_cancellation_while_waiting_for_approval_prevents_mutation(tmp_path) -> None:
    turns = [turn(call("edit_1", 0, "edit", {"path": "main.c", "old": "a", "new": "b"}))]
    workspace, _, agent, _, task, _, _ = prepare(tmp_path, turns)
    (workspace / "main.c").write_text("a\n", encoding="utf-8")

    class CancellingApproval:
        def request(self, request):
            del request
            agent.cancel_task(task, "cancelled during approval", executing=True)
            return True

    tool_agent = QwenToolAgent(agent, approval_gateway=CancellingApproval())
    result = tool_agent.run(task, "Edit")

    assert result.status == "cancelled"
    assert (workspace / "main.c").read_text() == "a\n"


def test_failed_check_returns_to_qwen_then_corrective_edit_succeeds(tmp_path) -> None:
    check = {
        "test": {
            "argv": [
                "python3",
                "-c",
                "import pathlib,sys; sys.exit(pathlib.Path('state.txt').read_text().strip() != 'good')",
            ]
        }
    }
    turns = [
        turn(call("check_1", 0, "run_check", {"check": "test"})),
        turn(call("edit_1", 0, "edit", {"path": "state.txt", "old": "bad", "new": "good"})),
        turn(call("check_2", 0, "run_check", {"check": "test"})),
        turn(text="Corrected the file after the failed check; the next check passed."),
    ]
    workspace, runtime, _, tool_agent, task, approval, _ = prepare(
        tmp_path, turns, decisions=[True, True, True], checks=check
    )
    (workspace / "state.txt").write_text("bad\n", encoding="utf-8")

    result = tool_agent.run(task, "Make test pass")

    assert result.status == "completed"
    first = json.loads(runtime.requests[1][-1]["content"])["check"]
    assert first["returncode"] == 1
    assert (workspace / "state.txt").read_text() == "good\n"
    last_check = json.loads(runtime.requests[3][-1]["content"])["check"]
    assert last_check["returncode"] == 0
    assert len(approval.requests) == 3


def test_unknown_check_and_arbitrary_shell_are_not_exposed(tmp_path) -> None:
    turns = [
        turn(call("bad", 0, "run_check", {"check": "shell"})),
        turn(text="Unknown checks are unavailable."),
    ]
    _, runtime, _, tool_agent, task, approval, _ = prepare(tmp_path, turns)

    result = tool_agent.run(task, "Run arbitrary command")

    assert result.status == "completed"
    assert approval.requests == []
    assert "unknown configured check" in json.loads(runtime.requests[1][-1]["content"])["error"]
    names = {schema["function"]["name"] for schema in runtime.schemas[0]}
    assert names == {
        "list_directory", "search_files", "read_file", "git_status", "git_diff",
        "edit", "write_file", "run_check",
    }
    assert "shell" not in names and "bash" not in names
    assert all(schema["function"]["strict"] is False for schema in runtime.schemas[0])


def test_one_steering_message_reaches_next_turn(tmp_path) -> None:
    inbox = ToolSteeringInbox()
    turns = [
        turn(call("read_1", 0, "read_file", {"path": "main.c"})),
        turn(text="Followed the steering message."),
    ]
    workspace, runtime, agent, _, task, approval, _ = prepare(tmp_path, turns)
    (workspace / "main.c").write_text("x\n", encoding="utf-8")
    inbox.queue("Also preserve formatting.")
    tool_agent = QwenToolAgent(agent, approval_gateway=approval, steering=inbox)

    tool_agent.run(task, "Inspect")

    assert runtime.requests[1][-1] == {"role": "user", "content": "Also preserve formatting."}


def test_loop_limit_stops_safely_without_action_plan_or_commit(tmp_path, monkeypatch) -> None:
    from agentcore_server.executor.plan import ActionPlan
    from agentcore_server.executor.proposal import PlanProposal

    monkeypatch.setattr(ActionPlan, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(PlanProposal, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    turns = [turn(call("read_1", 0, "read_file", {"path": "main.c"}))]
    limits = QwenToolAgentLimits(max_model_turns=1)
    workspace, _, _, tool_agent, task, _, sink = prepare(tmp_path, turns, limits=limits)
    (workspace / "main.c").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    before = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=workspace, text=True, stderr=subprocess.DEVNULL).strip() if (workspace / ".git/HEAD").exists() and subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=workspace, capture_output=True).returncode == 0 else "0"

    result = tool_agent.run(task, "Inspect forever")

    assert result.status == "failed"
    assert "max_model_turns" in (result.error or "")
    after = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=workspace, text=True, capture_output=True)
    assert (after.stdout.strip() if after.returncode == 0 else "0") == before
    types = [event.event_type for event in sink.events]
    assert "agent.loop.limit_reached" in types
    assert types.count("task.report") == 1
    assert sink.events[-2].event_type == "task.report"
    assert sink.events[-1].event_type == "git.diff"


def test_context_failure_is_terminal_and_reports_exact_capacity(tmp_path) -> None:
    class ContextFailureRuntime(ScriptedToolRuntime):
        def stream_tool_turn(self, session, tools, **kwargs):
            del session, tools
            assert kwargs["context_safety_margin_tokens"] == 128
            assert kwargs["minimum_output_tokens"] == 256
            yield ToolTurnChunk.started(metadata={
                "runtime": "sglang",
                "context_limit": 4096,
                "exact_prompt_tokens": 3800,
                "configured_max_tokens": 2048,
                "safety_margin_tokens": 128,
                "available_tokens": 168,
                "effective_max_tokens": 168,
                "minimum_output_tokens": 256,
                "sufficient": False,
            })
            raise RuntimeError(
                "Qwen tool turn has insufficient context capacity: available=168, minimum=256"
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = ContextFailureRuntime([])
    lab = make_lab(workspace, runtime)
    sink = ListEventSink()
    agent = lab.create_agent(event_sink=sink)
    task = agent.create_task(title="Tool task", description="test")
    tool_agent = QwenToolAgent(agent, approval_gateway=ScriptedApproval([]))

    result = tool_agent.run(task, "Inspect")

    assert result.status == "failed"
    assert result.final_text == ""
    assert result.report.final is True
    assert result.report.lifecycle_phase == "failed"
    assert "insufficient context capacity" in (result.report.failure_reason or "")
    event_types = [event.event_type for event in sink.events]
    assert "agent.context.insufficient" in event_types
    assert "agent.turn.failed" in event_types
    assert "agent.final" not in event_types
    assert event_types.count("task.report") == 1


def test_stream_error_is_not_treated_as_empty_final_answer(tmp_path) -> None:
    class StreamFailureRuntime(ScriptedToolRuntime):
        def stream_tool_turn(self, session, tools, **kwargs):
            del session, tools, kwargs
            raise RuntimeStreamError(
                "sglang",
                "Requested token count exceeds the model context length",
                error_type="BadRequestError",
                code=400,
            )
            yield  # pragma: no cover

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = StreamFailureRuntime([])
    lab = make_lab(workspace, runtime)
    sink = ListEventSink()
    agent = lab.create_agent(event_sink=sink)
    task = agent.create_task(title="Tool task", description="test")
    tool_agent = QwenToolAgent(agent, approval_gateway=ScriptedApproval([]))

    result = tool_agent.run(task, "Inspect")

    assert result.status == "failed"
    assert result.final_text == ""
    assert result.report.final is True
    assert "BadRequestError" in (result.report.failure_reason or "")
    event_types = [event.event_type for event in sink.events]
    assert "agent.turn.failed" in event_types
    assert "assistant.completed" not in event_types
    assert "agent.final" not in event_types
    assert event_types.count("task.report") == 1


def test_local_cli_prompt_file_approves_only_pending_tool_call(tmp_path) -> None:
    from agentcore_server.local.cli import LocalExitCode, run_cli

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.c").write_text("a\n", encoding="utf-8")
    prompt = tmp_path / "task.txt"
    prompt.write_text("Change a to b.\nPreserve this final newline.\n", encoding="utf-8")
    runtime = ScriptedToolRuntime([
        turn(call("edit_cli", 0, "edit", {"path": "main.c", "old": "a", "new": "b"})),
        turn(text="Changed the file."),
    ])
    lab = make_lab(workspace, runtime)

    class ScriptedTerminal:
        def __init__(self):
            self.calls = 0

        def readline(self):
            self.calls += 1
            if self.calls == 1:
                return "/approve\n"
            deadline = monotonic() + 2
            while len(runtime.requests) < 2 and monotonic() < deadline:
                sleep(0.01)
            sleep(0.05)
            return "/quit\n"

    code = run_cli(
        [
            "--config", "ignored.yaml",
            "--workspace", str(workspace),
            "--agent", "qwen-tools",
            "--prompt-file", str(prompt),
            "--no-warmup",
            "--no-color",
        ],
        stdin=ScriptedTerminal(),
        lab_factory=lambda path: lab,
    )

    assert code == LocalExitCode.SUCCESS
    assert (workspace / "main.c").read_text() == "b\n"
    assert runtime.requests[0][1]["content"] == prompt.read_text(encoding="utf-8")


def test_local_cli_rejects_planner_options_with_qwen_agent(tmp_path) -> None:
    from agentcore_server.local.cli import LocalExitCode, run_cli

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = run_cli([
        "--config", "ignored.yaml", "--workspace", str(workspace),
        "--agent", "qwen-tools", "--proposal-only", "--prompt", "task",
    ])
    assert code == LocalExitCode.CLI_OR_CONFIG_ERROR

    code = run_cli([
        "--config", "ignored.yaml", "--workspace", str(workspace),
        "--agent", "qwen-tools", "--planner", "simple", "--prompt", "task",
    ])
    assert code == LocalExitCode.CLI_OR_CONFIG_ERROR
