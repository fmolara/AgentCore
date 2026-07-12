from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentclient.cli import RemoteAgentCLI
from agentclient.commands import parse_command
from agentclient.config import ClientConfig
from agentclient.rendering import Renderer
from agentcore_protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    AgentCoreClient,
    AgentCoreCompatibilityError,
    AgentEvent,
    format_sse,
)


ROOT = Path(__file__).resolve().parents[1]
AGENTCLIENT_ROOT = ROOT / "packages" / "agentclient"


class FakeServer:
    def __init__(self, *, protocol_version: str = PROTOCOL_VERSION) -> None:
        self.protocol_version = protocol_version
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.workspace_root: str | None = None
        self.executed = False
        self.cancelled = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = _json_payload(request)
        self.calls.append((request.method, request.url.path, payload))
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "ready": True,
                    "protocol_version": self.protocol_version,
                    "schema_version": SCHEMA_VERSION,
                },
            )
        if request.method == "POST" and path == "/v1/agents":
            self.workspace_root = None if payload is None else payload.get("workspace_root")
            return httpx.Response(
                200,
                json={"id": "agent-1", "session_id": "session-1", "workspace": {"root": self.workspace_root}},
            )
        if request.method == "POST" and path == "/v1/agents/agent-1/tasks":
            return httpx.Response(200, json={"agent_id": "agent-1", "task": {"id": "task-1", "title": "Patch"}})
        if request.method == "POST" and path == "/v1/tasks/task-1/proposals/stream":
            return httpx.Response(200, text=_proposal_stream())
        if request.method == "POST" and path == "/v1/proposals/proposal-1/approve":
            return httpx.Response(200, json={"proposal": _proposal("approved")})
        if request.method == "POST" and path == "/v1/proposals/proposal-1/reject":
            return httpx.Response(200, json={"proposal": _proposal("rejected")})
        if request.method == "POST" and path == "/v1/proposals/proposal-1/execute":
            self.executed = True
            return httpx.Response(200, json={"task_id": "task-1", "status": "completed", "actions": []})
        if request.method == "POST" and path == "/v1/tasks/task-1/cancel":
            self.cancelled = True
            return httpx.Response(200, json={"task": {"id": "task-1", "status": "cancelled"}})
        if request.method == "GET" and path == "/v1/tasks/task-1/events":
            return httpx.Response(200, text=_task_event_stream())
        if request.method == "GET" and path == "/v1/tasks/task-1/report":
            return httpx.Response(200, json={"report": {"id": "task-1", "status": "completed"}})
        if request.method == "GET" and path == "/v1/agents/agent-1/git/diff":
            return httpx.Response(200, json={"returncode": 0, "stdout": "diff --git a/parser.c b/parser.c", "stderr": ""})
        return httpx.Response(404, json={"detail": f"unexpected path: {request.method} {path}"})


def make_cli(server: FakeServer, *, workspace: str | None = None, server_url: str = "http://testserver"):
    transport = httpx.MockTransport(server.handler)
    http_client = httpx.Client(transport=transport, base_url=server_url)
    client = AgentCoreClient(server_url, client=http_client)
    config = ClientConfig(server_url=server_url, default_workspace=workspace, color=False)
    return RemoteAgentCLI(client, config=config, renderer=Renderer(color=False)), client


def test_agentclient_imports_without_server_dependencies() -> None:
    import agentclient

    assert agentclient.ClientConfig().server_url == "http://127.0.0.1:8080"


def test_agentclient_package_does_not_import_server_or_runtime_dependencies() -> None:
    forbidden_roots = {
        "agentcore_server",
        "fastapi",
        "uvicorn",
        "torch",
        "transformers",
        "sglang",
        "lmdeploy",
    }
    for path in (AGENTCLIENT_ROOT / "src" / "agentclient").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in forbidden_roots, f"{path} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden_roots, f"{path} imports {node.module}"


def test_default_and_remote_endpoints() -> None:
    assert ClientConfig().server_url == "http://127.0.0.1:8080"
    config = ClientConfig().with_overrides(server_url="https://agentcore.example.invalid")
    assert config.server_url == "https://agentcore.example.invalid"


def test_protocol_compatibility_failure() -> None:
    cli, client = make_cli(FakeServer(protocol_version="2.0"))
    with client:
        with pytest.raises(AgentCoreCompatibilityError):
            cli.connect()


def test_create_agent_task_stream_plan_and_execute(capsys) -> None:
    server = FakeServer()
    cli, client = make_cli(server, workspace="/server/workspace")
    with client:
        cli.run(["Replace return 0 with return 1 in parser.c.", "/approve", "/quit"])

    output = capsys.readouterr().out
    assert "Assistant response" in output
    assert "Approval Requirements" in output
    assert server.workspace_root == "/server/workspace"
    assert server.executed is True
    assert [event.event_type for event in cli.state.events][-1] == "execution.completed"


def test_reject_prevents_execution() -> None:
    server = FakeServer()
    cli, client = make_cli(server)
    with client:
        cli.run(["Patch parser.c", "/reject no", "/approve", "/quit"])

    assert server.executed is False
    assert cli.state.rejected is True


def test_abort_sends_cancellation() -> None:
    server = FakeServer()
    cli, client = make_cli(server)
    with client:
        cli.run(["Patch parser.c", "/abort", "/quit"])

    assert server.cancelled is True


def test_invalid_commands_do_not_crash(capsys) -> None:
    server = FakeServer()
    cli, client = make_cli(server)
    with client:
        cli.run(["/not-a-command", "/quit"])

    assert "Unknown command" in capsys.readouterr().out


def test_workspace_path_is_request_data_only(tmp_path) -> None:
    workspace = tmp_path / "must-not-be-created-locally"
    server = FakeServer()
    cli, client = make_cli(server, workspace=str(workspace))
    with client:
        cli.run(["/quit"])

    assert server.workspace_root == str(workspace)
    assert not workspace.exists()


def test_parse_command() -> None:
    assert parse_command("/approve now").name == "/approve"
    assert parse_command("/approve now").argument == "now"
    assert parse_command("Patch parser").name == "instruction"


def _json_payload(request: httpx.Request) -> dict[str, Any] | None:
    if not request.content:
        return None
    import json

    data = json.loads(request.content.decode("utf-8"))
    return data if isinstance(data, dict) else None


def _proposal(status: str = "proposed") -> dict[str, Any]:
    return {
        "id": "proposal-1",
        "title": "Patch parser",
        "status": status,
        "action_plan": {
            "title": "Patch parser",
            "actions": [{"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}],
        },
        "approval_requirements": [
            {"action_index": 0, "action_type": "replace_text", "reason": "mutating action requires approval"}
        ],
    }


def _proposal_stream() -> str:
    events = [
        AgentEvent(event_type="assistant.started", task_id="task-1", summary="Assistant response started"),
        AgentEvent(
            event_type="assistant.delta",
            task_id="task-1",
            summary="Assistant response delta",
            payload={"delta": "{\"actions\": []}"},
        ),
        AgentEvent(event_type="assistant.completed", task_id="task-1", summary="Assistant response completed"),
        AgentEvent(
            event_type="plan.proposed",
            task_id="task-1",
            summary="Plan proposed: Patch parser",
            payload={"proposal": _proposal()},
        ),
    ]
    return "".join(format_sse(event, event_id=index) for index, event in enumerate(events, start=1))


def _task_event_stream() -> str:
    events = [
        AgentEvent(event_type="plan.approved", task_id="task-1", summary="Plan approved"),
        AgentEvent(event_type="execution.started", task_id="task-1", summary="Execution started"),
        AgentEvent(event_type="action.started", task_id="task-1", summary="Starting replace_text"),
        AgentEvent(event_type="action.completed", task_id="task-1", summary="Completed replace_text"),
        AgentEvent(event_type="execution.completed", task_id="task-1", summary="Execution completed"),
    ]
    return "".join(format_sse(event, event_id=index) for index, event in enumerate(events, start=1))
