from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import httpx
import pytest

from agentcore_protocol import (
    API_VERSION,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    AgentCoreClient,
    AgentCoreCompatibilityError,
    AgentCoreHTTPError,
    AgentEvent,
    AsyncAgentCoreClient,
    format_sse,
    parse_sse_lines,
)
from agentcore_protocol.schemas import (
    CreateAgentRequest,
    CreateProposalRequest,
    CreateTaskRequest,
    HealthResponse,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "packages" / "agentcore-protocol"


def test_protocol_dto_serialization_roundtrip() -> None:
    request = CreateAgentRequest(
        system_prompt="You are concise.",
        workspace_root="/tmp/workspace",
        workspace_metadata={"name": "demo"},
    )
    assert request.as_dict()["workspace_metadata"] == {"name": "demo"}
    assert CreateTaskRequest(title="Edit parser").as_dict()["title"] == "Edit parser"
    assert CreateProposalRequest(instruction="Patch parser").as_dict()["temperature"] == 0.0

    health = HealthResponse.from_dict(
        {
            "status": "ok",
            "ready": True,
            "api_version": API_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "runtime": {"runtime_name": "fake"},
        }
    )
    assert health.ready is True
    assert health.protocol_version == PROTOCOL_VERSION
    assert health.runtime["runtime_name"] == "fake"


def test_agent_event_sse_wire_roundtrip() -> None:
    event = AgentEvent(
        event_type="action.started",
        task_id="task-1",
        summary="reading parser.c",
        payload={"path": "parser.c"},
        timestamp="2026-01-01T00:00:00+00:00",
    )

    messages = list(parse_sse_lines(format_sse(event, event_id=7).splitlines()))

    assert len(messages) == 1
    assert messages[0].event == "action.started"
    assert messages[0].event_id == "7"
    assert messages[0].agent_event().as_dict() == event.as_dict()


def test_sync_client_uses_existing_http_api_with_fake_transport() -> None:
    events = [
        AgentEvent(
            event_type="assistant.delta",
            task_id="task-1",
            summary="planner text",
            payload={"delta": "{}"},
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        AgentEvent(
            event_type="plan.proposed",
            task_id="task-1",
            summary="plan proposed",
            payload={
                "proposal": {
                    "id": "proposal-1",
                    "approval_requirements": [
                        {"action_index": 0, "action_type": "replace_text", "reason": "mutating action"}
                    ],
                }
            },
            timestamp="2026-01-01T00:00:01+00:00",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "ready": True,
                    "api_version": API_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "schema_version": SCHEMA_VERSION,
                },
            )
        if request.method == "POST" and path == "/v1/agents":
            return httpx.Response(200, json={"id": "agent-1", "session_id": "session-1"})
        if request.method == "POST" and path == "/v1/agents/agent-1/tasks":
            return httpx.Response(200, json={"agent_id": "agent-1", "task": {"id": "task-1", "title": "Edit"}})
        if path == "/v1/tasks/task-1/proposals/stream":
            return httpx.Response(200, text="".join(format_sse(event, event_id=index) for index, event in enumerate(events)))
        if path == "/v1/proposals/proposal-1/approve":
            return httpx.Response(200, json={"proposal": events[-1].payload["proposal"]})
        if path == "/v1/proposals/proposal-1/execute":
            return httpx.Response(200, json={"task_id": "task-1", "status": "completed", "actions": []})
        if path == "/v1/agents/agent-1/git/diff":
            return httpx.Response(200, json={"returncode": 0, "stdout": "diff", "stderr": ""})
        return httpx.Response(404, json={"detail": f"unexpected path: {path}"})

    transport = httpx.MockTransport(handler)
    with AgentCoreClient("http://testserver", client=httpx.Client(transport=transport, base_url="http://testserver")) as client:
        assert client.check_compatibility().status == "ok"
        agent = client.create_agent(system_prompt="system")
        task = client.create_task(agent.id, title="Edit")
        streamed = list(client.stream_proposal(task.id, instruction="Patch parser"))
        approved = client.approve_proposal("proposal-1")
        executed = client.execute_proposal("proposal-1")
        diff = client.git_diff(agent.id)

    assert agent.id == "agent-1"
    assert task.id == "task-1"
    assert streamed[-1].event_type == "plan.proposed"
    assert approved.id == "proposal-1"
    assert executed.status == "completed"
    assert diff.stdout == "diff"


@pytest.mark.anyio
async def test_async_client_uses_existing_http_api_with_fake_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "ready": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "schema_version": SCHEMA_VERSION,
                },
            )
        if request.method == "POST" and path == "/v1/agents":
            return httpx.Response(200, json={"id": "agent-1", "session_id": "session-1"})
        if request.method == "POST" and path == "/v1/agents/agent-1/tasks":
            return httpx.Response(200, json={"agent_id": "agent-1", "task": {"id": "task-1", "title": "Edit"}})
        if path == "/v1/agents/agent-1/git/status":
            return httpx.Response(200, json={"returncode": 0, "stdout": "clean", "stderr": ""})
        return httpx.Response(404, json={"detail": f"unexpected path: {path}"})

    transport = httpx.MockTransport(handler)
    async with AsyncAgentCoreClient(
        "http://testserver",
        client=httpx.AsyncClient(transport=transport, base_url="http://testserver"),
    ) as client:
        assert (await client.check_compatibility()).status == "ok"
        agent = await client.create_agent(system_prompt="system")
        task = await client.create_task(agent.id, title="Edit")
        status = await client.git_status(agent.id)

    assert agent.id == "agent-1"
    assert task.id == "task-1"
    assert status.stdout == "clean"


def test_client_normalizes_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"detail": "approval required"}))
    client = AgentCoreClient("http://testserver", client=httpx.Client(transport=transport, base_url="http://testserver"))

    with pytest.raises(AgentCoreHTTPError) as exc_info:
        client.execute_proposal("proposal-1")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "approval required"


def test_protocol_major_version_mismatch_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"status": "ok", "ready": True, "protocol_version": "2.0", "schema_version": SCHEMA_VERSION},
        )
    )
    client = AgentCoreClient("http://testserver", client=httpx.Client(transport=transport, base_url="http://testserver"))

    with pytest.raises(AgentCoreCompatibilityError):
        client.check_compatibility()


def test_protocol_package_has_only_allowed_dependencies() -> None:
    pyproject = tomllib.loads((PROTOCOL_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert dependencies == ["httpx>=0.28"]


def test_protocol_package_does_not_import_server_or_runtime_dependencies() -> None:
    forbidden_roots = {
        "a100_agent_lab",
        "fastapi",
        "uvicorn",
        "torch",
        "transformers",
        "sglang",
        "lmdeploy",
    }

    for path in (PROTOCOL_ROOT / "src" / "agentcore_protocol").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    assert imported.split(".", 1)[0] not in forbidden_roots, f"{path} imports {imported}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
                assert imported.split(".", 1)[0] not in forbidden_roots, f"{path} imports {imported}"


def test_protocol_package_imports_without_server_modules() -> None:
    import agentcore_protocol

    assert agentcore_protocol.PROTOCOL_VERSION == PROTOCOL_VERSION
    assert "fastapi" not in json.dumps(agentcore_protocol.__all__).lower()
