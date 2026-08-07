from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from agentcore_protocol import API_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION
from agentcore_server.api.client import AgentLab
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.server import create_app
from agentcore_server.sessions import Session, SessionStore
from agentcore_server.workspace import Workspace


class FakeRuntime(Runtime):
    def __init__(self, response: str):
        self.response = response
        self.loaded = True

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


def make_lab(workspace_parent: Path, *, response: str) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_parent)}}
    lab.project_root = workspace_parent
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
                {"type": "git_diff"},
            ],
            "metadata": {"source": "fake-model"},
        }
    )


def prepare_workspace(path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "agentcore-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "agentcore-test@example.invalid")
    workspace = Workspace(path)
    workspace.git.init()
    workspace.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    workspace.git.add(["parser.c"])
    workspace.git.commit("Initial parser")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def make_client(tmp_path, monkeypatch, *, response: str = None):
    response = response or plan_json()
    workspace = tmp_path / "workspace"
    prepare_workspace(workspace, monkeypatch)
    app = create_app(
        lab=make_lab(tmp_path, response=response),
        start_runtime=False,
        warmup=False,
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    agent = (await client.post("/v1/agents", json={"workspace_root": str(workspace)})).json()
    agent_id = agent["id"]
    task = (await client.post(
        f"/v1/agents/{agent_id}/tasks",
        json={"title": "Edit parser", "description": "Replace parser return value."},
    )).json()["task"]
    return client, workspace, agent_id, task["id"]


async def create_proposal(client: httpx.AsyncClient, task_id: str):
    response = await client.post(
        f"/v1/tasks/{task_id}/proposals",
        json={"instruction": "Replace return 0 with return 1 in parser.c", "max_tokens": 128},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "proposed"
    return data["proposal"]


@pytest.mark.anyio
async def test_health_endpoint_and_localhost_safe_default(tmp_path, monkeypatch) -> None:
    client, _, _, _ = await make_client(tmp_path, monkeypatch)

    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["safe_default_host"] == "127.0.0.1"
    assert data["ready"] is True
    assert data["api_version"] == API_VERSION
    assert data["protocol_version"] == PROTOCOL_VERSION
    assert data["schema_version"] == SCHEMA_VERSION


@pytest.mark.anyio
async def test_agent_and_task_creation(tmp_path, monkeypatch) -> None:
    client, _, agent_id, task_id = await make_client(tmp_path, monkeypatch)

    assert (await client.get(f"/v1/agents/{agent_id}")).status_code == 200
    task = (await client.get(f"/v1/tasks/{task_id}")).json()["task"]
    assert task["title"] == "Edit parser"


@pytest.mark.anyio
async def test_valid_plan_proposal_and_unapproved_execution_rejected(tmp_path, monkeypatch) -> None:
    client, _, _, task_id = await make_client(tmp_path, monkeypatch)
    proposal = await create_proposal(client, task_id)

    response = await client.post(f"/v1/proposals/{proposal['id']}/execute")

    assert proposal["approval_requirements"][0]["action_type"] == "replace_text"
    assert response.status_code == 400
    assert response.json()["detail"] == "proposal requires explicit approval before execution"


@pytest.mark.anyio
async def test_approval_then_execution_succeeds_and_git_diff_reports_change(tmp_path, monkeypatch) -> None:
    client, workspace, agent_id, task_id = await make_client(tmp_path, monkeypatch)
    proposal = await create_proposal(client, task_id)

    approved = await client.post(f"/v1/proposals/{proposal['id']}/approve")
    executed = await client.post(f"/v1/proposals/{proposal['id']}/execute")
    diff = await client.get(f"/v1/agents/{agent_id}/git/diff")

    assert approved.status_code == 200
    assert executed.status_code == 200
    assert executed.json()["status"] == "completed"
    assert executed.json()["report"]["final"] is True
    assert executed.json()["report"]["lifecycle_phase"] == "completed"
    assert "return 1;" in (workspace / "parser.c").read_text(encoding="utf-8")
    assert "+    return 1;" in diff.json()["stdout"]


@pytest.mark.anyio
async def test_rejection_prevents_execution(tmp_path, monkeypatch) -> None:
    client, _, _, task_id = await make_client(tmp_path, monkeypatch)
    proposal = await create_proposal(client, task_id)

    rejected = await client.post(f"/v1/proposals/{proposal['id']}/reject", json={"reason": "no"})
    executed = await client.post(f"/v1/proposals/{proposal['id']}/execute")

    assert rejected.status_code == 200
    assert rejected.json()["proposal"]["status"] == "rejected"
    assert executed.status_code == 409


@pytest.mark.anyio
async def test_task_report_endpoint(tmp_path, monkeypatch) -> None:
    client, _, _, task_id = await make_client(tmp_path, monkeypatch)

    response = await client.get(f"/v1/tasks/{task_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["id"] == task_id
    assert response.json()["report"]["final"] is False
    assert response.json()["report"]["lifecycle_phase"] == "created"


@pytest.mark.anyio
async def test_sse_event_ordering(tmp_path, monkeypatch) -> None:
    client, _, _, task_id = await make_client(tmp_path, monkeypatch)
    proposal = await create_proposal(client, task_id)
    await client.post(f"/v1/proposals/{proposal['id']}/approve")
    await client.post(f"/v1/proposals/{proposal['id']}/execute")

    async with client.stream("GET", f"/v1/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])

    event_types = _sse_event_types(body)
    expected = [
        "task.created",
        "plan.proposed",
        "plan.approved",
        "task.started",
        "execution.started",
        "action.started",
        "action.completed",
        "workspace.modified",
        "git.diff",
        "task.completed",
        "execution.completed",
    ]
    positions = [event_types.index(event_type) for event_type in expected]
    assert positions == sorted(positions)


@pytest.mark.anyio
async def test_duplicate_execution_rejected(tmp_path, monkeypatch) -> None:
    client, _, _, task_id = await make_client(tmp_path, monkeypatch)
    proposal = await create_proposal(client, task_id)
    await client.post(f"/v1/proposals/{proposal['id']}/approve")
    assert (await client.post(f"/v1/proposals/{proposal['id']}/execute")).status_code == 200

    response = await client.post(f"/v1/proposals/{proposal['id']}/execute")

    assert response.status_code == 409
    assert response.json()["detail"] == "proposal already executed"


@pytest.mark.anyio
async def test_unknown_identifiers_return_404(tmp_path, monkeypatch) -> None:
    client, _, _, _ = await make_client(tmp_path, monkeypatch)

    assert (await client.get("/v1/agents/missing")).status_code == 404
    assert (await client.get("/v1/tasks/missing")).status_code == 404
    assert (await client.get("/v1/proposals/missing")).status_code == 404


def _sse_event_types(body: str) -> list[str]:
    event_types: list[str] = []
    for line in body.splitlines():
        if line.startswith("event: "):
            event_types.append(line.removeprefix("event: "))
    return event_types
