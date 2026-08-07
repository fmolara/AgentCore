from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from agentcore_server import AgentLab, ListEventSink, StreamChunk, TaskExecutor, WriteFileAction
from agentcore_server.executor.actions import ActionResult
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.server import create_app
from agentcore_server.sessions import Session, SessionStore


class FakeStreamingRuntime(Runtime):
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
        completed = None
        for chunk in self.stream(session, prompt, **kwargs):
            if chunk.chunk_type == "completed":
                completed = chunk
        assert completed is not None and completed.metrics is not None
        return GenerationResult(text=completed.text, metrics=completed.metrics)

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        session.add_user_message(prompt)
        metrics = GenerationMetrics(
            prompt_tokens=1,
            generated_tokens=max(1, len(self.response.split())),
            ttft_sec=0.01,
            tokens_per_sec=100.0,
            wall_sec=0.02,
        )
        yield StreamChunk.started(metadata={"prompt_tokens": 1, "runtime": "fake"})
        yield StreamChunk.delta(self.response)
        session.add_assistant_message(self.response)
        yield StreamChunk.completed(text=self.response, metrics=metrics, metadata={"runtime": "fake"})

    def tokenize(self, text_or_messages: Any) -> int:
        return 1


class RequestCancelAction:
    id = "request-cancel-action"

    @property
    def action_type(self) -> str:
        return "request_cancel"

    def execute(self, context) -> ActionResult:
        context.task.request_cancellation("test cancellation")
        return ActionResult.ok(action_id=self.id, action_type=self.action_type)


def make_lab(workspace_root: Path, *, response: str) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = FakeStreamingRuntime(response)
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def plan_json() -> str:
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
    sink = ListEventSink()
    lab = make_lab(tmp_path / "workspace", response=response)
    agent = lab.create_agent(event_sink=sink)
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    task = agent.create_task(title="Edit parser")
    return agent, task, sink


def test_task_executor_cancels_between_actions(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response=plan_json())
    executor = TaskExecutor(agent)

    result = executor.execute(
        task,
        [
            RequestCancelAction(),
            WriteFileAction("after.txt", "must not be written\n"),
        ],
    )

    assert result.status == "cancelled"
    assert task.status.value == "cancelled"
    assert agent.workspace.exists("after.txt") is False
    assert "cancellation.completed" in [event.event_type for event in sink.events]


@pytest.mark.anyio
async def test_http_cancel_before_execution_rejects_execution(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = create_app(
        lab=make_lab(tmp_path, response=plan_json()),
        start_runtime=False,
        warmup=False,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        agent = (await client.post("/v1/agents", json={"workspace_root": str(workspace)})).json()
        task = (
            await client.post(
                f"/v1/agents/{agent['id']}/tasks",
                json={"title": "Edit parser", "description": "Replace parser return value."},
            )
        ).json()["task"]
        async with client.stream(
            "POST",
            f"/v1/tasks/{task['id']}/proposals/stream",
            json={"instruction": "Replace return 0 with return 1 in parser.c", "max_tokens": 128},
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

        event_types = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
        assert event_types[:2] == ["assistant.started", "assistant.delta"]
        assert event_types[-1] == "plan.proposed"

        cancelled = await client.post(f"/v1/tasks/{task['id']}/cancel", json={"reason": "user abort"})
        assert cancelled.status_code == 200
        assert cancelled.json()["task"]["status"] == "cancelled"
        proposal_id = _proposal_id_from_sse(body)
        execute = await client.post(f"/v1/proposals/{proposal_id}/execute")
        assert execute.status_code == 409
        assert execute.json()["detail"] == "task is cancelled"


def _proposal_id_from_sse(body: str) -> str:
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = json.loads(line.removeprefix("data: "))
        if data["event_type"] == "plan.proposed":
            return data["payload"]["proposal"]["id"]
    raise AssertionError("plan.proposed event not found")
