from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agentcore_server.api.client import AgentLab
from agentcore_protocol import format_sse
from agentcore_server.server.schemas import (
    CancelTaskRequest,
    CreateAgentRequest,
    CreateProposalRequest,
    CreateTaskRequest,
    RejectProposalRequest,
)
from agentcore_server.server.state import AgentCoreServerState


def create_app(
    *,
    config_path: str | Path | None = None,
    lab: AgentLab | None = None,
    workspace_root: str | Path | None = None,
    warmup: bool = True,
    start_runtime: bool = True,
) -> FastAPI:
    if lab is None:
        if config_path is None:
            raise ValueError("config_path is required when lab is not provided")
        lab = AgentLab.from_config(config_path)
    state = AgentCoreServerState(
        lab,
        workspace_root=workspace_root,
        warmup=warmup,
        start_runtime=start_runtime,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.start()
        try:
            yield
        finally:
            state.shutdown()

    app = FastAPI(title="AgentCore HTTP API", lifespan=lifespan)
    app.state.agentcore = state

    @app.get("/health")
    async def health():
        return state.health()

    @app.post("/v1/agents")
    async def create_agent(request: CreateAgentRequest):
        wire = request.to_protocol()
        record = state.create_agent(
            system_prompt=wire.system_prompt,
            workspace_root=wire.workspace_root,
            workspace_mode=wire.workspace_mode,
            workspace_metadata=wire.workspace_metadata,
            generation_options=wire.generation_options,
        )
        return _agent_response(record.id, record.agent)

    @app.get("/v1/agents")
    async def list_agents():
        return {"agents": [_agent_response(record.id, record.agent) for record in state.list_agents()]}

    @app.get("/v1/agents/{agent_id}")
    async def get_agent(agent_id: str):
        record = state.get_agent(agent_id)
        return _agent_response(record.id, record.agent)

    @app.delete("/v1/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        state.delete_agent(agent_id)
        return {"deleted": True, "agent_id": agent_id}

    @app.post("/v1/agents/{agent_id}/tasks")
    async def create_task(agent_id: str, request: CreateTaskRequest):
        wire = request.to_protocol()
        record = state.create_task(
            agent_id,
            title=wire.title,
            description=wire.description,
            metadata=wire.metadata,
        )
        return {"agent_id": agent_id, "task": record.task.as_dict()}

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        return {"task": state.get_task(task_id).task.as_dict()}

    @app.get("/v1/tasks/{task_id}/report")
    async def get_task_report(task_id: str):
        return {"report": state.get_task(task_id).task.report().as_dict()}

    @app.post("/v1/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, request: CancelTaskRequest):
        wire = request.to_protocol()
        return {"task": state.cancel_task(task_id, wire.reason).task.as_dict()}

    @app.post("/v1/tasks/{task_id}/proposals")
    async def create_proposal(task_id: str, request: CreateProposalRequest):
        wire = request.to_protocol()
        result = state.create_proposal(
            task_id,
            instruction=wire.instruction,
            max_tokens=wire.max_tokens,
            temperature=wire.temperature,
        )
        return result.as_dict()

    @app.post("/v1/tasks/{task_id}/proposals/stream")
    async def stream_proposal(task_id: str, request: CreateProposalRequest):
        wire = request.to_protocol()

        async def generate():
            event_id = 1
            for event in state.stream_proposal(
                task_id,
                instruction=wire.instruction,
                max_tokens=wire.max_tokens,
                temperature=wire.temperature,
            ):
                yield format_sse(event, event_id=event_id)
                event_id += 1

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/v1/proposals/{proposal_id}")
    async def get_proposal(proposal_id: str):
        return {"proposal": state.get_proposal(proposal_id).proposal.as_dict()}

    @app.post("/v1/proposals/{proposal_id}/approve")
    async def approve_proposal(proposal_id: str):
        return {"proposal": state.approve_proposal(proposal_id).proposal.as_dict()}

    @app.post("/v1/proposals/{proposal_id}/reject")
    async def reject_proposal(proposal_id: str, request: RejectProposalRequest):
        wire = request.to_protocol()
        return {"proposal": state.reject_proposal(proposal_id, wire.reason).proposal.as_dict()}

    @app.post("/v1/proposals/{proposal_id}/execute")
    async def execute_proposal(proposal_id: str):
        return state.execute_proposal(proposal_id).as_dict()

    @app.get("/v1/agents/{agent_id}/git/status")
    async def git_status(agent_id: str):
        result = state.get_agent(agent_id).agent.git.status()
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    @app.get("/v1/agents/{agent_id}/git/diff")
    async def git_diff(agent_id: str):
        result = state.get_agent(agent_id).agent.git.diff()
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    @app.get("/v1/tasks/{task_id}/events")
    async def task_events(task_id: str):
        state.get_task(task_id)
        return StreamingResponse(
            state.events.iter_sse_async(task_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def _agent_response(agent_id: str, agent) -> dict:
    return {
        "id": agent_id,
        "session_id": agent.session.id,
        "workspace": agent.workspace.as_dict(),
        "statistics": agent.statistics(),
    }
