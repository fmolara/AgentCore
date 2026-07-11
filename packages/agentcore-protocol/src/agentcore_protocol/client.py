from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from agentcore_protocol.errors import (
    AgentCoreCompatibilityError,
    AgentCoreConnectionError,
    AgentCoreHTTPError,
    AgentCoreProtocolError,
)
from agentcore_protocol.events import AgentEvent
from agentcore_protocol.schemas import (
    AgentResponse,
    CancelTaskRequest,
    CreateAgentRequest,
    CreateProposalRequest,
    CreateTaskRequest,
    ExecutionResult,
    GitResult,
    HealthResponse,
    PlannerResult,
    ProposalResponse,
    RejectProposalRequest,
    TaskResponse,
)
from agentcore_protocol.sse import parse_sse_lines
from agentcore_protocol.version import API_VERSION, PROTOCOL_VERSION, compatible_protocol


class AgentCoreClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "AgentCoreClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def health(self) -> HealthResponse:
        return HealthResponse.from_dict(self._request("GET", "/health"))

    def check_compatibility(self) -> HealthResponse:
        health = self.health()
        server_protocol = health.protocol_version
        if server_protocol is not None and not compatible_protocol(server_protocol, PROTOCOL_VERSION):
            raise AgentCoreCompatibilityError(
                f"incompatible protocol: client={PROTOCOL_VERSION} server={server_protocol}"
            )
        return health

    def create_agent(
        self,
        *,
        system_prompt: str | None = None,
        workspace_root: str | None = None,
        workspace_mode: str = "read_write",
        workspace_metadata: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> AgentResponse:
        request = CreateAgentRequest(
            system_prompt=system_prompt,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            workspace_metadata=workspace_metadata or {},
            generation_options=generation_options or {},
        )
        return AgentResponse.from_dict(self._request("POST", "/v1/agents", json=request.as_dict()))

    def list_agents(self) -> tuple[AgentResponse, ...]:
        data = self._request("GET", "/v1/agents")
        agents = data.get("agents", [])
        if not isinstance(agents, list):
            raise AgentCoreProtocolError("agents response field must be a list")
        return tuple(AgentResponse.from_dict(agent) for agent in agents if isinstance(agent, dict))

    def get_agent(self, agent_id: str) -> AgentResponse:
        return AgentResponse.from_dict(self._request("GET", f"/v1/agents/{agent_id}"))

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/agents/{agent_id}")

    def create_task(
        self,
        agent_id: str,
        *,
        title: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskResponse:
        request = CreateTaskRequest(title=title, description=description, metadata=metadata or {})
        return TaskResponse.from_dict(self._request("POST", f"/v1/agents/{agent_id}/tasks", json=request.as_dict()))

    def get_task(self, task_id: str) -> TaskResponse:
        return TaskResponse.from_dict(self._request("GET", f"/v1/tasks/{task_id}"))

    def task_report(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/tasks/{task_id}/report").get("report", {})

    def create_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> PlannerResult:
        request = CreateProposalRequest(instruction=instruction, max_tokens=max_tokens, temperature=temperature)
        return PlannerResult.from_dict(
            self._request("POST", f"/v1/tasks/{task_id}/proposals", json=request.as_dict())
        )

    def stream_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> Iterator[AgentEvent]:
        request = CreateProposalRequest(instruction=instruction, max_tokens=max_tokens, temperature=temperature)
        yield from self._stream("POST", f"/v1/tasks/{task_id}/proposals/stream", json=request.as_dict())

    def get_proposal(self, proposal_id: str) -> ProposalResponse:
        return ProposalResponse.from_dict(self._request("GET", f"/v1/proposals/{proposal_id}"))

    def approve_proposal(self, proposal_id: str) -> ProposalResponse:
        return ProposalResponse.from_dict(self._request("POST", f"/v1/proposals/{proposal_id}/approve"))

    def reject_proposal(self, proposal_id: str, reason: str = "rejected by user") -> ProposalResponse:
        request = RejectProposalRequest(reason=reason)
        return ProposalResponse.from_dict(
            self._request("POST", f"/v1/proposals/{proposal_id}/reject", json=request.as_dict())
        )

    def execute_proposal(self, proposal_id: str) -> ExecutionResult:
        return ExecutionResult.from_dict(self._request("POST", f"/v1/proposals/{proposal_id}/execute"))

    def cancel_task(self, task_id: str, reason: str = "cancel requested") -> TaskResponse:
        request = CancelTaskRequest(reason=reason)
        return TaskResponse.from_dict(self._request("POST", f"/v1/tasks/{task_id}/cancel", json=request.as_dict()))

    def git_status(self, agent_id: str) -> GitResult:
        return GitResult.from_dict(self._request("GET", f"/v1/agents/{agent_id}/git/status"))

    def git_diff(self, agent_id: str) -> GitResult:
        return GitResult.from_dict(self._request("GET", f"/v1/agents/{agent_id}/git/diff"))

    def stream_task_events(self, task_id: str) -> Iterator[AgentEvent]:
        yield from self._stream("GET", f"/v1/tasks/{task_id}/events")

    def _stream(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> Iterator[AgentEvent]:
        try:
            with self.client.stream(method, path, json=json) as response:
                self._raise_for_status(response)
                for message in parse_sse_lines(response.iter_lines()):
                    yield message.agent_event()
        except httpx.RequestError as exc:
            raise AgentCoreConnectionError(str(exc)) from exc

    def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, json=json)
        except httpx.RequestError as exc:
            raise AgentCoreConnectionError(str(exc)) from exc
        self._raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AgentCoreProtocolError("response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise AgentCoreProtocolError("response JSON must be an object")
        return data

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body: dict[str, Any] = {}
        message = response.text
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                body = parsed
                detail = parsed.get("detail")
                if isinstance(detail, str):
                    message = detail
        except ValueError:
            pass
        raise AgentCoreHTTPError(response.status_code, message, response=body)


class AsyncAgentCoreClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> "AsyncAgentCoreClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def health(self) -> HealthResponse:
        return HealthResponse.from_dict(await self._request("GET", "/health"))

    async def check_compatibility(self) -> HealthResponse:
        health = await self.health()
        server_protocol = health.protocol_version
        if server_protocol is not None and not compatible_protocol(server_protocol, PROTOCOL_VERSION):
            raise AgentCoreCompatibilityError(
                f"incompatible protocol: client={PROTOCOL_VERSION} server={server_protocol}"
            )
        return health

    async def create_agent(
        self,
        *,
        system_prompt: str | None = None,
        workspace_root: str | None = None,
        workspace_mode: str = "read_write",
        workspace_metadata: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> AgentResponse:
        request = CreateAgentRequest(
            system_prompt=system_prompt,
            workspace_root=workspace_root,
            workspace_mode=workspace_mode,
            workspace_metadata=workspace_metadata or {},
            generation_options=generation_options or {},
        )
        return AgentResponse.from_dict(await self._request("POST", "/v1/agents", json=request.as_dict()))

    async def list_agents(self) -> tuple[AgentResponse, ...]:
        data = await self._request("GET", "/v1/agents")
        agents = data.get("agents", [])
        if not isinstance(agents, list):
            raise AgentCoreProtocolError("agents response field must be a list")
        return tuple(AgentResponse.from_dict(agent) for agent in agents if isinstance(agent, dict))

    async def get_agent(self, agent_id: str) -> AgentResponse:
        return AgentResponse.from_dict(await self._request("GET", f"/v1/agents/{agent_id}"))

    async def delete_agent(self, agent_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/agents/{agent_id}")

    async def create_task(
        self,
        agent_id: str,
        *,
        title: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskResponse:
        request = CreateTaskRequest(title=title, description=description, metadata=metadata or {})
        return TaskResponse.from_dict(
            await self._request("POST", f"/v1/agents/{agent_id}/tasks", json=request.as_dict())
        )

    async def get_task(self, task_id: str) -> TaskResponse:
        return TaskResponse.from_dict(await self._request("GET", f"/v1/tasks/{task_id}"))

    async def task_report(self, task_id: str) -> dict[str, Any]:
        return (await self._request("GET", f"/v1/tasks/{task_id}/report")).get("report", {})

    async def create_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> PlannerResult:
        request = CreateProposalRequest(instruction=instruction, max_tokens=max_tokens, temperature=temperature)
        return PlannerResult.from_dict(
            await self._request("POST", f"/v1/tasks/{task_id}/proposals", json=request.as_dict())
        )

    async def get_proposal(self, proposal_id: str) -> ProposalResponse:
        return ProposalResponse.from_dict(await self._request("GET", f"/v1/proposals/{proposal_id}"))

    async def approve_proposal(self, proposal_id: str) -> ProposalResponse:
        return ProposalResponse.from_dict(await self._request("POST", f"/v1/proposals/{proposal_id}/approve"))

    async def reject_proposal(self, proposal_id: str, reason: str = "rejected by user") -> ProposalResponse:
        request = RejectProposalRequest(reason=reason)
        return ProposalResponse.from_dict(
            await self._request("POST", f"/v1/proposals/{proposal_id}/reject", json=request.as_dict())
        )

    async def execute_proposal(self, proposal_id: str) -> ExecutionResult:
        return ExecutionResult.from_dict(await self._request("POST", f"/v1/proposals/{proposal_id}/execute"))

    async def cancel_task(self, task_id: str, reason: str = "cancel requested") -> TaskResponse:
        request = CancelTaskRequest(reason=reason)
        return TaskResponse.from_dict(
            await self._request("POST", f"/v1/tasks/{task_id}/cancel", json=request.as_dict())
        )

    async def git_status(self, agent_id: str) -> GitResult:
        return GitResult.from_dict(await self._request("GET", f"/v1/agents/{agent_id}/git/status"))

    async def git_diff(self, agent_id: str) -> GitResult:
        return GitResult.from_dict(await self._request("GET", f"/v1/agents/{agent_id}/git/diff"))

    async def stream_task_events(self, task_id: str) -> AsyncIterator[AgentEvent]:
        async for event in self._stream("GET", f"/v1/tasks/{task_id}/events"):
            yield event

    async def stream_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> AsyncIterator[AgentEvent]:
        request = CreateProposalRequest(instruction=instruction, max_tokens=max_tokens, temperature=temperature)
        async for event in self._stream("POST", f"/v1/tasks/{task_id}/proposals/stream", json=request.as_dict()):
            yield event

    async def _stream(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        try:
            async with self.client.stream(method, path, json=json) as response:
                AgentCoreClient._raise_for_status(response)
                async for message in _async_parse_sse(response):
                    yield message.agent_event()
        except httpx.RequestError as exc:
            raise AgentCoreConnectionError(str(exc)) from exc

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = await self.client.request(method, path, json=json)
        except httpx.RequestError as exc:
            raise AgentCoreConnectionError(str(exc)) from exc
        AgentCoreClient._raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AgentCoreProtocolError("response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise AgentCoreProtocolError("response JSON must be an object")
        return data


async def _async_parse_sse(response: httpx.Response):
    from agentcore_protocol.sse import parse_sse_lines

    buffer: list[str] = []
    async for line in response.aiter_lines():
        buffer.append(line)
        if line == "":
            for message in parse_sse_lines(buffer):
                yield message
            buffer.clear()
    if buffer:
        for message in parse_sse_lines(buffer):
            yield message
