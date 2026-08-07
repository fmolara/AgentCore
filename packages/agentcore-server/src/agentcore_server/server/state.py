from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from agentcore_protocol import API_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION
from agentcore_server.api.client import AgentLab
from agentcore_server.agents import Agent
from agentcore_server.executor import PlanProposalStatus, TaskExecutor
from agentcore_server.planning import PlannerResult
from agentcore_server.planning import build_planner
from agentcore_server.server.events import ServerEventSink, TaskEventBus
from agentcore_server.tasks import Task, TaskStatus


@dataclass
class AgentRecord:
    id: str
    agent: Agent


@dataclass
class TaskRecord:
    id: str
    agent_id: str
    task: Task


@dataclass
class ProposalRecord:
    id: str
    task_id: str
    proposal: Any


class AgentCoreServerState:
    def __init__(
        self,
        lab: AgentLab,
        *,
        workspace_root: str | Path | None = None,
        warmup: bool = True,
        start_runtime: bool = True,
    ) -> None:
        self.lab = lab
        self.workspace_root = None if workspace_root is None else Path(workspace_root)
        self.warmup_enabled = warmup
        self.start_runtime = start_runtime
        self.events = TaskEventBus()
        self.planner = build_planner(lab.config)
        self._lock = RLock()
        self._started = False
        self._agents: dict[str, AgentRecord] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._proposals: dict[str, ProposalRecord] = {}
        self._executing_tasks: set[str] = set()

    def start(self) -> None:
        if self._started:
            return
        if self.start_runtime:
            self.lab.start()
            if self.warmup_enabled:
                self.lab.warmup(max_tokens=8)
        self._started = True

    def shutdown(self) -> None:
        if self.start_runtime:
            self.lab.shutdown()
        self._started = False

    def health(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                "agents": len(self._agents),
                "tasks": len(self._tasks),
                "proposals": len(self._proposals),
            }
        return {
            "status": "ok",
            "ready": self.lab.ready(),
            "safe_default_host": "127.0.0.1",
            "api_version": API_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "runtime": self.lab.statistics(),
            "counts": counts,
        }

    def create_agent(
        self,
        *,
        system_prompt: str | None,
        workspace_root: str | None,
        workspace_mode: str,
        workspace_metadata: dict[str, Any],
        generation_options: dict[str, Any],
    ) -> AgentRecord:
        agent_id = uuid4().hex
        root = self._workspace_root(agent_id, workspace_root)
        agent = self.lab.create_agent(
            system_prompt=system_prompt,
            workspace_root=root,
            workspace_mode=workspace_mode,
            workspace_metadata=workspace_metadata,
            generation_options=generation_options,
            event_sink=ServerEventSink(self.events),
        )
        record = AgentRecord(id=agent_id, agent=agent)
        with self._lock:
            self._agents[agent_id] = record
        return record

    def list_agents(self) -> list[AgentRecord]:
        with self._lock:
            return list(self._agents.values())

    def get_agent(self, agent_id: str) -> AgentRecord:
        with self._lock:
            record = self._agents.get(agent_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown agent id")
        return record

    def delete_agent(self, agent_id: str) -> bool:
        with self._lock:
            record = self._agents.pop(agent_id, None)
            if record is None:
                raise HTTPException(status_code=404, detail="unknown agent id")
            task_ids = [task_id for task_id, task in self._tasks.items() if task.agent_id == agent_id]
            for task_id in task_ids:
                self._tasks.pop(task_id, None)
            proposal_ids = [pid for pid, proposal in self._proposals.items() if proposal.task_id in task_ids]
            for proposal_id in proposal_ids:
                self._proposals.pop(proposal_id, None)
        return True

    def create_task(self, agent_id: str, *, title: str, description: str, metadata: dict[str, Any]) -> TaskRecord:
        agent = self.get_agent(agent_id).agent
        task = agent.create_task(title=title, description=description, metadata=metadata)
        record = TaskRecord(id=task.id, agent_id=agent_id, task=task)
        with self._lock:
            self._tasks[task.id] = record
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        with self._lock:
            record = self._tasks.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown task id")
        return record

    def create_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int,
        temperature: float,
    ) -> PlannerResult:
        task_record = self.get_task(task_id)
        agent = self.get_agent(task_record.agent_id).agent
        result = agent.propose_plan(
            task_record.task,
            instruction=instruction,
            planner=self.planner,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if result.proposal is not None:
            with self._lock:
                self._proposals[result.proposal.id] = ProposalRecord(
                    id=result.proposal.id,
                    task_id=task_id,
                    proposal=result.proposal,
                )
        return result

    def stream_proposal(
        self,
        task_id: str,
        *,
        instruction: str,
        max_tokens: int,
        temperature: float,
    ):
        task_record = self.get_task(task_id)
        agent = self.get_agent(task_record.agent_id).agent
        iterator = agent.propose_plan_stream(
            task_record.task,
            instruction=instruction,
            planner=self.planner,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result: PlannerResult | None = None
        while True:
            try:
                event = next(iterator)
            except StopIteration as stop:
                result = stop.value
                break
            yield event
        if result is not None and result.proposal is not None:
            with self._lock:
                self._proposals[result.proposal.id] = ProposalRecord(
                    id=result.proposal.id,
                    task_id=task_id,
                    proposal=result.proposal,
                )

    def get_proposal(self, proposal_id: str) -> ProposalRecord:
        with self._lock:
            record = self._proposals.get(proposal_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown proposal id")
        return record

    def approve_proposal(self, proposal_id: str) -> ProposalRecord:
        record = self.get_proposal(proposal_id)
        task_record = self.get_task(record.task_id)
        agent = self.get_agent(task_record.agent_id).agent
        try:
            agent.approve_proposal(task_record.task, record.proposal)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=f"proposal is {record.proposal.status.value}") from exc
        return record

    def reject_proposal(self, proposal_id: str, reason: str) -> ProposalRecord:
        record = self.get_proposal(proposal_id)
        task_record = self.get_task(record.task_id)
        agent = self.get_agent(task_record.agent_id).agent
        try:
            agent.reject_proposal(task_record.task, record.proposal, reason)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return record

    def execute_proposal(self, proposal_id: str):
        record = self.get_proposal(proposal_id)
        task_record = self.get_task(record.task_id)
        task_id = task_record.id
        proposal = record.proposal
        with self._lock:
            if task_record.task.status == TaskStatus.CANCELLED:
                raise HTTPException(status_code=409, detail="task is cancelled")
            if task_id in self._executing_tasks:
                raise HTTPException(status_code=409, detail="task execution already in progress")
            if proposal.status == PlanProposalStatus.EXECUTED:
                raise HTTPException(status_code=409, detail="proposal already executed")
            if proposal.status == PlanProposalStatus.REJECTED:
                raise HTTPException(status_code=409, detail="proposal was rejected")
            if proposal.approval_requirements and proposal.status != PlanProposalStatus.APPROVED:
                raise HTTPException(status_code=400, detail="proposal requires explicit approval before execution")
            self._executing_tasks.add(task_id)
        try:
            agent = self.get_agent(task_record.agent_id).agent
            return proposal.execute(TaskExecutor(agent), task_record.task)
        finally:
            with self._lock:
                self._executing_tasks.discard(task_id)

    def cancel_task(self, task_id: str, reason: str) -> TaskRecord:
        task_record = self.get_task(task_id)
        agent = self.get_agent(task_record.agent_id).agent
        task = task_record.task
        with self._lock:
            executing = task_id in self._executing_tasks
        try:
            agent.cancel_task(task, reason, executing=executing)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return task_record

    def _workspace_root(self, agent_id: str, requested: str | None) -> Path | None:
        if requested:
            return Path(requested)
        if self.workspace_root is None:
            return None
        return self.workspace_root / agent_id
