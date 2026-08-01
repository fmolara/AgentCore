from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread
from time import monotonic
from typing import Any

from agentcore_server.agents import Agent
from agentcore_server.api.client import AgentLab
from agentcore_server.events import EventSink
from agentcore_server.tasks import Task, TaskReport
from agentcore_server.tool_agent import (
    QWEN_TOOL_AGENT_SYSTEM_PROMPT,
    QwenToolAgent,
    QwenToolAgentLimits,
    QwenToolRunResult,
    ToolApprovalRequest,
    ToolSteeringInbox,
)


class InteractiveToolApprovalGateway:
    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: ToolApprovalRequest | None = None
        self._decision: bool | None = None
        self._closed = False

    @property
    def pending(self) -> ToolApprovalRequest | None:
        with self._condition:
            return self._pending

    def request(self, request: ToolApprovalRequest) -> bool:
        with self._condition:
            if self._closed:
                return False
            if self._pending is not None:
                raise RuntimeError("another tool approval is already pending")
            self._pending = request
            self._decision = None
            self._condition.notify_all()
            while self._decision is None and not self._closed:
                self._condition.wait()
            decision = False if self._decision is None else self._decision
            self._pending = None
            self._decision = None
            self._condition.notify_all()
            return decision

    def decide(self, approved: bool, *, tool_call_id: str | None = None) -> None:
        with self._condition:
            if self._pending is None:
                raise ValueError("no tool approval is pending")
            if tool_call_id is not None and self._pending.call.id != tool_call_id:
                raise ValueError("approval tool-call ID does not match the pending call")
            if self._decision is not None:
                raise ValueError("pending tool call already has a decision")
            self._decision = approved
            self._condition.notify_all()

    def wait_for_pending(self, timeout: float | None = None) -> ToolApprovalRequest | None:
        with self._condition:
            deadline = None if timeout is None else monotonic() + timeout
            while self._pending is None and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._pending

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass
class LocalQwenToolHandle:
    task: Task
    thread: Thread
    done: Event
    result: QwenToolRunResult | None = None
    error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def wait(self, timeout: float | None = None) -> QwenToolRunResult | None:
        self.thread.join(timeout)
        return self.result


class LocalQwenToolApp:
    """Local composition for the topology-neutral Qwen native tool loop."""

    def __init__(
        self,
        lab: AgentLab,
        *,
        workspace: str,
        system_prompt: str | None = None,
        event_sink: EventSink | None = None,
        approval_gateway: InteractiveToolApprovalGateway | None = None,
    ) -> None:
        self.lab = lab
        self.event_sink = event_sink
        self.approval_gateway = approval_gateway or InteractiveToolApprovalGateway()
        self.steering = ToolSteeringInbox()
        self.agent: Agent = lab.create_agent(
            system_prompt=system_prompt or QWEN_TOOL_AGENT_SYSTEM_PROMPT,
            workspace_root=workspace,
            event_sink=event_sink,
        )
        self.tool_agent = QwenToolAgent(
            self.agent,
            approval_gateway=self.approval_gateway,
            limits=QwenToolAgentLimits.from_config(lab.config.get("tool_agent")),
            steering=self.steering,
        )
        self._started = False
        self._handle: LocalQwenToolHandle | None = None
        self._lock = RLock()

    def start(self, *, warmup: bool = True) -> None:
        if self._started:
            return
        self.lab.start()
        self._started = True
        try:
            if warmup:
                self.lab.warmup(max_tokens=8)
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        with self._lock:
            handle = self._handle
        if handle is not None and handle.running:
            self.cancel(handle.task, "local runner shutdown")
            self.approval_gateway.close()
            handle.wait()
        else:
            self.approval_gateway.close()
        if self._started:
            self.lab.shutdown()
            self._started = False

    def create_task(self, instruction: str) -> Task:
        title = next((line.strip() for line in instruction.splitlines() if line.strip()), "Qwen tool task")
        return self.agent.create_task(
            title=title[:120],
            description=instruction,
            metadata={"topology": "local", "agent": "qwen-tools"},
        )

    def run_async(self, task: Task, instruction: str) -> LocalQwenToolHandle:
        with self._lock:
            if self._handle is not None and self._handle.running:
                raise RuntimeError("a Qwen tool-agent task is already running")
            done = Event()
            handle: LocalQwenToolHandle

            def run() -> None:
                try:
                    handle.result = self.tool_agent.run(task, instruction)
                except BaseException as exc:
                    handle.error = exc
                finally:
                    done.set()

            thread = Thread(target=run, name=f"agentcore-qwen-tools-{task.id[:8]}")
            handle = LocalQwenToolHandle(task=task, thread=thread, done=done)
            self._handle = handle
            thread.start()
            return handle

    def approve_pending(self) -> None:
        self.approval_gateway.decide(True)

    def reject_pending(self) -> None:
        self.approval_gateway.decide(False)

    def queue_steering(self, message: str) -> bool:
        return self.steering.queue(message)

    def cancel(self, task: Task, reason: str = "user abort") -> Task:
        result = self.agent.cancel_task(task, reason, executing=True)
        pending = self.approval_gateway.pending
        if pending is not None:
            try:
                self.approval_gateway.decide(False, tool_call_id=pending.call.id)
            except ValueError:
                pass
        return result

    def status(self, task: Task) -> dict[str, Any]:
        pending = self.approval_gateway.pending
        with self._lock:
            running = self._handle is not None and self._handle.running
        return {
            "runtime_ready": self.lab.ready(),
            "task": task.as_dict(),
            "agent": "qwen-tools",
            "running": running,
            "pending_approval": None if pending is None else pending.as_dict(),
            "workspace": self.agent.workspace.as_dict(),
        }

    def report(self, task: Task) -> TaskReport:
        return task.report()

    def diff(self) -> str:
        if not self.agent.git.is_repo():
            return ""
        return self.agent.git.diff().stdout
