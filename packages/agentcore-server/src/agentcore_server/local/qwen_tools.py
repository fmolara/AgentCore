from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, Callable

from agentcore_server.agents import Agent
from agentcore_server.api.client import AgentLab
from agentcore_server.events import EventSink
from agentcore_server.tasks import Task, TaskReport
from agentcore_server.tool_agent import (
    TOOL_AGENT_SYSTEM_PROMPT,
    ToolAgentLimits,
    ToolLoopAgent,
    ToolRunResult,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolSteeringInbox,
)


class InteractiveToolApprovalGateway:
    def __init__(
        self,
        *,
        presenter: Callable[[ToolApprovalRequest], None] | None = None,
        preview_directory: str | Path | None = None,
    ) -> None:
        self._condition = Condition()
        self._pending: ToolApprovalRequest | None = None
        self._decision: ToolApprovalDecision | None = None
        self._closed = False
        self._presenter = presenter
        self._temporary_directory: TemporaryDirectory[str] | None = None
        if preview_directory is None:
            self._temporary_directory = TemporaryDirectory(
                prefix="agentcore-tool-previews-"
            )
            self.preview_directory = Path(self._temporary_directory.name)
        else:
            self.preview_directory = Path(preview_directory).expanduser().resolve()
            self.preview_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.preview_directory.chmod(0o700)

    @property
    def pending(self) -> ToolApprovalRequest | None:
        with self._condition:
            return self._pending

    def request(self, request: ToolApprovalRequest) -> ToolApprovalDecision:
        artifact = self._materialize(request)
        prepared = replace(request, preview_artifact=str(artifact))
        with self._condition:
            if self._closed:
                return ToolApprovalDecision(
                    approved=False,
                    tool_call_id=request.call.id,
                    preview_digest=request.preview.digest,
                    reason="approval gateway closed",
                )
            if self._pending is not None:
                raise RuntimeError("another tool approval is already pending")
            self._pending = prepared
            self._decision = None
            self._condition.notify_all()
        if self._presenter is not None:
            self._presenter(prepared)
        with self._condition:
            while self._decision is None and not self._closed:
                self._condition.wait()
            decision = self._decision or ToolApprovalDecision(
                approved=False,
                tool_call_id=prepared.call.id,
                preview_digest=prepared.preview.digest,
                reason="approval gateway closed",
            )
            self._pending = None
            self._decision = None
            self._condition.notify_all()
            return decision

    def decide(
        self,
        approved: bool,
        *,
        tool_call_id: str,
        preview_digest: str,
        reason: str | None = None,
    ) -> None:
        with self._condition:
            if self._pending is None:
                raise ValueError("no tool approval is pending")
            if self._pending.call.id != tool_call_id:
                raise ValueError("approval tool-call ID does not match the pending call")
            if self._pending.preview.digest != preview_digest:
                raise ValueError("approval preview digest does not match the pending preview")
            if self._decision is not None:
                raise ValueError("pending tool call already has a decision")
            self._decision = ToolApprovalDecision(
                approved=approved,
                tool_call_id=tool_call_id,
                preview_digest=preview_digest,
                reason=reason,
            )
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
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def _materialize(self, request: ToolApprovalRequest) -> Path:
        suffix = ".diff" if request.preview.metadata.get("preview_format") == "unified_diff" else ".txt"
        path = self.preview_directory / f"{request.preview.preview_id}{suffix}"
        path.write_text(request.preview.content, encoding="utf-8")
        path.chmod(0o600)
        return path


@dataclass
class LocalToolLoopHandle:
    task: Task
    thread: Thread
    done: Event
    result: ToolRunResult | None = None
    error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def wait(self, timeout: float | None = None) -> ToolRunResult | None:
        self.thread.join(timeout)
        return self.result


class LocalToolLoopApp:
    """Local composition for the topology-neutral native tool loop."""

    def __init__(
        self,
        lab: AgentLab,
        *,
        workspace: str,
        system_prompt: str | None = None,
        event_sink: EventSink | None = None,
        approval_gateway: InteractiveToolApprovalGateway | None = None,
        approval_presenter: Callable[[ToolApprovalRequest], None] | None = None,
        preview_directory: str | Path | None = None,
    ) -> None:
        self.lab = lab
        self.event_sink = event_sink
        self.approval_gateway = approval_gateway or InteractiveToolApprovalGateway(
            presenter=approval_presenter,
            preview_directory=preview_directory,
        )
        self.steering = ToolSteeringInbox()
        self.agent: Agent = lab.create_agent(
            system_prompt=system_prompt or TOOL_AGENT_SYSTEM_PROMPT,
            workspace_root=workspace,
            event_sink=event_sink,
        )
        self.tool_agent = ToolLoopAgent(
            self.agent,
            approval_gateway=self.approval_gateway,
            limits=ToolAgentLimits.from_config(lab.config.get("tool_agent")),
            steering=self.steering,
        )
        self._started = False
        self._handle: LocalToolLoopHandle | None = None
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
        title = next((line.strip() for line in instruction.splitlines() if line.strip()), "Native tool task")
        protocol = getattr(self.lab.runtime, "tool_protocol", None)
        protocol_name = getattr(protocol, "name", "qwen")
        return self.agent.create_task(
            title=title[:120],
            description=instruction,
            metadata={"topology": "local", "agent": "tool-loop", "protocol": protocol_name},
        )

    def run_async(self, task: Task, instruction: str) -> LocalToolLoopHandle:
        with self._lock:
            if self._handle is not None and self._handle.running:
                raise RuntimeError("a native tool-agent task is already running")
            done = Event()
            handle: LocalToolLoopHandle

            def run() -> None:
                try:
                    handle.result = self.tool_agent.run(task, instruction)
                except BaseException as exc:
                    handle.error = exc
                finally:
                    done.set()

            thread = Thread(target=run, name=f"agentcore-tool-loop-{task.id[:8]}")
            handle = LocalToolLoopHandle(task=task, thread=thread, done=done)
            self._handle = handle
            thread.start()
            return handle

    def approve_pending(self) -> None:
        pending = self.approval_gateway.pending
        if pending is None:
            raise ValueError("no tool approval is pending")
        self.approval_gateway.decide(
            True,
            tool_call_id=pending.call.id,
            preview_digest=pending.preview.digest,
        )

    def reject_pending(self, reason: str | None = None) -> None:
        pending = self.approval_gateway.pending
        if pending is None:
            raise ValueError("no tool approval is pending")
        self.approval_gateway.decide(
            False,
            tool_call_id=pending.call.id,
            preview_digest=pending.preview.digest,
            reason=reason or "rejected by local operator",
        )

    def queue_steering(self, message: str) -> bool:
        return self.steering.queue(message)

    def cancel(self, task: Task, reason: str = "user abort") -> Task:
        result = self.agent.cancel_task(task, reason, executing=True)
        pending = self.approval_gateway.pending
        if pending is not None:
            try:
                self.approval_gateway.decide(
                    False,
                    tool_call_id=pending.call.id,
                    preview_digest=pending.preview.digest,
                    reason=reason,
                )
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
            "agent": "tool-loop",
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


LocalQwenToolApp = LocalToolLoopApp
LocalQwenToolHandle = LocalToolLoopHandle
