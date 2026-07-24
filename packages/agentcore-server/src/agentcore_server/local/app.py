from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock, Thread
from typing import Any

from agentcore_server.agents import Agent
from agentcore_server.api.client import AgentLab
from agentcore_server.events import AgentEvent, EventSink
from agentcore_server.executor import ApprovalPolicy, PlanProposal, TaskExecutionResult
from agentcore_server.planning import PlannerResult, build_planner
from agentcore_server.tasks import Task, TaskReport


class InvalidProposalError(ValueError):
    def __init__(self, message: str, *, result: PlannerResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class LocalExecutionHandle:
    task: Task
    proposal: PlanProposal
    thread: Thread
    done: Event
    result: TaskExecutionResult | None = None
    error: BaseException | None = None

    def wait(self, timeout: float | None = None) -> TaskExecutionResult | None:
        self.thread.join(timeout)
        return self.result

    @property
    def running(self) -> bool:
        return self.thread.is_alive()


class LocalAgentCoreApp:
    """Topology-neutral AgentCore domain objects composed in one process."""

    def __init__(
        self,
        lab: AgentLab,
        *,
        workspace: str,
        system_prompt: str | None = None,
        planner: Any | None = None,
        planner_mode: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.lab = lab
        self.event_sink = event_sink
        self.planner = planner or build_planner(
            lab.config,
            mode_override=planner_mode,
        )
        self.approval_policy = approval_policy or ApprovalPolicy.default()
        self.agent: Agent = lab.create_agent(
            system_prompt=system_prompt,
            workspace_root=workspace,
            event_sink=event_sink,
        )
        self._started = False
        self._execution: LocalExecutionHandle | None = None
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
            handle = self._execution
        if handle is not None and handle.running:
            try:
                self.cancel(handle.task, "local runner shutdown")
            except ValueError:
                pass
            handle.wait()
        if self._started:
            self.lab.shutdown()
            self._started = False

    def create_task(self, instruction: str) -> Task:
        title = next((line.strip() for line in instruction.splitlines() if line.strip()), "Local task")
        return self.agent.create_task(
            title=title[:120],
            description=instruction,
            metadata={"topology": "local"},
        )

    def propose(
        self,
        task: Task,
        instruction: str,
        *,
        stream: bool = True,
        **generation_options: Any,
    ) -> PlannerResult:
        managed_diagnostics = bool(
            getattr(self.planner, "diagnostics_managed", False)
        )
        if not managed_diagnostics:
            prompt = self.planner.build_prompt(self.agent, task, instruction)
            self._emit(
                "planner.prompt",
                "Effective planner prompt prepared",
                task,
                {"prompt": prompt, "sanitized": True},
            )
        if stream:
            iterator = self.agent.propose_plan_stream(
                task,
                instruction=instruction,
                planner=self.planner,
                approval_policy=self.approval_policy,
                **generation_options,
            )
            result = self._consume_stream(iterator)
        else:
            result = self.agent.propose_plan(
                task,
                instruction=instruction,
                planner=self.planner,
                approval_policy=self.approval_policy,
                **generation_options,
            )
        if not managed_diagnostics:
            self._emit(
                "planner.raw_output",
                "Visible final model output captured",
                task,
                {"text": result.raw_text, "content_kind": "visible_model_text"},
            )
            self._emit("planner.result", "Planner result parsed", task, result.as_dict())
        self._validate_result(result, task)
        proposal = result.proposal
        assert proposal is not None
        if not managed_diagnostics:
            self._emit(
                "planner.validation",
                "ActionPlan structural validation passed",
                task,
                {"warnings": [], "action_count": len(proposal.action_plan.actions)},
            )
            self._emit(
                "approval.policy",
                "Approval policy evaluated",
                task,
                {
                    "requirements": [
                        requirement.as_dict() for requirement in proposal.approval_requirements
                    ],
                    "explicit_local_approval_required": True,
                },
            )
        return result

    def approve(self, task: Task, proposal: PlanProposal) -> None:
        self.agent.approve_proposal(task, proposal)
        task.metadata["local_approval"] = {
            "approved": True,
            "proposal_id": proposal.id,
            "topology": "local",
        }

    def reject(self, task: Task, proposal: PlanProposal, reason: str) -> None:
        self.agent.reject_proposal(task, proposal, reason)

    def execute(self, task: Task, proposal: PlanProposal) -> TaskExecutionResult:
        result = self.agent.execute_proposal(task, proposal)
        self._emit_final_state(task)
        return result

    def execute_async(self, task: Task, proposal: PlanProposal) -> LocalExecutionHandle:
        with self._lock:
            if self._execution is not None and self._execution.running:
                raise RuntimeError("local task execution is already running")
            done = Event()
            handle: LocalExecutionHandle

            def run() -> None:
                try:
                    handle.result = self.execute(task, proposal)
                except BaseException as exc:
                    handle.error = exc
                finally:
                    done.set()

            thread = Thread(target=run, name=f"agentcore-local-{task.id[:8]}")
            handle = LocalExecutionHandle(task=task, proposal=proposal, thread=thread, done=done)
            self._execution = handle
            handle.thread.start()
            return handle

    def cancel(self, task: Task, reason: str = "user abort") -> Task:
        with self._lock:
            executing = self._execution is not None and self._execution.running
        return self.agent.cancel_task(task, reason, executing=executing)

    def status(self, task: Task, proposal: PlanProposal | None = None) -> dict[str, Any]:
        with self._lock:
            running = self._execution is not None and self._execution.running
        return {
            "runtime_ready": self.lab.ready(),
            "task": task.as_dict(),
            "proposal": None if proposal is None else proposal.as_dict(),
            "execution_running": running,
            "workspace": self.agent.workspace.as_dict(),
        }

    def report(self, task: Task) -> TaskReport:
        return task.report()

    def diff(self) -> str:
        if not self.agent.git.is_repo():
            return ""
        return self.agent.git.diff().stdout

    def _consume_stream(self, iterator) -> PlannerResult:
        while True:
            try:
                next(iterator)
            except StopIteration as stop:
                result = stop.value
                if not isinstance(result, PlannerResult):
                    raise InvalidProposalError("planner stream ended without a PlannerResult")
                return result

    def _validate_result(self, result: PlannerResult, task: Task) -> None:
        if not result.ok or result.proposal is None:
            raise InvalidProposalError(result.error or "planner did not produce a proposal", result=result)
        if not result.proposal.action_plan.actions:
            self._emit(
                "planner.validation",
                "ActionPlan structural validation failed",
                task,
                {"warnings": [], "error": "action plan must contain at least one action"},
            )
            raise InvalidProposalError("action plan must contain at least one action", result=result)

    def _emit_final_state(self, task: Task) -> None:
        report = task.report()
        self._emit("task.report", "Final task report captured", task, {"report": report.as_dict()})
        self._emit("git.diff", "Final Git diff captured", task, {"diff": self.diff()})

    def _emit(
        self,
        event_type: str,
        summary: str,
        task: Task,
        payload: dict[str, Any],
    ) -> None:
        if self.event_sink is None:
            return
        event = AgentEvent(
            event_type=event_type,
            summary=summary,
            task_id=task.id,
            session_id=self.agent.session.id,
            payload=payload,
        )
        self.event_sink.emit(event)
