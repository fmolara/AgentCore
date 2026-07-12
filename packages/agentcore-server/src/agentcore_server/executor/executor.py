from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from agentcore_server.executor.actions import Action, ActionResult
from agentcore_server.tasks import Task, TaskReport, TaskStatus

if TYPE_CHECKING:
    from agentcore_server.agents import Agent
    from agentcore_server.executor.plan import ActionPlan, ApprovalPolicy


@dataclass(frozen=True)
class TaskExecutionContext:
    agent: Agent
    task: Task


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    status: str
    actions: tuple[ActionResult, ...] = ()
    report: TaskReport | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "actions": [action.as_dict() for action in self.actions],
            "report": None if self.report is None else self.report.as_dict(),
            "error": self.error,
        }


class TaskExecutor:
    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, task: Task, actions: Iterable[Action]) -> TaskExecutionResult:
        if task.status == TaskStatus.CANCELLED:
            return TaskExecutionResult(
                task_id=task.id,
                status="cancelled",
                actions=(),
                report=task.report(),
                error="task is cancelled",
            )
        if task.status == TaskStatus.CREATED:
            task.start()
        elif task.status != TaskStatus.RUNNING:
            raise ValueError(f"cannot execute task with status: {task.status.value}")

        context = TaskExecutionContext(agent=self.agent, task=task)
        results: list[ActionResult] = []
        self._emit_structured_event(
            task,
            "execution.started",
            f"Execution started for task: {task.title}",
        )

        for action in actions:
            if task.cancellation_requested:
                return self._cancel_execution(task, results)
            self._emit_structured_event(
                task,
                "action.started",
                f"Starting action: {action.action_type}",
                {"action_id": action.id, "action_type": action.action_type},
            )
            try:
                result = action.execute(context)
            except Exception as exc:
                result = ActionResult.failed(
                    action_id=action.id,
                    action_type=action.action_type,
                    error=str(exc),
                )
                self._record_action(task, result)
                self._emit_action_event(task, result)
                self._emit_structured_action_result(task, result)
                task.fail(str(exc))
                results.append(result)
                return TaskExecutionResult(
                    task_id=task.id,
                    status="failed",
                    actions=tuple(results),
                    report=task.report(),
                    error=str(exc),
                )

            self._record_action(task, result)
            self._emit_action_event(task, result)
            self._emit_structured_action_result(task, result)
            results.append(result)
            if task.cancellation_requested:
                return self._cancel_execution(task, results)

        task.complete()
        self._emit_structured_event(
            task,
            "execution.completed",
            f"Execution completed for task: {task.title}",
            {"actions": len(results)},
        )
        return TaskExecutionResult(
            task_id=task.id,
            status="completed",
            actions=tuple(results),
            report=task.report(),
        )

    def _cancel_execution(self, task: Task, results: list[ActionResult]) -> TaskExecutionResult:
        if task.status != TaskStatus.CANCELLED:
            task.cancel(task.cancellation_reason or "cancel requested")
        self._emit_structured_event(
            task,
            "cancellation.completed",
            f"Cancellation completed for task: {task.title}",
            {"actions": len(results), "reason": task.cancellation_reason},
        )
        self._emit_structured_event(
            task,
            "execution.completed",
            f"Execution cancelled for task: {task.title}",
            {"actions": len(results), "status": "cancelled"},
        )
        return TaskExecutionResult(
            task_id=task.id,
            status="cancelled",
            actions=tuple(results),
            report=task.report(),
            error=task.cancellation_reason,
        )

    def execute_plan(
        self,
        task: Task,
        plan: ActionPlan,
        *,
        approval_policy: ApprovalPolicy | None = None,
        approved: bool = False,
    ) -> TaskExecutionResult:
        required = plan.required_approvals(approval_policy)
        if required and not approved:
            reasons = "; ".join(
                f"action {requirement.action_index} ({requirement.action_type}): {requirement.reason}"
                for requirement in required
            )
            return TaskExecutionResult(
                task_id=task.id,
                status="approval_required",
                actions=(),
                report=task.report(),
                error="action plan requires approval: " + reasons,
            )
        return self.execute(task, plan.actions)

    def _record_action(self, task: Task, result: ActionResult) -> None:
        actions = task.metadata.setdefault("actions", [])
        if not isinstance(actions, list):
            raise ValueError("task metadata field 'actions' must be a list")
        actions.append(result.as_dict())

    def _emit_action_event(self, task: Task, result: ActionResult) -> None:
        writer = getattr(self.agent.lab.runtime, "log_writer", None)
        if writer is None:
            return
        runtime_name = self.agent.lab.statistics().get("runtime_name", "unknown")
        writer.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": "task_action",
                "runtime": runtime_name,
                "session_id": self.agent.session.id,
                "task_id": task.id,
                "action": result.as_dict(),
                "status": result.status,
            }
        )

    def _emit_structured_action_result(self, task: Task, result: ActionResult) -> None:
        if result.status == "ok":
            self._emit_structured_event(
                task,
                "action.completed",
                f"Completed action: {result.action_type}",
                {"action": result.as_dict()},
            )
            self._emit_derived_events(task, result)
            return
        self._emit_structured_event(
            task,
            "action.failed",
            f"Failed action: {result.action_type}",
            {"action": result.as_dict()},
        )

    def _emit_derived_events(self, task: Task, result: ActionResult) -> None:
        if result.action_type in {"write_file", "replace_text"}:
            files = result.data.get("files_changed", [])
            self._emit_structured_event(
                task,
                "workspace.modified",
                "Workspace modified",
                {"files_changed": files, "action": result.as_dict()},
            )
        elif result.action_type == "create_checkpoint":
            checkpoint = result.data.get("checkpoint", {})
            label = checkpoint.get("label") if isinstance(checkpoint, dict) else None
            self._emit_structured_event(
                task,
                "checkpoint.created",
                f"Checkpoint created: {label or 'checkpoint'}",
                {"checkpoint": checkpoint},
            )
        elif result.action_type == "git_diff":
            self._emit_structured_event(
                task,
                "git.diff",
                "Git diff captured",
                {"diff": result.data.get("stdout", "")},
            )

    def _emit_structured_event(
        self,
        task: Task,
        event_type: str,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        emit = getattr(self.agent, "emit_event", None)
        if emit is None:
            return
        emit(event_type, summary, task=task, payload=payload or {})
