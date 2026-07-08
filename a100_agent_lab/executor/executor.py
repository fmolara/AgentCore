from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from a100_agent_lab.executor.actions import Action, ActionResult
from a100_agent_lab.tasks import Task, TaskReport, TaskStatus

if TYPE_CHECKING:
    from a100_agent_lab.agents import Agent


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
        if task.status == TaskStatus.CREATED:
            task.start()
        elif task.status != TaskStatus.RUNNING:
            raise ValueError(f"cannot execute task with status: {task.status.value}")

        context = TaskExecutionContext(agent=self.agent, task=task)
        results: list[ActionResult] = []

        for action in actions:
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
            results.append(result)

        task.complete()
        return TaskExecutionResult(
            task_id=task.id,
            status="completed",
            actions=tuple(results),
            report=task.report(),
        )

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
