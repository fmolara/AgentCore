from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.logging.events import task_event
from a100_agent_lab.sessions.session import Session
from a100_agent_lab.tasks import (
    Task,
    TaskCheckpoint,
    TaskCheckpointRestorePlan,
    TaskCheckpointRestoreResult,
    TaskReport,
)
from a100_agent_lab.workspace import Workspace

if TYPE_CHECKING:
    from a100_agent_lab.api.client import AgentLab


class Agent:
    def __init__(
        self,
        lab: AgentLab,
        *,
        system_prompt: str | None = None,
        session: Session | None = None,
        workspace: Workspace | None = None,
        workspace_root: str | Path | None = None,
        workspace_mode: str = Workspace.READ_WRITE,
        workspace_metadata: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.lab = lab
        self.session = session or lab.create_session(system_prompt=system_prompt)
        self.workspace = workspace or Workspace(
            workspace_root or lab.default_workspace_root(),
            mode=workspace_mode,
            metadata=workspace_metadata,
        )
        self.generation_options = dict(generation_options or {})
        self.generation_options.update({key: value for key, value in kwargs.items() if value is not None})
        self._last_result: GenerationResult | None = None
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0
        self._tasks: list[Task] = []
        self._current_task: Task | None = None

    @property
    def runtime(self):
        return self.lab.runtime

    @property
    def git(self):
        return self.workspace.git

    @property
    def files(self):
        return self.workspace.files

    @property
    def last_metrics(self) -> GenerationMetrics | None:
        return None if self._last_result is None else self._last_result.metrics

    def ask(self, prompt: str, **kwargs: Any) -> GenerationResult:
        options = self._merged_options(kwargs)
        result = self.lab.generate(self.session, prompt, **options)
        self._record_result(result)
        return result

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        options = self._merged_options(kwargs)
        yield from self.lab.stream(self.session, prompt, **options)

    def reset(self) -> None:
        self.lab.reset_session(self.session.id)
        self._last_result = None
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0

    def create_task(
        self,
        *,
        title: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        task_metadata = self._task_metadata(metadata)
        task = Task(
            title=title,
            description=description,
            metadata=task_metadata,
            _on_event=self._log_task_event,
            _reporter=self._build_task_report,
            _checkpoint_builder=self._build_task_checkpoint,
            _restore_plan_builder=self._build_task_restore_plan,
            _restore_executor=self._restore_task_checkpoint,
        )
        self._tasks.append(task)
        self._current_task = task
        self._log_task_event(task, "task_created")
        return task

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def current_task(self) -> Task | None:
        return self._current_task

    def current_task_report(self) -> TaskReport | None:
        task = self.current_task()
        return None if task is None else task.report()

    def propose_plan(
        self,
        task: Task,
        *,
        instruction: str,
        planner: Any | None = None,
        approval_policy: Any | None = None,
        **generation_options: Any,
    ):
        if task not in self._tasks:
            raise ValueError("task is not owned by this agent")
        if planner is None:
            from a100_agent_lab.planning import SimpleLLMPlanner

            planner = SimpleLLMPlanner()
        return planner.propose(
            self,
            task,
            instruction=instruction,
            approval_policy=approval_policy,
            **generation_options,
        )

    def statistics(self) -> dict[str, Any]:
        metrics = self.last_metrics
        return {
            "session_id": self.session.id,
            "conversation_turns": self.session.turn_count,
            "prompt_tokens": self._total_prompt_tokens,
            "generated_tokens": self._total_generated_tokens,
            "last_ttft_sec": None if metrics is None else metrics.ttft_sec,
            "last_tokens_per_sec": None if metrics is None else metrics.tokens_per_sec,
            "last_wall_sec": None if metrics is None else metrics.wall_sec,
            "generation_options": dict(self.generation_options),
            "workspace": self.workspace.as_dict(),
            "tasks": {
                "count": len(self._tasks),
                "current_task_id": None if self._current_task is None else self._current_task.id,
            },
        }

    def _merged_options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        options = dict(self.generation_options)
        options.update({key: value for key, value in overrides.items() if value is not None})
        return options

    def _record_result(self, result: GenerationResult) -> None:
        self._last_result = result
        self._total_prompt_tokens += result.metrics.prompt_tokens
        self._total_generated_tokens += result.metrics.generated_tokens

    def _task_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        task_metadata = dict(metadata or {})
        task_metadata.setdefault("workspace", self.workspace.as_dict())
        if self.git.is_repo():
            task_metadata.setdefault("git_commit_before", self.git.current_commit())
        else:
            task_metadata.setdefault("git_commit_before", None)
        task_metadata.setdefault("git_commit_after", None)
        return task_metadata

    def _log_task_event(self, task: Task, event_type: str) -> None:
        if event_type in {"task_completed", "task_failed", "task_cancelled"} and self.git.is_repo():
            task.metadata["git_commit_after"] = self.git.current_commit()
        if event_type in {"task_completed", "task_failed", "task_cancelled"}:
            self._current_task = None if self._current_task is task else self._current_task
        writer = getattr(self.lab.runtime, "log_writer", None)
        if writer is None:
            return
        runtime_name = self.lab.statistics().get("runtime_name", "unknown")
        writer.write(task_event(runtime_name, self.session, task, event_type=event_type))

    def _build_task_report(self, task: Task) -> TaskReport:
        if not self.git.is_repo():
            return TaskReport.from_task(task)
        status = self.git.status().stdout
        return TaskReport.from_task(
            task,
            git_branch=self.git.current_branch(),
            git_status=status,
            git_diff=self.git.diff().stdout,
            files_changed=_files_changed_from_status(status),
        )

    def _build_task_checkpoint(
        self,
        task: Task,
        label: str,
        description: str | None,
        metadata: dict[str, Any] | None,
    ) -> TaskCheckpoint:
        checkpoint_metadata = dict(metadata or {})
        if not self.git.is_repo():
            return TaskCheckpoint.from_task(
                task,
                label=label,
                description=description,
                metadata=checkpoint_metadata,
            )
        status = self.git.status().stdout
        diff = self.git.diff().stdout
        checkpoint_metadata.setdefault("_file_snapshots", self._file_snapshots(status, diff))
        return TaskCheckpoint.from_task(
            task,
            label=label,
            description=description,
            git_branch=self.git.current_branch(),
            git_status=status,
            git_diff=diff,
            metadata=checkpoint_metadata,
        )

    def _build_task_restore_plan(self, task: Task, checkpoint: TaskCheckpoint) -> TaskCheckpointRestorePlan:
        if not self.git.is_repo():
            return TaskCheckpointRestorePlan.from_checkpoint(checkpoint)
        return TaskCheckpointRestorePlan.from_checkpoint(
            checkpoint,
            current_git_status=self.git.status().stdout,
            current_git_diff=self.git.diff().stdout,
        )

    def _restore_task_checkpoint(
        self,
        task: Task,
        checkpoint: TaskCheckpoint,
        plan: TaskCheckpointRestorePlan,
        force: bool,
    ) -> TaskCheckpointRestoreResult:
        snapshots = checkpoint.metadata.get("_file_snapshots")
        if not isinstance(snapshots, dict) or not snapshots:
            raise ValueError("target checkpoint has no restorable file snapshots")
        restored_files: list[str] = []
        for path, content in snapshots.items():
            if not isinstance(path, str) or not isinstance(content, str):
                raise ValueError("target checkpoint contains invalid file snapshot metadata")
            self.files.write_text(path, content)
            restored_files.append(path)
        return TaskCheckpointRestoreResult(
            target_checkpoint_id=checkpoint.id,
            target_checkpoint_label=checkpoint.label,
            restored_files=tuple(sorted(restored_files)),
            warnings=plan.warnings,
            forced=force,
            safe_plan=plan.safe_to_restore,
        )

    def _file_snapshots(self, status: str, diff: str) -> dict[str, str]:
        snapshots: dict[str, str] = {}
        files = set(_files_changed_from_status(status))
        files.update(_files_changed_from_diff(diff))
        for path in sorted(files):
            try:
                snapshots[path] = self.files.read_text(path)
            except FileNotFoundError:
                continue
        return snapshots


def _files_changed_from_status(status: str) -> tuple[str, ...]:
    files: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        path = line[3:] if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path.strip())
    return tuple(files)


def _files_changed_from_diff(diff: str) -> tuple[str, ...]:
    files: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        files.append(path)
    return tuple(files)
