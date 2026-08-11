from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Callable

from agentcore_server.events import AgentEvent


class LocalEventSink:
    """Fan out events to the terminal, an ordered trace, and passive run metrics."""

    def __init__(
        self,
        *,
        renderer: Callable[[AgentEvent], None] | None = None,
        trace_file: str | Path | None = None,
        metrics_file: str | Path | None = None,
        metrics_context: dict | None = None,
    ) -> None:
        self.renderer = renderer
        self.trace_file = None if trace_file is None else Path(trace_file).expanduser().resolve()
        self.metrics_file = (
            None if metrics_file is None else Path(metrics_file).expanduser().resolve()
        )
        self.metrics_context = dict(metrics_context or {})
        self.events: list[AgentEvent] = []
        self.metrics_errors: list[str] = []
        self._sequence = 0
        self._lock = RLock()
        self._runs: dict[str, dict] = {}
        if self.trace_file is not None:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            self.trace_file.write_text("", encoding="utf-8")
        if self.metrics_file is not None:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: AgentEvent) -> None:
        with self._lock:
            self._sequence += 1
            self.events.append(event)
            if self.trace_file is not None:
                record = {"sequence": self._sequence, **event.as_dict()}
                with self.trace_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            try:
                self._record_metrics(event)
            except (OSError, TypeError, ValueError) as exc:
                # Passive telemetry must never change task execution semantics.
                self.metrics_errors.append(str(exc) or exc.__class__.__name__)
            renderer = self.renderer
        if renderer is not None:
            renderer(event)

    def as_dicts(self) -> list[dict]:
        with self._lock:
            return [
                {"sequence": index, **event.as_dict()}
                for index, event in enumerate(self.events, start=1)
            ]

    def _record_metrics(self, event: AgentEvent) -> None:
        if self.metrics_file is None or event.task_id is None:
            return
        task_id = event.task_id
        if event.event_type == "agent.loop.started":
            limits = event.payload.get("limits") or {}
            normal_turn_limit = limits.get("max_model_turns")
            runway_turns = int(limits.get("completion_runway_turns") or 0)
            self._runs[task_id] = {
                "started_monotonic": monotonic(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "session_id": event.session_id,
                "protocol": event.payload.get("protocol"),
                "model_turns": 0,
                "tool_calls": 0,
                "approvals": 0,
                "rejections": 0,
                "tools_completed": 0,
                "tools_failed": 0,
                "checks_completed": 0,
                "checks_failed": 0,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "peak_prompt_tokens": 0,
                "generation_wall_sec": 0.0,
                "first_ttft_sec": None,
                "final_response_present": False,
                "runway_granted": False,
                "runway_turns": runway_turns,
                "runway_turns_used": 0,
                "normal_turn_limit": normal_turn_limit,
                "absolute_turn_limit": (
                    None
                    if normal_turn_limit is None
                    else int(normal_turn_limit) + runway_turns
                ),
                "report": None,
            }
            return
        run = self._runs.get(task_id)
        if run is None:
            return
        if event.event_type == "agent.turn.completed":
            metrics = event.payload.get("metrics") or {}
            run["model_turns"] += 1
            prompt_tokens = int(metrics.get("prompt_tokens") or 0)
            run["prompt_tokens"] += prompt_tokens
            run["generated_tokens"] += int(metrics.get("generated_tokens") or 0)
            run["peak_prompt_tokens"] = max(run["peak_prompt_tokens"], prompt_tokens)
            run["generation_wall_sec"] += float(metrics.get("wall_sec") or 0.0)
            if run["first_ttft_sec"] is None and metrics.get("ttft_sec") is not None:
                run["first_ttft_sec"] = float(metrics["ttft_sec"])
            normal_limit = run.get("normal_turn_limit")
            if normal_limit is not None:
                run["runway_turns_used"] = max(0, run["model_turns"] - normal_limit)
        elif event.event_type == "agent.turn_runway.granted":
            run["runway_granted"] = True
            run["runway_turns"] = int(event.payload["runway_turns"])
            run["normal_turn_limit"] = int(event.payload["base_limit"])
            run["absolute_turn_limit"] = int(event.payload["absolute_limit"])
        elif event.event_type == "tool.call.received":
            run["tool_calls"] += 1
        elif event.event_type == "tool.approved":
            run["approvals"] += 1
        elif event.event_type == "tool.rejected":
            run["rejections"] += 1
        elif event.event_type in {"tool.completed", "tool.failed"}:
            succeeded = event.event_type == "tool.completed"
            run["tools_completed" if succeeded else "tools_failed"] += 1
            if event.payload.get("tool") == "run_check":
                run["checks_completed" if succeeded else "checks_failed"] += 1
        elif event.event_type == "agent.final":
            run["final_response_present"] = bool(str(event.payload.get("text") or "").strip())
        elif event.event_type == "task.report":
            run["report"] = event.payload.get("report")
        elif event.event_type == "git.diff":
            self._finish_metrics(task_id, run, event)

    def _finish_metrics(self, task_id: str, run: dict, event: AgentEvent) -> None:
        report = run.get("report") or {}
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "started_at": run["started_at"],
            "wall_sec": round(monotonic() - run["started_monotonic"], 6),
            "task_id": task_id,
            "session_id": run["session_id"],
            **self.metrics_context,
            "protocol": run["protocol"],
            "status": report.get("status"),
            "model_turns": run["model_turns"],
            "tool_calls": run["tool_calls"],
            "tools_completed": run["tools_completed"],
            "tools_failed": run["tools_failed"],
            "approvals": run["approvals"],
            "rejections": run["rejections"],
            "checks_completed": run["checks_completed"],
            "checks_failed": run["checks_failed"],
            "prompt_tokens": run["prompt_tokens"],
            "generated_tokens": run["generated_tokens"],
            "peak_prompt_tokens": run["peak_prompt_tokens"],
            "generation_wall_sec": round(run["generation_wall_sec"], 6),
            "first_ttft_sec": run["first_ttft_sec"],
            "final_response_present": run["final_response_present"],
            "runway_granted": run["runway_granted"],
            "runway_turns": run["runway_turns"],
            "runway_turns_used": run["runway_turns_used"],
            "normal_turn_limit": run["normal_turn_limit"],
            "absolute_turn_limit": run["absolute_turn_limit"],
            "files_changed": report.get("files_changed", []),
            "diff_bytes": len(str(event.payload.get("diff") or "").encode("utf-8")),
            "trace_file": None if self.trace_file is None else str(self.trace_file),
        }
        with self.metrics_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._runs.pop(task_id, None)
