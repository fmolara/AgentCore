from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentcore_server.generation.result import GenerationResult
from agentcore_server.sessions.session import Session
from agentcore_server.tasks import Task


def generation_event(
    runtime: str,
    session: Session,
    result: GenerationResult,
    health: dict[str, Any],
    *,
    event_type: str = "generation",
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "runtime": runtime,
        "session_id": session.id,
        "turn": sum(1 for message in session.messages if message.role == "assistant"),
        "metrics": result.metrics.as_dict(),
        "health": health,
        "status": "ok",
    }


def task_event(
    runtime: str,
    session: Session,
    task: Task,
    *,
    event_type: str,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "runtime": runtime,
        "session_id": session.id,
        "task": task.as_dict(),
        "status": "ok",
    }
