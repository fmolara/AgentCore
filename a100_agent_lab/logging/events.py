from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from a100_agent_lab.generation.result import GenerationResult
from a100_agent_lab.sessions.session import Session


def generation_event(
    runtime: str,
    session: Session,
    result: GenerationResult,
    health: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "generation",
        "runtime": runtime,
        "session_id": session.id,
        "turn": sum(1 for message in session.messages if message.role == "assistant"),
        "metrics": result.metrics.as_dict(),
        "health": health,
        "status": "ok",
    }

