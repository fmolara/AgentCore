from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

try:
    from agentcore_protocol import AgentEvent
except ModuleNotFoundError:  # pragma: no cover - source-tree bootstrap for the transitional monorepo split.
    protocol_src = Path(__file__).resolve().parents[1] / "packages" / "agentcore-protocol" / "src"
    if protocol_src.exists():
        sys.path.insert(0, str(protocol_src))
    from agentcore_protocol import AgentEvent


class EventSink(Protocol):
    def emit(self, event: AgentEvent) -> None:
        ...


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]
