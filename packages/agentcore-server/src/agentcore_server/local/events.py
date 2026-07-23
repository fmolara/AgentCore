from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Callable

from agentcore_server.events import AgentEvent


class LocalEventSink:
    """Fan out the existing AgentEvent stream to terminal and ordered JSONL."""

    def __init__(
        self,
        *,
        renderer: Callable[[AgentEvent], None] | None = None,
        trace_file: str | Path | None = None,
    ) -> None:
        self.renderer = renderer
        self.trace_file = None if trace_file is None else Path(trace_file).expanduser().resolve()
        self.events: list[AgentEvent] = []
        self._sequence = 0
        self._lock = RLock()
        if self.trace_file is not None:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            self.trace_file.write_text("", encoding="utf-8")

    def emit(self, event: AgentEvent) -> None:
        with self._lock:
            self._sequence += 1
            self.events.append(event)
            if self.trace_file is not None:
                record = {"sequence": self._sequence, **event.as_dict()}
                with self.trace_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            renderer = self.renderer
        if renderer is not None:
            renderer(event)

    def as_dicts(self) -> list[dict]:
        with self._lock:
            return [
                {"sequence": index, **event.as_dict()}
                for index, event in enumerate(self.events, start=1)
            ]
