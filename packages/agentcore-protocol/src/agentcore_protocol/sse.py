from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator

from agentcore_protocol.events import AgentEvent
from agentcore_protocol.errors import AgentCoreProtocolError


@dataclass(frozen=True)
class SSEMessage:
    event: str | None
    data: str
    event_id: str | None = None

    def agent_event(self) -> AgentEvent:
        try:
            payload = json.loads(self.data)
        except json.JSONDecodeError as exc:
            raise AgentCoreProtocolError("SSE data is not valid JSON") from exc
        try:
            return AgentEvent.from_dict(payload)
        except ValueError as exc:
            raise AgentCoreProtocolError(str(exc)) from exc


def format_sse(event: AgentEvent, *, event_id: int | None = None) -> str:
    data = json.dumps(event.as_dict(), ensure_ascii=False, sort_keys=True)
    event_id_line = "" if event_id is None else f"id: {event_id}\n"
    return f"event: {event.event_type}\n{event_id_line}data: {data}\n\n"


def parse_sse_lines(lines: Iterable[str]) -> Iterator[SSEMessage]:
    event: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield SSEMessage(event=event, event_id=event_id, data="\n".join(data_lines))
            event = None
            event_id = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield SSEMessage(event=event, event_id=event_id, data="\n".join(data_lines))


def parse_agent_events(lines: Iterable[str]) -> Iterator[AgentEvent]:
    for message in parse_sse_lines(lines):
        yield message.agent_event()
