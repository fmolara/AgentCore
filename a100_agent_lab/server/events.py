from __future__ import annotations

import json
import queue
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

from a100_agent_lab.events import AgentEvent


@dataclass(frozen=True)
class StoredEvent:
    id: int
    event: AgentEvent

    def as_sse(self) -> str:
        return format_sse(self.event, event_id=self.id)


def format_sse(event: AgentEvent, *, event_id: int | None = None) -> str:
    data = json.dumps(event.as_dict(), ensure_ascii=False, sort_keys=True)
    event_id_line = "" if event_id is None else f"id: {event_id}\n"
    return f"event: {event.event_type}\n{event_id_line}data: {data}\n\n"


class ServerEventSink:
    def __init__(self, bus: "TaskEventBus") -> None:
        self.bus = bus

    def emit(self, event: AgentEvent) -> None:
        self.bus.publish(event)


class TaskEventBus:
    def __init__(self) -> None:
        self._lock = RLock()
        self._next_id = 1
        self._history: dict[str, list[StoredEvent]] = {}
        self._subscribers: dict[str, list[queue.Queue[StoredEvent | None]]] = {}

    def publish(self, event: AgentEvent) -> StoredEvent | None:
        if event.task_id is None:
            return None
        with self._lock:
            stored = StoredEvent(id=self._next_id, event=event)
            self._next_id += 1
            self._history.setdefault(event.task_id, []).append(stored)
            subscribers = list(self._subscribers.get(event.task_id, []))
        for subscriber in subscribers:
            subscriber.put(stored)
        return stored

    def history(self, task_id: str) -> list[StoredEvent]:
        with self._lock:
            return list(self._history.get(task_id, []))

    def iter_sse(self, task_id: str) -> Iterator[str]:
        subscriber: queue.Queue[StoredEvent | None] = queue.Queue()
        with self._lock:
            history = list(self._history.get(task_id, []))
            self._subscribers.setdefault(task_id, []).append(subscriber)
        try:
            for stored in history:
                yield stored.as_sse()
                if _is_terminal(stored.event):
                    return
            while True:
                try:
                    stored = subscriber.get(timeout=10)
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    continue
                if stored is None:
                    return
                yield stored.as_sse()
                if _is_terminal(stored.event):
                    return
        finally:
            with self._lock:
                subscribers = self._subscribers.get(task_id, [])
                if subscriber in subscribers:
                    subscribers.remove(subscriber)


def _is_terminal(event: AgentEvent) -> bool:
    return event.event_type in {
        "execution.completed",
        "task.failed",
        "cancellation.completed",
    }
