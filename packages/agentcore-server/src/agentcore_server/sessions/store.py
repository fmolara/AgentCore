from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from agentcore_server.sessions.session import Session


class SessionStore:
    def __init__(self, factory: Callable[..., Session] = Session):
        self._factory = factory
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def create(self, *, system_prompt: str | None = None) -> Session:
        session = self._factory(system_prompt=system_prompt)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def add(self, session: Session) -> Session:
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            return self._sessions[session_id]

    def list(self) -> list[Session]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda session: session.created_at)

    def reset(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions[session_id]
            session.reset()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
