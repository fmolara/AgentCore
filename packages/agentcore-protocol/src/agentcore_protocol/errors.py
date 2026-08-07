from __future__ import annotations

from typing import Any


class AgentCoreError(Exception):
    """Base protocol/client error."""


class AgentCoreConnectionError(AgentCoreError):
    """Raised when the server cannot be reached."""


class AgentCoreProtocolError(AgentCoreError):
    """Raised when a response cannot be parsed as AgentCore protocol data."""


class AgentCoreCompatibilityError(AgentCoreError):
    """Raised when client and server protocol versions are incompatible."""


class AgentCoreHTTPError(AgentCoreError):
    def __init__(self, status_code: int, message: str, *, response: dict[str, Any] | None = None):
        super().__init__(f"AgentCore HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.response = response or {}
