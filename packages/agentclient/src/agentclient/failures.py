from __future__ import annotations

import errno
import re
import socket
import ssl
from dataclasses import dataclass
from enum import Enum
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx

from agentcore_protocol import (
    AgentCoreCompatibilityError,
    AgentCoreConnectionError,
    AgentCoreHTTPError,
    AgentCoreProtocolError,
)

from agentclient.exit_codes import ExitCode


class OperationContext(str, Enum):
    CONNECT = "connect"
    REQUEST = "request"
    STREAM = "stream"


@dataclass(frozen=True)
class ClientFailure:
    exit_code: ExitCode
    cause: str
    url: str
    host: str
    port: int | None
    exception: BaseException | None = None

    @property
    def category(self) -> str:
        return self.exit_code.name

    def as_text(self) -> str:
        port = "" if self.port is None else f":{self.port}"
        return "\n".join(
            (
                "AgentClient operation failed",
                f"URL: {self.url}",
                f"Host: {self.host}{port}",
                f"Category: {self.category}",
                f"Exit code: {int(self.exit_code)}",
                f"Cause: {self.cause}",
            )
        )


class ClientFailureError(Exception):
    def __init__(self, failure: ClientFailure) -> None:
        super().__init__(failure.cause)
        self.failure = failure


def classify_exception(
    exc: BaseException,
    url: str,
    *,
    operation: OperationContext = OperationContext.REQUEST,
    stream_started: bool = False,
) -> ClientFailure:
    safe_url, host, port = describe_url(url)

    if isinstance(exc, ClientFailureError):
        return exc.failure
    if isinstance(exc, AgentCoreCompatibilityError):
        return _failure(ExitCode.PROTOCOL_INCOMPATIBLE, exc, safe_url, host, port)
    if isinstance(exc, AgentCoreProtocolError):
        code = ExitCode.STREAM_ERROR if operation is OperationContext.STREAM else ExitCode.PROTOCOL_INCOMPATIBLE
        return _failure(code, exc, safe_url, host, port)
    if isinstance(exc, AgentCoreHTTPError):
        return _failure(ExitCode.HTTP_ERROR, exc, safe_url, host, port)

    chain = tuple(exception_chain(exc))
    if _has_tls_failure(chain):
        return _failure(ExitCode.TLS_ERROR, _root_cause(chain), safe_url, host, port)
    if _has_errno(chain, errno.ECONNREFUSED):
        cause = f"connection refused by {_host_port(host, port)}"
        return ClientFailure(ExitCode.CONNECTION_REFUSED, cause, safe_url, host, port, exc)

    if operation is OperationContext.STREAM and stream_started:
        return _failure(ExitCode.STREAM_ERROR, _root_cause(chain), safe_url, host, port)
    if any(isinstance(item, httpx.TimeoutException) for item in chain):
        return _failure(ExitCode.NETWORK_UNREACHABLE, _root_cause(chain), safe_url, host, port)
    if _has_network_failure(chain):
        return _failure(ExitCode.NETWORK_UNREACHABLE, _root_cause(chain), safe_url, host, port)
    if isinstance(exc, AgentCoreConnectionError) or any(isinstance(item, httpx.RequestError) for item in chain):
        code = (
            ExitCode.STREAM_ERROR
            if operation is OperationContext.STREAM and stream_started
            else ExitCode.NETWORK_UNREACHABLE
        )
        return _failure(code, _root_cause(chain), safe_url, host, port)

    return _failure(ExitCode.INTERNAL_CLIENT_ERROR, exc, safe_url, host, port)


def describe_url(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    host = parsed.hostname or "(unknown)"
    try:
        explicit_port = parsed.port
    except ValueError:
        explicit_port = None
    port = explicit_port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else None
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if explicit_port is not None:
        netloc = f"{netloc}:{explicit_port}"
    safe_url = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return safe_url, host, port


def exception_chain(exc: BaseException) -> Iterable[BaseException]:
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in _nested_exceptions(current):
            if id(nested) not in seen:
                pending.append(nested)


def _nested_exceptions(exc: BaseException) -> Iterable[BaseException]:
    for nested in (exc.__cause__, exc.__context__):
        if isinstance(nested, BaseException):
            yield nested
    group = getattr(exc, "exceptions", ())
    if isinstance(group, tuple | list):
        for nested in group:
            if isinstance(nested, BaseException):
                yield nested
    for name in ("error", "exc", "exception"):
        nested = getattr(exc, name, None)
        if isinstance(nested, BaseException):
            yield nested
    for arg in exc.args:
        if isinstance(arg, BaseException):
            yield arg


def _has_tls_failure(chain: tuple[BaseException, ...]) -> bool:
    return any(
        isinstance(item, ssl.SSLError)
        or item.__class__.__module__.startswith("ssl")
        or "tls" in item.__class__.__name__.lower()
        for item in chain
    )


def _has_errno(chain: tuple[BaseException, ...], target: int) -> bool:
    return any(isinstance(item, OSError) and item.errno == target for item in chain)


def _has_network_failure(chain: tuple[BaseException, ...]) -> bool:
    network_errnos = {
        errno.EADDRNOTAVAIL,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    }
    return any(
        isinstance(item, socket.gaierror)
        or isinstance(item, OSError) and item.errno in network_errnos
        or isinstance(item, (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError))
        for item in chain
    )


def _root_cause(chain: tuple[BaseException, ...]) -> BaseException:
    return chain[-1] if chain else RuntimeError("unknown client failure")


def _failure(
    code: ExitCode,
    exc: BaseException,
    safe_url: str,
    host: str,
    port: int | None,
) -> ClientFailure:
    cause = _concise_cause(exc)
    return ClientFailure(code, cause, safe_url, host, port, exc)


def _concise_cause(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = exc.__class__.__name__
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1", text)
    return text[:500]


def _host_port(host: str, port: int | None) -> str:
    return host if port is None else f"{host}:{port}"
