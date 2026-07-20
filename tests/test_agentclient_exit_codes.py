from __future__ import annotations

import errno
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
from typing import Any

import httpx

from agentclient.cli import run_cli
from agentclient.exit_codes import ExitCode
from agentclient.failures import OperationContext, classify_exception
from agentcore_protocol import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    AgentCoreClient,
    AgentCoreConnectionError,
    AgentEvent,
    format_sse,
)


ROOT = Path(__file__).resolve().parents[1]


class ExitCodeServer:
    def __init__(
        self,
        *,
        health_ready: bool = True,
        runtime_ready: bool | None = True,
        protocol_version: str = PROTOCOL_VERSION,
        health_status_code: int = 200,
        execution_status: str = "completed",
        terminal_event: str = "task.completed",
        malformed_proposal_stream: bool = False,
        malformed_after_terminal: bool = False,
    ) -> None:
        self.health_ready = health_ready
        self.runtime_ready = runtime_ready
        self.protocol_version = protocol_version
        self.health_status_code = health_status_code
        self.execution_status = execution_status
        self.terminal_event = terminal_event
        self.malformed_proposal_stream = malformed_proposal_stream
        self.malformed_after_terminal = malformed_after_terminal
        self.executed = False
        self.cancelled = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if self.health_status_code != 200:
                return httpx.Response(self.health_status_code, json={"detail": "health unavailable"})
            runtime: dict[str, Any] = {}
            if self.runtime_ready is not None:
                runtime["ready"] = self.runtime_ready
            if self.runtime_ready is False:
                runtime["last_error"] = "model unavailable"
            return httpx.Response(
                200,
                json={
                    "status": "ok" if self.health_ready else "starting",
                    "ready": self.health_ready,
                    "protocol_version": self.protocol_version,
                    "schema_version": SCHEMA_VERSION,
                    "runtime": runtime,
                },
            )
        if request.method == "POST" and path == "/v1/agents":
            payload = _json_payload(request)
            return httpx.Response(
                200,
                json={
                    "id": "agent-1",
                    "session_id": "session-1",
                    "workspace": {"root": payload.get("workspace_root")},
                },
            )
        if request.method == "POST" and path == "/v1/agents/agent-1/tasks":
            return httpx.Response(200, json={"agent_id": "agent-1", "task": {"id": "task-1"}})
        if request.method == "POST" and path == "/v1/tasks/task-1/proposals/stream":
            if self.malformed_proposal_stream:
                return httpx.Response(200, text="event: assistant.delta\ndata: not-json\n\n")
            return httpx.Response(200, text=_proposal_stream())
        if request.method == "POST" and path == "/v1/proposals/proposal-1/approve":
            return httpx.Response(200, json={"proposal": _proposal("approved")})
        if request.method == "POST" and path == "/v1/proposals/proposal-1/reject":
            return httpx.Response(200, json={"proposal": _proposal("rejected")})
        if request.method == "POST" and path == "/v1/proposals/proposal-1/execute":
            self.executed = True
            return httpx.Response(
                200,
                json={"task_id": "task-1", "status": self.execution_status, "actions": []},
            )
        if request.method == "GET" and path == "/v1/tasks/task-1/events":
            stream = format_sse(
                AgentEvent(event_type=self.terminal_event, task_id="task-1", summary=self.terminal_event),
                event_id=1,
            )
            if self.malformed_after_terminal:
                stream += "event: invalid\ndata: not-json\n\n"
            return httpx.Response(200, text=stream)
        if request.method == "POST" and path == "/v1/tasks/task-1/cancel":
            self.cancelled = True
            return httpx.Response(200, json={"task": {"id": "task-1", "status": "cancelled"}})
        return httpx.Response(404, json={"detail": f"unexpected endpoint {request.method} {path}"})


def test_successful_execution_returns_success() -> None:
    code = _run(ExitCodeServer(), ["Patch parser.c", "/approve", "/quit"])
    assert code is ExitCode.SUCCESS


def test_invalid_cli_options_return_usage_error() -> None:
    assert run_cli(["--not-an-option"]) is ExitCode.CLI_USAGE_ERROR


def test_malformed_server_url_returns_usage_error() -> None:
    assert run_cli(["--server", "not-a-url"], inputs=[]) is ExitCode.CLI_USAGE_ERROR


def test_network_unreachable_and_initial_timeout_classification() -> None:
    network = _connection_error(OSError(errno.ENETUNREACH, "network unreachable"))
    timeout = _connection_error(httpx.ReadTimeout("initial response timed out", request=_request()))
    direct_failure = classify_exception(network, "http://10.121.0.10:8080")
    assert direct_failure.exit_code is ExitCode.NETWORK_UNREACHABLE
    assert direct_failure.url == "http://10.121.0.10:8080"
    assert classify_exception(timeout, "http://10.121.0.10:8080").exit_code is ExitCode.NETWORK_UNREACHABLE


def test_nested_connection_refused_is_classified_by_errno() -> None:
    nested = _connection_error(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
    failure = classify_exception(nested, "http://127.0.0.1:8080")
    assert failure.exit_code is ExitCode.CONNECTION_REFUSED
    assert failure.url == "http://127.0.0.1:8080"


def test_dns_failure_is_network_unreachable() -> None:
    nested = _connection_error(socket.gaierror(socket.EAI_NONAME, "name not known"))
    assert classify_exception(nested, "http://missing.invalid:8080").exit_code is ExitCode.NETWORK_UNREACHABLE


def test_tls_validation_failure() -> None:
    nested = _connection_error(ssl.SSLCertVerificationError(1, "certificate verify failed"))
    assert classify_exception(nested, "https://agentcore.invalid").exit_code is ExitCode.TLS_ERROR


def test_timeout_after_stream_started_is_stream_error() -> None:
    timeout = _connection_error(httpx.ReadTimeout("stream stalled", request=_request()))
    failure = classify_exception(
        timeout,
        "http://127.0.0.1:8080",
        operation=OperationContext.STREAM,
        stream_started=True,
    )
    assert failure.exit_code is ExitCode.STREAM_ERROR


def test_generic_transport_failure_before_stream_start_is_network_unreachable() -> None:
    failure = classify_exception(
        AgentCoreConnectionError("transport failed"),
        "http://127.0.0.1:8080",
        operation=OperationContext.STREAM,
        stream_started=False,
    )
    assert failure.exit_code is ExitCode.NETWORK_UNREACHABLE


def test_unexpected_http_status_returns_http_error() -> None:
    assert _run(ExitCodeServer(health_status_code=503), []) is ExitCode.HTTP_ERROR


def test_protocol_major_mismatch_returns_protocol_incompatible() -> None:
    assert _run(ExitCodeServer(protocol_version="2.0"), []) is ExitCode.PROTOCOL_INCOMPATIBLE


def test_server_not_ready() -> None:
    server = ExitCodeServer(health_ready=False, runtime_ready=True)
    assert _run(server, []) is ExitCode.SERVER_NOT_READY


def test_runtime_not_ready() -> None:
    server = ExitCodeServer(health_ready=False, runtime_ready=False)
    assert _run(server, []) is ExitCode.RUNTIME_NOT_READY


def test_task_failed_terminal_event_wins_over_trailing_stream_error() -> None:
    server = ExitCodeServer(
        execution_status="failed",
        terminal_event="task.failed",
        malformed_after_terminal=True,
    )
    assert _run(server, ["Patch parser.c", "/approve", "/quit"]) is ExitCode.TASK_FAILED


def test_task_cancelled_terminal_event() -> None:
    server = ExitCodeServer(execution_status="cancelled", terminal_event="task.cancelled")
    assert _run(server, ["Patch parser.c", "/approve", "/quit"]) is ExitCode.TASK_CANCELLED


def test_explicit_rejection() -> None:
    assert _run(ExitCodeServer(), ["Patch parser.c", "/reject no", "/quit"]) is ExitCode.PROPOSAL_REJECTED


def test_noninteractive_pending_approval() -> None:
    assert _run(ExitCodeServer(), ["Patch parser.c"]) is ExitCode.APPROVAL_REQUIRED


def test_malformed_proposal_sse() -> None:
    server = ExitCodeServer(malformed_proposal_stream=True)
    assert _run(server, ["Patch parser.c"]) is ExitCode.STREAM_ERROR


def test_successful_abort_returns_task_cancelled() -> None:
    server = ExitCodeServer()
    assert _run(server, ["Patch parser.c", "/abort", "/quit"]) is ExitCode.TASK_CANCELLED
    assert server.cancelled is True


def test_unexpected_client_error_and_debug_have_same_code(capsys) -> None:
    def broken_factory(*args, **kwargs):
        raise RuntimeError("client defect")

    normal = run_cli(["--no-color"], inputs=[], client_factory=broken_factory)
    normal_stderr = capsys.readouterr().err
    debug = run_cli(["--no-color", "--debug"], inputs=[], client_factory=broken_factory)
    debug_stderr = capsys.readouterr().err
    assert normal is debug is ExitCode.INTERNAL_CLIENT_ERROR
    assert "Traceback" not in normal_stderr
    assert "Traceback" in debug_stderr


def test_failure_diagnostics_use_stderr(capsys) -> None:
    code = _run(ExitCodeServer(health_status_code=503), [])
    captured = capsys.readouterr()
    assert code is ExitCode.HTTP_ERROR
    assert "Category: HTTP_ERROR" in captured.err
    assert "Category: HTTP_ERROR" not in captured.out


def test_module_and_deprecated_wrapper_propagate_usage_code() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(ROOT / "packages" / "agentclient" / "src"),
            str(ROOT / "packages" / "agentcore-protocol" / "src"),
        )
    )
    module = subprocess.run(
        [sys.executable, "-m", "agentclient.cli", "--not-an-option"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wrapper = subprocess.run(
        [sys.executable, "scripts/agentcore_cli.py", "--not-an-option"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == int(ExitCode.CLI_USAGE_ERROR)
    assert wrapper.returncode == int(ExitCode.CLI_USAGE_ERROR)
    assert "deprecated" in wrapper.stderr


def _run(server: ExitCodeServer, inputs: list[str]) -> ExitCode:
    def factory(base_url: str, *, timeout: float) -> AgentCoreClient:
        transport = httpx.MockTransport(server.handler)
        http_client = httpx.Client(transport=transport, base_url=base_url, timeout=timeout)
        return AgentCoreClient(base_url, client=http_client)

    return run_cli(
        ["--server", "http://127.0.0.1:8080", "--workspace", "/server/workspace", "--no-color"],
        inputs=inputs,
        client_factory=factory,
    )


def _proposal(status: str = "proposed") -> dict[str, Any]:
    return {
        "id": "proposal-1",
        "title": "Patch parser",
        "status": status,
        "action_plan": {
            "title": "Patch parser",
            "actions": [{"type": "replace_text", "path": "parser.c", "old": "return 0", "new": "return 1"}],
        },
        "approval_requirements": [
            {"action_index": 0, "action_type": "replace_text", "reason": "mutating action requires approval"}
        ],
    }


def _proposal_stream() -> str:
    events = [
        AgentEvent(event_type="assistant.started", task_id="task-1", summary="started"),
        AgentEvent(event_type="assistant.delta", task_id="task-1", summary="delta", payload={"delta": "{}"}),
        AgentEvent(event_type="assistant.completed", task_id="task-1", summary="completed"),
        AgentEvent(
            event_type="plan.proposed",
            task_id="task-1",
            summary="proposed",
            payload={"proposal": _proposal()},
        ),
    ]
    return "".join(format_sse(event, event_id=index) for index, event in enumerate(events, start=1))


def _json_payload(request: httpx.Request) -> dict[str, Any]:
    import json

    data = json.loads(request.content.decode("utf-8"))
    assert isinstance(data, dict)
    return data


def _request() -> httpx.Request:
    return httpx.Request("GET", "http://127.0.0.1:8080/health")


def _connection_error(inner: BaseException) -> AgentCoreConnectionError:
    transport = httpx.ConnectError("connection failed", request=_request())
    transport.__cause__ = inner
    wrapper = AgentCoreConnectionError("connection failed")
    wrapper.__cause__ = transport
    return wrapper
