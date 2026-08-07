from __future__ import annotations

import argparse
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from agentcore_protocol import AgentCoreClient, AgentEvent

from agentclient.commands import HELP_TEXT, is_known_command, parse_command
from agentclient.config import ClientConfig, load_client_config
from agentclient.exit_codes import ExitCode
from agentclient.failures import (
    ClientFailure,
    ClientFailureError,
    OperationContext,
    classify_exception,
    describe_url,
)
from agentclient.rendering import Renderer


TERMINAL_EVENT_CODES = {
    "task.completed": ExitCode.SUCCESS,
    "execution.completed": ExitCode.SUCCESS,
    "task.failed": ExitCode.TASK_FAILED,
    "execution.failed": ExitCode.TASK_FAILED,
    "task.cancelled": ExitCode.TASK_CANCELLED,
    "execution.cancelled": ExitCode.TASK_CANCELLED,
    "cancellation.completed": ExitCode.TASK_CANCELLED,
}


@dataclass
class CLIState:
    agent_id: str | None = None
    task_id: str | None = None
    proposal_id: str | None = None
    instruction: str | None = None
    proposal: dict[str, Any] | None = None
    executed: bool = False
    rejected: bool = False
    events: list[AgentEvent] = field(default_factory=list)


class RemoteAgentCLI:
    def __init__(
        self,
        client: AgentCoreClient,
        *,
        config: ClientConfig,
        renderer: Renderer | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.renderer = renderer or Renderer(color=config.color)
        self.state = CLIState()
        self.final_exit_code = ExitCode.SUCCESS
        self._task_stream_error: BaseException | None = None
        self._task_stream_started = False
        self._task_terminal_code: ExitCode | None = None

    def connect(self) -> None:
        health = self.client.check_compatibility()
        runtime_ready = health.runtime.get("ready")
        runtime_error = health.runtime.get("last_error")
        if runtime_ready is False or runtime_error:
            cause = str(runtime_error or "configured inference runtime is not ready")
            raise ClientFailureError(_failure(ExitCode.RUNTIME_NOT_READY, cause, self.config.server_url))
        if not health.ready:
            cause = f"server health status is {health.status or 'not ready'}"
            raise ClientFailureError(_failure(ExitCode.SERVER_NOT_READY, cause, self.config.server_url))
        self.renderer.header(
            f"Connected to {self.config.server_url}\n"
            "Git commits are never automatic. Shell access is not available.\n"
            f"Protocol: {health.protocol_version or 'unknown'}"
        )

    def create_agent(self) -> None:
        agent = self.client.create_agent(
            system_prompt=self.config.system_prompt,
            workspace_root=self.config.default_workspace,
            workspace_metadata={"source": "agentclient"},
        )
        self.state.agent_id = agent.id
        workspace_root = agent.workspace.get("root") or self.config.default_workspace or "(server default)"
        self.renderer.info(f"Agent: {agent.id}")
        self.renderer.info(f"Workspace: {workspace_root}")

    def handle_line(self, line: str) -> bool:
        parsed = parse_command(line)
        if not parsed.name:
            return True
        if parsed.name == "instruction":
            self.start_task(parsed.argument)
            return True
        if not is_known_command(parsed.name):
            self.renderer.error(f"Unknown command: {parsed.name}")
            return True
        if parsed.name == "/quit":
            return False
        if parsed.name == "/help":
            self.renderer.section("Help", HELP_TEXT)
        elif parsed.name == "/status":
            self.show_status()
        elif parsed.name == "/plan":
            instruction = parsed.argument or self.state.instruction
            if instruction is None:
                self.renderer.error("No instruction available. Enter a task first.")
            else:
                self.request_plan(instruction)
        elif parsed.name == "/approve":
            self.approve_and_execute()
        elif parsed.name == "/reject":
            self.reject(parsed.argument or "rejected by user")
        elif parsed.name == "/diff":
            self.show_diff()
        elif parsed.name == "/report":
            self.show_report()
        elif parsed.name == "/abort":
            self.abort(parsed.argument or "aborted by user")
        return True

    def start_task(self, instruction: str) -> None:
        if self.state.agent_id is None:
            raise RuntimeError("agent has not been created")
        title = instruction[:80] or "AgentClient task"
        task = self.client.create_task(self.state.agent_id, title=title, description=instruction)
        self.state = CLIState(agent_id=self.state.agent_id, task_id=task.id, instruction=instruction)
        self.renderer.info(f"Task: {task.id}")
        self.request_plan(instruction)

    def request_plan(self, instruction: str) -> None:
        if self.state.task_id is None:
            self.start_task(instruction)
            return
        self.state.instruction = instruction
        proposal_events: list[AgentEvent] = []
        try:
            for event in self.client.stream_proposal(
                self.state.task_id,
                instruction=instruction,
                max_tokens=768,
                temperature=0,
            ):
                proposal_events.append(event)
                self.state.events.append(event)
                self.renderer.event(event)
        except Exception as exc:
            raise ClientFailureError(
                classify_exception(
                    exc,
                    self.config.server_url,
                    operation=OperationContext.STREAM,
                    stream_started=bool(proposal_events),
                )
            ) from exc
        proposal = _proposal_from_events(proposal_events)
        if proposal is None:
            raise ClientFailureError(
                _failure(ExitCode.STREAM_ERROR, "proposal stream ended without plan.proposed", self.config.server_url)
            )
        self.state.proposal = proposal
        self.state.proposal_id = str(proposal["id"])
        self.state.executed = False
        self.state.rejected = False
        self.renderer.plan(proposal)
        self.renderer.approvals(proposal)
        self.renderer.info("Use /approve to execute, /reject to reject, /diff to inspect, /quit to exit.")

    def approve_and_execute(self) -> None:
        if self.state.proposal_id is None or self.state.task_id is None:
            self.renderer.error("No proposal to approve.")
            return
        if self.state.executed:
            self.renderer.info("Proposal already executed.")
            return
        if self.state.rejected:
            self.renderer.error("Cannot approve a rejected proposal.")
            return
        self.client.approve_proposal(self.state.proposal_id)
        self.renderer.info("Proposal approved. Executing...")
        self._task_stream_error = None
        self._task_stream_started = False
        self._task_terminal_code = None
        event_thread = threading.Thread(target=self._collect_task_events, daemon=True)
        event_thread.start()
        result = self.client.execute_proposal(self.state.proposal_id)
        event_thread.join(timeout=30)
        self.renderer.json("Execution Result", result.raw)

        result_code = _execution_status_code(result.status)
        terminal_code = self._task_terminal_code if self._task_terminal_code is not None else result_code
        if result_code in (ExitCode.TASK_FAILED, ExitCode.TASK_CANCELLED):
            terminal_code = result_code
        if terminal_code in (ExitCode.TASK_FAILED, ExitCode.TASK_CANCELLED):
            self._record_outcome(terminal_code)
            return
        if event_thread.is_alive():
            raise ClientFailureError(
                _failure(ExitCode.STREAM_ERROR, "task event stream did not terminate", self.config.server_url)
            )
        if self._task_stream_error is not None and self._task_terminal_code is None:
            raise ClientFailureError(
                classify_exception(
                    self._task_stream_error,
                    self.config.server_url,
                    operation=OperationContext.STREAM,
                    stream_started=self._task_stream_started,
                )
            ) from self._task_stream_error
        if self._task_terminal_code is None:
            raise ClientFailureError(
                _failure(
                    ExitCode.STREAM_ERROR,
                    "task event stream ended without a terminal event",
                    self.config.server_url,
                )
            )
        self.state.executed = terminal_code is ExitCode.SUCCESS

    def reject(self, reason: str) -> None:
        if self.state.proposal_id is None:
            self.renderer.error("No proposal to reject.")
            return
        if self.state.executed:
            self.renderer.error("Cannot reject an executed proposal.")
            return
        proposal = self.client.reject_proposal(self.state.proposal_id, reason)
        self.state.rejected = True
        self._record_outcome(ExitCode.PROPOSAL_REJECTED)
        self.renderer.json("Proposal Rejected", proposal.raw)

    def show_status(self) -> None:
        self.renderer.json(
            "Status",
            {
                "agent_id": self.state.agent_id,
                "task_id": self.state.task_id,
                "proposal_id": self.state.proposal_id,
                "executed": self.state.executed,
                "rejected": self.state.rejected,
            },
        )

    def show_diff(self) -> None:
        if self.state.agent_id is None:
            self.renderer.error("No agent is active.")
            return
        self.renderer.diff(self.client.git_diff(self.state.agent_id).stdout)

    def show_report(self) -> None:
        if self.state.task_id is None:
            self.renderer.error("No task is active.")
            return
        self.renderer.json("Task Report", self.client.task_report(self.state.task_id))

    def abort(self, reason: str) -> None:
        if self.state.task_id is None:
            self.renderer.error("No task is active.")
            return
        task = self.client.cancel_task(self.state.task_id, reason)
        self._record_outcome(ExitCode.TASK_CANCELLED)
        self.renderer.json("Cancellation Requested", task.raw)

    def run(self, inputs: Iterable[str] | None = None) -> ExitCode:
        self.connect()
        self.create_agent()
        self.renderer.section("Help", HELP_TEXT)
        non_interactive = inputs is not None
        source = inputs if inputs is not None else _interactive_input()
        for line in source:
            try:
                if not self.handle_line(line):
                    break
            except Exception as exc:
                failure = classify_exception(exc, self.config.server_url)
                self.renderer.failure(failure)
                if self.renderer.debug:
                    self.renderer.traceback(exc, redact=self.config.server_url)
                self._record_outcome(failure.exit_code)
                if non_interactive:
                    break
        if non_interactive and self.final_exit_code is ExitCode.SUCCESS and self._approval_is_pending():
            return ExitCode.APPROVAL_REQUIRED
        return self.final_exit_code

    def _collect_task_events(self) -> None:
        if self.state.task_id is None:
            return
        try:
            for event in self.client.stream_task_events(self.state.task_id):
                self._task_stream_started = True
                self.state.events.append(event)
                terminal_code = TERMINAL_EVENT_CODES.get(event.event_type)
                if terminal_code is not None:
                    self._task_terminal_code = terminal_code
                self.renderer.event(event)
        except Exception as exc:
            self._task_stream_error = exc

    def _approval_is_pending(self) -> bool:
        if self.state.proposal is None or self.state.executed or self.state.rejected:
            return False
        requirements = self.state.proposal.get("approval_requirements")
        return isinstance(requirements, list) and bool(requirements)

    def _record_outcome(self, code: ExitCode) -> None:
        if self.final_exit_code is ExitCode.SUCCESS and code is not ExitCode.SUCCESS:
            self.final_exit_code = code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote AgentCore CLI.")
    parser.add_argument("--server", default=None, help="AgentCore server URL.")
    parser.add_argument("--workspace", default=None, help="Server-side workspace path.")
    parser.add_argument("--system-prompt", default=None, help="System prompt for the remote agent.")
    parser.add_argument("--config", default=None, help="Client config TOML path.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--debug", action="store_true", help="Include exception tracebacks in diagnostics.")
    return parser.parse_args(argv)


def run_cli(
    argv: list[str] | None = None,
    *,
    inputs: Iterable[str] | None = None,
    client_factory: Callable[..., AgentCoreClient] = AgentCoreClient,
) -> ExitCode:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return ExitCode.SUCCESS if exc.code == 0 else ExitCode.CLI_USAGE_ERROR

    renderer = Renderer(color=not args.no_color, debug=args.debug)
    try:
        config = load_client_config(args.config).with_overrides(
            server_url=args.server,
            default_workspace=args.workspace,
            system_prompt=args.system_prompt,
            color=False if args.no_color else None,
        )
        _validate_server_url(config.server_url)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        url = args.server or ClientConfig.server_url
        failure = _failure(ExitCode.CLI_USAGE_ERROR, f"invalid client configuration: {exc}", url)
        renderer.failure(failure)
        if args.debug:
            renderer.traceback(exc, redact=url)
        return failure.exit_code

    renderer = Renderer(color=config.color, debug=args.debug)
    effective_inputs = inputs
    if effective_inputs is None and not sys.stdin.isatty():
        effective_inputs = sys.stdin
    try:
        with client_factory(config.server_url, timeout=config.request_timeout_sec) as client:
            return RemoteAgentCLI(client, config=config, renderer=renderer).run(effective_inputs)
    except Exception as exc:
        failure = classify_exception(exc, config.server_url, operation=OperationContext.CONNECT)
        renderer.failure(failure)
        if args.debug:
            renderer.traceback(exc, redact=config.server_url)
        return failure.exit_code


def main(argv: list[str] | None = None) -> int:
    return int(run_cli(argv))


def _proposal_from_events(events: Iterable[AgentEvent]) -> dict[str, Any] | None:
    for event in events:
        if event.event_type == "plan.proposed":
            proposal = event.payload.get("proposal")
            if isinstance(proposal, dict):
                return proposal
    return None


def _execution_status_code(status: str) -> ExitCode | None:
    normalized = status.lower()
    if normalized in {"completed", "success", "succeeded"}:
        return ExitCode.SUCCESS
    if normalized in {"cancelled", "canceled"}:
        return ExitCode.TASK_CANCELLED
    if normalized in {"failed", "error"}:
        return ExitCode.TASK_FAILED
    return None


def _failure(code: ExitCode, cause: str, url: str) -> ClientFailure:
    safe_url, host, port = describe_url(url)
    return ClientFailure(code, cause, safe_url, host, port)


def _validate_server_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("server URL must be an absolute HTTP or HTTPS URL")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("server URL contains an invalid port") from exc


def _interactive_input() -> Iterable[str]:
    while True:
        try:
            yield input("agentclient> ")
        except EOFError:
            return


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
