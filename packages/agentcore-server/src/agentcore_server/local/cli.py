from __future__ import annotations

import argparse
import sys
import traceback
from enum import IntEnum
from pathlib import Path
from typing import Sequence, TextIO

from agentcore_server.api.client import AgentLab
from agentcore_server.executor import PlanProposal, PlanProposalStatus
from agentcore_server.tasks import TaskStatus

from .approval import StaticApprovalGateway
from .app import InvalidProposalError, LocalAgentCoreApp, LocalExecutionHandle
from .events import LocalEventSink
from .rendering import TerminalRenderer


class LocalExitCode(IntEnum):
    SUCCESS = 0
    CLI_OR_CONFIG_ERROR = 2
    RUNTIME_UNAVAILABLE = 10
    INVALID_PROPOSAL = 20
    PROPOSAL_REJECTED = 21
    APPROVAL_REQUIRED = 22
    TASK_FAILED = 23
    TASK_CANCELLED = 24
    INTERNAL_ERROR = 70


class CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="agentcore-local",
        description="Run AgentCore orchestration locally without HTTP or agentclient.",
    )
    parser.add_argument("--config", required=True, help="AgentCore runtime configuration")
    parser.add_argument("--workspace", required=True, help="Local workspace root")
    parser.add_argument("--system-prompt", default="You are a concise coding agent.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Task instruction")
    prompt_group.add_argument("--prompt-file", help="Read the task instruction from a UTF-8 file")
    parser.add_argument("--proposal-only", action="store_true", help="Propose a plan without execution")
    parser.add_argument("--approve", action="store_true", help="Explicitly approve non-interactive execution")
    parser.add_argument("--trace-file", help="Write ordered public events as JSONL")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-warmup", action="store_true", help="Skip runtime warmup")
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    lab_factory=AgentLab.from_config,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        _validate_args(args)
        instruction = _instruction_from_args(args)
    except CliUsageError as exc:
        stderr.write(f"agentcore-local: {exc}\n")
        return int(LocalExitCode.CLI_OR_CONFIG_ERROR)
    except OSError as exc:
        stderr.write(f"agentcore-local: cannot read prompt: {exc}\n")
        return int(LocalExitCode.CLI_OR_CONFIG_ERROR)
    except SystemExit as exc:
        return int(exc.code)

    renderer = TerminalRenderer(
        stdout=stdout,
        stderr=stderr,
        color=not args.no_color,
        debug=args.debug,
    )
    try:
        lab = lab_factory(args.config)
        sink = LocalEventSink(renderer=renderer.render_event, trace_file=args.trace_file)
        app = LocalAgentCoreApp(
            lab,
            workspace=args.workspace,
            system_prompt=args.system_prompt,
            event_sink=sink,
        )
    except Exception as exc:
        _render_exception(renderer, exc, debug=args.debug)
        return int(LocalExitCode.CLI_OR_CONFIG_ERROR)

    started = False
    try:
        try:
            app.start(warmup=not args.no_warmup)
            started = True
        except Exception as exc:
            _render_exception(renderer, exc, debug=args.debug)
            return int(LocalExitCode.RUNTIME_UNAVAILABLE)

        if instruction is None:
            return int(_run_interactive(app, renderer))
        return int(_run_noninteractive(app, renderer, instruction, args))
    except InvalidProposalError as exc:
        _render_exception(renderer, exc, debug=args.debug)
        return int(LocalExitCode.INVALID_PROPOSAL)
    except KeyboardInterrupt:
        renderer.error("interrupted")
        return int(LocalExitCode.TASK_CANCELLED)
    except Exception as exc:
        _render_exception(renderer, exc, debug=args.debug)
        return int(LocalExitCode.INTERNAL_ERROR)
    finally:
        if started:
            try:
                app.shutdown()
            except Exception as exc:
                _render_exception(renderer, exc, debug=args.debug)


def main() -> None:
    raise SystemExit(run_cli())


def _validate_args(args: argparse.Namespace) -> None:
    if args.proposal_only and args.approve:
        raise CliUsageError("--proposal-only and --approve cannot be used together")
    if (args.proposal_only or args.approve) and not (args.prompt or args.prompt_file):
        raise CliUsageError("--proposal-only and --approve require --prompt or --prompt-file")
    workspace = Path(args.workspace).expanduser()
    if not workspace.exists():
        raise CliUsageError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise CliUsageError(f"workspace is not a directory: {workspace}")


def _instruction_from_args(args: argparse.Namespace) -> str | None:
    if args.prompt is not None:
        instruction = args.prompt
    elif args.prompt_file is not None:
        instruction = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    else:
        return None
    if not instruction.strip():
        raise CliUsageError("task instruction must not be empty")
    return instruction


def _run_noninteractive(
    app: LocalAgentCoreApp,
    renderer: TerminalRenderer,
    instruction: str,
    args: argparse.Namespace,
) -> LocalExitCode:
    task = app.create_task(instruction)
    result = app.propose(task, instruction, stream=True)
    proposal = _proposal_from_result(result)
    renderer.show_plan(proposal)
    renderer.show_approval_requirements(proposal)
    if args.proposal_only:
        renderer.info("Proposal-only mode: no actions were executed.")
        return LocalExitCode.SUCCESS
    approval = StaticApprovalGateway(args.approve)
    if not approval.approve(proposal):
        renderer.error("explicit --approve is required for non-interactive execution")
        return LocalExitCode.APPROVAL_REQUIRED

    app.approve(task, proposal)
    execution = app.execute(task, proposal)
    renderer.show_report(task.report())
    renderer.show_diff(app.diff())
    return _execution_exit_code(execution.status)


def _run_interactive(app: LocalAgentCoreApp, renderer: TerminalRenderer) -> LocalExitCode:
    renderer.info("AgentCore local mode. Git commits are never automatic. Type /help for commands.")
    try:
        instruction = input("Task> ").strip()
    except EOFError:
        return LocalExitCode.SUCCESS
    if not instruction:
        renderer.error("task instruction must not be empty")
        return LocalExitCode.CLI_OR_CONFIG_ERROR

    task = app.create_task(instruction)
    result = app.propose(task, instruction, stream=True)
    proposal = _proposal_from_result(result)
    renderer.show_plan(proposal)
    renderer.show_approval_requirements(proposal)
    handle: LocalExecutionHandle | None = None

    while True:
        if handle is not None and not handle.running:
            code = _completed_handle_code(handle)
            renderer.show_report(task.report())
            renderer.show_diff(app.diff())
            return code
        try:
            command = input("agentcore> ").strip()
        except EOFError:
            command = "/quit"
        if not command:
            continue
        if not command.startswith("/"):
            renderer.error("commands must start with '/'; use /help")
            continue
        name, _, argument = command.partition(" ")
        if name == "/help":
            renderer.info(
                "/status /plan /approve /reject [reason] /diff /report "
                "/abort /quit /help"
            )
        elif name == "/status":
            renderer.show_status(app.status(task, proposal))
        elif name == "/plan":
            renderer.show_plan(proposal)
            renderer.show_approval_requirements(proposal)
        elif name == "/approve":
            if handle is not None:
                renderer.error("execution has already started")
                continue
            try:
                app.approve(task, proposal)
                handle = app.execute_async(task, proposal)
                renderer.info("Execution started. Use /abort to request cooperative cancellation.")
            except ValueError as exc:
                renderer.error(str(exc))
        elif name == "/reject":
            if handle is not None:
                renderer.error("cannot reject after execution has started")
                continue
            reason = argument.strip() or "rejected by local operator"
            app.reject(task, proposal, reason)
            return LocalExitCode.PROPOSAL_REJECTED
        elif name == "/diff":
            renderer.show_diff(app.diff())
        elif name == "/report":
            renderer.show_report(app.report(task))
        elif name == "/abort":
            try:
                app.cancel(task, argument.strip() or "aborted by local operator")
            except ValueError as exc:
                renderer.error(str(exc))
                continue
            if handle is not None:
                handle.wait()
            return LocalExitCode.TASK_CANCELLED
        elif name == "/quit":
            if handle is not None and handle.running:
                app.cancel(task, "local operator quit")
                handle.wait()
                return LocalExitCode.TASK_CANCELLED
            return _task_exit_code(task.status, proposal.status)
        else:
            renderer.error(f"unknown command: {name}")


def _proposal_from_result(result) -> PlanProposal:
    if result.proposal is None:
        raise InvalidProposalError(result.error or "planner did not produce a proposal", result=result)
    return result.proposal


def _completed_handle_code(handle: LocalExecutionHandle) -> LocalExitCode:
    if handle.error is not None:
        raise handle.error
    if handle.result is None:
        return LocalExitCode.INTERNAL_ERROR
    return _execution_exit_code(handle.result.status)


def _execution_exit_code(status: str) -> LocalExitCode:
    return {
        "completed": LocalExitCode.SUCCESS,
        "approval_required": LocalExitCode.APPROVAL_REQUIRED,
        "failed": LocalExitCode.TASK_FAILED,
        "cancelled": LocalExitCode.TASK_CANCELLED,
    }.get(status, LocalExitCode.INTERNAL_ERROR)


def _task_exit_code(task_status: TaskStatus, proposal_status: PlanProposalStatus) -> LocalExitCode:
    if proposal_status == PlanProposalStatus.REJECTED:
        return LocalExitCode.PROPOSAL_REJECTED
    if task_status == TaskStatus.FAILED:
        return LocalExitCode.TASK_FAILED
    if task_status == TaskStatus.CANCELLED:
        return LocalExitCode.TASK_CANCELLED
    return LocalExitCode.SUCCESS


def _render_exception(renderer: TerminalRenderer, exc: BaseException, *, debug: bool) -> None:
    renderer.error(str(exc) or exc.__class__.__name__)
    if debug:
        traceback.print_exception(exc, file=renderer.stderr)
