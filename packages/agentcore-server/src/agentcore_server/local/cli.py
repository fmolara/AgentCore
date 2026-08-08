from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
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
from .qwen_tools import LocalToolLoopApp, LocalToolLoopHandle
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
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", help="AgentCore runtime configuration")
    config_group.add_argument(
        "--profile",
        choices=("fast", "strong"),
        help="Use an installed local operational profile (default: fast for tool-loop mode)",
    )
    parser.add_argument("--workspace", required=True, help="Local workspace root")
    parser.add_argument("--system-prompt", help="Override the selected mode's system prompt")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--planner",
        choices=("simple", "iterative"),
        help="Override planner.mode from configuration",
    )
    mode_group.add_argument(
        "--agent",
        choices=("tool-loop", "qwen-tools"),
        help="Run the incremental native tool agent instead of a planner workflow",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Task instruction")
    prompt_group.add_argument("--prompt-file", help="Read the task instruction from a UTF-8 file")
    parser.add_argument("--proposal-only", action="store_true", help="Propose a plan without execution")
    parser.add_argument("--approve", action="store_true", help="Explicitly approve non-interactive execution")
    parser.add_argument("--trace-file", help="Write ordered public events as JSONL")
    parser.add_argument(
        "--approval-preview-dir",
        help="Retain complete native-tool approval previews in this local directory",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-warmup", action="store_true", help="Skip runtime warmup")
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    lab_factory=AgentLab.from_config,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        _validate_args(args)
        instruction = _instruction_from_args(args)
        config_path, selected_profile = _resolve_config(args)
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
        lab = lab_factory(config_path)
        trace_file = args.trace_file
        if trace_file is None and args.agent in {"tool-loop", "qwen-tools"}:
            trace_file = _automatic_trace_file(lab)
        metrics_file = _configured_path(lab, "metrics_path")
        sink = LocalEventSink(
            renderer=renderer.render_event,
            trace_file=trace_file,
            metrics_file=metrics_file,
            metrics_context=_metrics_context(lab, selected_profile),
        )
        if args.agent in {"tool-loop", "qwen-tools"}:
            app = LocalToolLoopApp(
                lab,
                workspace=args.workspace,
                system_prompt=args.system_prompt,
                event_sink=sink,
                approval_presenter=renderer.show_tool_approval,
                preview_directory=args.approval_preview_dir,
            )
        else:
            app = LocalAgentCoreApp(
                lab,
                workspace=args.workspace,
                system_prompt=args.system_prompt or "You are a concise coding agent.",
                planner_mode=args.planner,
                event_sink=sink,
            )
        if trace_file is not None and args.trace_file is None:
            renderer.info(f"Automatic trace: {trace_file}")
        if metrics_file is not None:
            renderer.info(f"Passive metrics: {metrics_file}")
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

        if args.agent in {"tool-loop", "qwen-tools"}:
            assert isinstance(app, LocalToolLoopApp)
            return int(
                _run_tool_loop_interactive(
                    app,
                    renderer,
                    instruction=instruction,
                    stdin=stdin,
                    stdout=stdout,
                )
            )
        if args.proposal_only or args.approve:
            assert instruction is not None
            return int(_run_noninteractive(app, renderer, instruction, args))
        return int(
            _run_interactive(
                app,
                renderer,
                instruction=instruction,
                stdin=stdin,
                stdout=stdout,
            )
        )
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
    if args.agent in {"tool-loop", "qwen-tools"} and (args.proposal_only or args.approve):
        raise CliUsageError("native tool-agent mode cannot be combined with --proposal-only or --approve")
    if args.approval_preview_dir and args.agent not in {"tool-loop", "qwen-tools"}:
        raise CliUsageError("--approval-preview-dir requires native tool-agent mode")
    workspace = Path(args.workspace).expanduser()
    if not workspace.exists():
        raise CliUsageError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise CliUsageError(f"workspace is not a directory: {workspace}")


def _resolve_config(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.config:
        return args.config, args.profile
    if args.agent not in {"tool-loop", "qwen-tools"}:
        raise CliUsageError("--config is required for legacy planner mode")
    profile = args.profile or "fast"
    candidates: list[Path] = []
    profile_dir = os.environ.get("AGENTCORE_PROFILE_DIR")
    if profile_dir:
        candidates.append(Path(profile_dir).expanduser() / f"{profile}.yaml")
    candidates.extend(
        [
            Path.home() / ".config" / "agentcore" / "profiles" / f"{profile}.yaml",
            Path.cwd() / "config" / "local" / f"{profile}.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve()), profile
    searched = ", ".join(str(path) for path in candidates)
    raise CliUsageError(
        f"local profile '{profile}' is not installed; searched: {searched}. "
        "Use --config or install the profile under ~/.config/agentcore/profiles."
    )


def _configured_path(lab: AgentLab, key: str) -> str | None:
    logging_config = lab.config.get("logging", {})
    if not isinstance(logging_config, dict):
        raise CliUsageError("logging configuration must be a mapping")
    value = logging_config.get(key)
    if value in {None, ""}:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = lab.project_root / path
    return str(path.resolve())


def _automatic_trace_file(lab: AgentLab) -> str | None:
    directory = _configured_path(lab, "trace_path")
    if directory is None:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return str(Path(directory) / f"tool-loop-{stamp}-{os.getpid()}.jsonl")


def _metrics_context(lab: AgentLab, profile: str | None) -> dict[str, str | None]:
    model_config = lab.config.get("model", {})
    model_value = model_config.get("name") or model_config.get("path")
    model_name = None if model_value is None else Path(str(model_value)).name
    return {
        "profile": profile or "custom",
        "runtime": str(lab.config.get("runtime") or "unknown"),
        "model": model_name,
    }


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


def _run_interactive(
    app: LocalAgentCoreApp,
    renderer: TerminalRenderer,
    *,
    instruction: str | None,
    stdin: TextIO,
    stdout: TextIO,
) -> LocalExitCode:
    renderer.info("AgentCore local mode. Git commits are never automatic. Type /help for commands.")
    if instruction is None:
        try:
            instruction = _read_terminal_line("Task> ", stdin=stdin, stdout=stdout).strip()
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
            command = _read_terminal_line("agentcore> ", stdin=stdin, stdout=stdout).strip()
        except EOFError:
            if handle is not None:
                handle.wait()
                return _completed_handle_code(handle)
            return LocalExitCode.APPROVAL_REQUIRED
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


def _run_tool_loop_interactive(
    app: LocalToolLoopApp,
    renderer: TerminalRenderer,
    *,
    instruction: str | None,
    stdin: TextIO,
    stdout: TextIO,
) -> LocalExitCode:
    renderer.info(
        "AgentCore native tool mode. Side effects require per-call approval; Git commits are never automatic."
    )
    if instruction is None:
        try:
            instruction = _read_terminal_line("Task> ", stdin=stdin, stdout=stdout)
        except EOFError:
            return LocalExitCode.SUCCESS
        if not instruction.strip():
            renderer.error("task instruction must not be empty")
            return LocalExitCode.CLI_OR_CONFIG_ERROR
    task = app.create_task(instruction)
    handle = app.run_async(task, instruction)
    renderer.info(
        "Commands: /status /diff /report /preview /approve /reject /abort /quit /help. "
        "Plain text queues one steering message."
    )
    while True:
        if not handle.running:
            code = _completed_tool_loop_handle_code(handle)
            renderer.show_report(task.report())
            renderer.show_diff(app.diff())
            return code
        try:
            command = _read_terminal_line("agentcore> ", stdin=stdin, stdout=stdout)
        except EOFError:
            app.cancel(task, "terminal input closed")
            handle.wait()
            return _completed_tool_loop_handle_code(handle)
        if not command.strip():
            continue
        if not command.startswith("/"):
            if app.queue_steering(command):
                renderer.info("Steering message queued for the next model turn.")
            else:
                renderer.error("one steering message is already queued")
            continue
        name, _, argument = command.partition(" ")
        if name == "/help":
            renderer.info(
                "/status /diff /report /preview /approve /reject /abort /quit /help; "
                "plain text queues steering"
            )
        elif name == "/status":
            renderer.show_status(app.status(task))
        elif name == "/diff":
            renderer.show_diff(app.diff())
        elif name == "/report":
            renderer.show_report(app.report(task))
        elif name == "/preview":
            pending = app.approval_gateway.wait_for_pending(timeout=1.0)
            if pending is None:
                renderer.error("no tool approval is pending")
            else:
                renderer.show_tool_approval(pending)
        elif name in {"/approve", "/reject"}:
            pending = app.approval_gateway.wait_for_pending(timeout=1.0)
            if pending is None:
                renderer.error("no tool approval is pending")
                continue
            try:
                if name == "/approve":
                    app.approve_pending()
                    renderer.info(f"Approved tool call {pending.call.id} only.")
                else:
                    app.reject_pending(argument.strip() or "rejected by local operator")
                    renderer.info(f"Rejected tool call {pending.call.id} only.")
            except ValueError as exc:
                renderer.error(str(exc))
        elif name == "/abort":
            app.cancel(task, argument.strip() or "aborted by local operator")
            handle.wait()
            return _completed_tool_loop_handle_code(handle)
        elif name == "/quit":
            if handle.running:
                app.cancel(task, "local operator quit")
                handle.wait()
            return _completed_tool_loop_handle_code(handle)
        else:
            renderer.error(f"unknown command: {name}")


def _read_terminal_line(prompt: str, *, stdin: TextIO, stdout: TextIO) -> str:
    stdout.write(prompt)
    stdout.flush()
    line = stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


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


def _completed_tool_loop_handle_code(handle: LocalToolLoopHandle) -> LocalExitCode:
    if handle.error is not None:
        raise handle.error
    if handle.result is None:
        return LocalExitCode.INTERNAL_ERROR
    return {
        "completed": LocalExitCode.SUCCESS,
        "failed": LocalExitCode.TASK_FAILED,
        "cancelled": LocalExitCode.TASK_CANCELLED,
    }.get(handle.result.status, LocalExitCode.INTERNAL_ERROR)


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
