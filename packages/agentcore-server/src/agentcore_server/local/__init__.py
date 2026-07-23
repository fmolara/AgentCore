"""Single-process AgentCore composition and terminal runner."""

from .app import InvalidProposalError, LocalAgentCoreApp, LocalExecutionHandle
from .cli import LocalExitCode, main, run_cli
from .events import LocalEventSink

__all__ = [
    "InvalidProposalError",
    "LocalAgentCoreApp",
    "LocalEventSink",
    "LocalExecutionHandle",
    "LocalExitCode",
    "main",
    "run_cli",
]
