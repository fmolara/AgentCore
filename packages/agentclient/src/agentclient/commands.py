from __future__ import annotations

from dataclasses import dataclass


COMMANDS = {
    "/status",
    "/plan",
    "/approve",
    "/reject",
    "/diff",
    "/report",
    "/abort",
    "/quit",
    "/help",
}

HELP_TEXT = """\
Commands:
  /status   Show current remote task/proposal state
  /plan     Request or re-request a streamed plan proposal
  /approve  Explicitly approve and execute the current proposal
  /reject   Reject the current proposal
  /diff     Fetch Git diff from the remote workspace
  /report   Fetch the current task report
  /abort    Request cooperative cancellation for the current task
  /quit     Exit
  /help     Show this help

Enter a plain instruction to create a new task and request a plan.
"""


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    argument: str = ""

    @property
    def is_command(self) -> bool:
        return self.name.startswith("/")


def parse_command(line: str) -> ParsedCommand:
    stripped = line.strip()
    if not stripped:
        return ParsedCommand("")
    if not stripped.startswith("/"):
        return ParsedCommand("instruction", stripped)
    name, _, argument = stripped.partition(" ")
    return ParsedCommand(name, argument.strip())


def is_known_command(name: str) -> bool:
    return name in COMMANDS
