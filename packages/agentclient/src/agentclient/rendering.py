from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from agentcore_protocol import AgentEvent

from agentclient.failures import ClientFailure

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
except Exception:  # pragma: no cover - defensive fallback for unusual installs.
    Console = None
    Panel = None
    Syntax = None


class Renderer:
    def __init__(self, *, color: bool = True, debug: bool = False) -> None:
        self.debug = debug
        self.console = Console(no_color=not color) if Console is not None else None
        self.error_console = Console(no_color=not color, stderr=True) if Console is not None else None

    def header(self, text: str) -> None:
        if self.console is not None and Panel is not None:
            self.console.print(Panel(text, title="AgentClient"))
        else:
            print(f"== AgentClient ==\n{text}")

    def info(self, text: str) -> None:
        if self.console is not None:
            self.console.print(text)
        else:
            print(text)

    def error(self, text: str) -> None:
        if self.error_console is not None:
            self.error_console.print(text, style="red", markup=False)
        else:
            print(f"ERROR: {text}", file=sys.stderr)

    def failure(self, failure: ClientFailure) -> None:
        self.error(failure.as_text())

    def traceback(self, exc: BaseException, *, redact: str | None = None) -> None:
        text = "".join(traceback.format_exception(exc))
        if redact:
            text = text.replace(redact, "<redacted-url>")
        if self.error_console is not None:
            self.error_console.print(text, markup=False, highlight=False)
        else:
            print(text, file=sys.stderr, end="" if text.endswith("\n") else "\n")

    def section(self, title: str, body: str) -> None:
        if self.console is not None and Panel is not None:
            self.console.print(Panel(body or "(empty)", title=title))
        else:
            print(f"\n== {title} ==\n{body or '(empty)'}")

    def json(self, title: str, data: Any) -> None:
        text = json.dumps(data, indent=2, sort_keys=True)
        if self.console is not None and Syntax is not None:
            self.console.print(Panel(Syntax(text, "json", word_wrap=True), title=title))
        else:
            self.section(title, text)

    def event(self, event: AgentEvent) -> None:
        if event.event_type == "assistant.started":
            self.info("Assistant response:")
            return
        if event.event_type == "assistant.delta":
            delta = str(event.payload.get("delta", ""))
            if self.console is not None:
                self.console.print(delta, end="")
            else:
                print(delta, end="", flush=True)
            return
        if event.event_type == "assistant.completed":
            if self.console is not None:
                self.console.print()
            else:
                print()
            return
        if event.event_type == "assistant.failed":
            self.error(str(event.payload.get("error", event.summary)))
            return
        if event.event_type == "workspace.modified":
            files = event.payload.get("files_changed", [])
            suffix = "" if not files else " " + ", ".join(str(item) for item in files)
            self.info(f"[{event.event_type}] {event.summary}{suffix}")
            return
        self.info(f"[{event.event_type}] {event.summary}")

    def plan(self, proposal: dict[str, Any]) -> None:
        self.json("Proposed ActionPlan", proposal.get("action_plan", {}))

    def approvals(self, proposal: dict[str, Any]) -> None:
        self.json("Approval Requirements", proposal.get("approval_requirements", []))

    def diff(self, diff_text: str) -> None:
        if self.console is not None and Syntax is not None and diff_text:
            self.console.print(Panel(Syntax(diff_text, "diff", word_wrap=True), title="Git Diff"))
        else:
            self.section("Git Diff", diff_text or "(empty)")
