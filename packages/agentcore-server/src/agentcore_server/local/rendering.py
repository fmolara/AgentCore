from __future__ import annotations

import json
import sys
from typing import TextIO

from agentcore_server.events import AgentEvent
from agentcore_server.executor import PlanProposal
from agentcore_server.tasks import TaskReport
from agentcore_server.tool_agent import ToolApprovalRequest


_DIAGNOSTIC_EVENTS = {
    "planner.prompt",
    "planner.raw_output",
    "planner.result",
    "planner.validation",
    "approval.policy",
}


class TerminalRenderer:
    def __init__(
        self,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        color: bool = True,
        debug: bool = False,
    ) -> None:
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.color = color and bool(getattr(self.stdout, "isatty", lambda: False)())
        self.debug = debug
        self._assistant_open = False

    def render_event(self, event: AgentEvent) -> None:
        if event.event_type in _DIAGNOSTIC_EVENTS and not self.debug:
            return
        if event.event_type == "assistant.started":
            self._assistant_open = True
            self._write(self.stdout, self._style("Assistant: ", "cyan"), end="")
            return
        if event.event_type == "assistant.delta":
            self._write(self.stdout, str(event.payload.get("delta", "")), end="")
            return
        if event.event_type == "assistant.completed":
            if self._assistant_open:
                self._write(self.stdout, "")
            self._assistant_open = False
            return
        if event.event_type == "assistant.failed":
            self._assistant_open = False
            self.error(event.payload.get("error", event.summary))
            return
        if event.event_type == "tool.approval.required":
            self.heading("Tool approval required")
            self._write(self.stdout, json.dumps(event.payload, indent=2, sort_keys=True))
            self._write(
                self.stdout,
                "The complete AgentCore-generated effect preview follows separately. "
                "Use /preview to show it again, then /approve or /reject.",
            )
            return
        stream = self.stderr if event.event_type.endswith(".failed") else self.stdout
        self._write(stream, f"[{event.event_type}] {event.summary}")
        if self.debug and event.event_type in _DIAGNOSTIC_EVENTS:
            self._write(stream, json.dumps(event.payload, indent=2, sort_keys=True))

    def show_plan(self, proposal: PlanProposal) -> None:
        self.heading("Plan proposal")
        self._write(self.stdout, json.dumps(proposal.as_dict(), indent=2, sort_keys=True))

    def show_approval_requirements(self, proposal: PlanProposal) -> None:
        self.heading("Approval requirements")
        if not proposal.approval_requirements:
            self._write(self.stdout, "No policy-generated requirements.")
        else:
            for requirement in proposal.approval_requirements:
                self._write(
                    self.stdout,
                    f"- action {requirement.action_index}: "
                    f"{requirement.action_type}: {requirement.reason}",
                )
        self._write(self.stdout, "Local execution still requires explicit approval.")

    def show_tool_approval(self, request: ToolApprovalRequest) -> None:
        preview = request.preview
        self.heading("Complete tool effect preview")
        self._write(self.stdout, f"Tool-call ID: {request.call.id}")
        self._write(self.stdout, f"Tool: {request.call.function_name}")
        self._write(self.stdout, f"Target: {request.target or '(none)'}")
        self._write(self.stdout, f"Preview ID: {preview.preview_id}")
        self._write(self.stdout, f"Preview digest: {preview.digest}")
        self._write(self.stdout, f"Match count: {preview.match_count}")
        self._write(self.stdout, f"Changed bytes: {preview.changed_bytes}")
        self._write(self.stdout, f"Changed lines: {preview.changed_lines}")
        self._write(self.stdout, f"Within configured limits: {preview.within_limits}")
        if request.preview_artifact:
            self._write(self.stdout, f"Complete artifact: {request.preview_artifact}")
        self._write(self.stdout, "--- BEGIN COMPLETE PREVIEW ---")
        self._write(self.stdout, preview.content)
        self._write(self.stdout, "--- END COMPLETE PREVIEW ---")

    def show_report(self, report: TaskReport) -> None:
        self.heading("Task report")
        self._write(self.stdout, json.dumps(report.as_dict(), indent=2, sort_keys=True))

    def show_diff(self, diff: str) -> None:
        self.heading("Git diff")
        self._write(self.stdout, diff if diff else "(clean)")

    def show_status(self, status: dict) -> None:
        self.heading("Status")
        self._write(self.stdout, json.dumps(status, indent=2, sort_keys=True))

    def heading(self, text: str) -> None:
        self._write(self.stdout, self._style(f"\n== {text} ==", "bold"))

    def info(self, text: str) -> None:
        self._write(self.stdout, text)

    def error(self, text: object) -> None:
        self._write(self.stderr, self._style(f"Error: {text}", "red"))

    def _write(self, stream: TextIO, text: str, *, end: str = "\n") -> None:
        stream.write(text + end)
        stream.flush()

    def _style(self, text: str, style: str) -> str:
        if not self.color:
            return text
        code = {"bold": "1", "cyan": "36", "red": "31"}[style]
        return f"\033[{code}m{text}\033[0m"
