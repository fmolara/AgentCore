from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from agentcore_protocol import AgentCoreClient, AgentCoreError, AgentEvent

from agentclient.commands import HELP_TEXT, is_known_command, parse_command
from agentclient.config import ClientConfig, load_client_config
from agentclient.rendering import Renderer


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

    def connect(self) -> None:
        health = self.client.check_compatibility()
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
        proposal_events = list(
            self.client.stream_proposal(
                self.state.task_id,
                instruction=instruction,
                max_tokens=768,
                temperature=0,
            )
        )
        self.state.events.extend(proposal_events)
        for event in proposal_events:
            self.renderer.event(event)
        proposal = _proposal_from_events(proposal_events)
        if proposal is None:
            self.renderer.error("Planner did not return a proposal.")
            return
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
        event_thread = threading.Thread(target=self._collect_task_events, daemon=True)
        event_thread.start()
        result = self.client.execute_proposal(self.state.proposal_id)
        event_thread.join(timeout=30)
        self.state.executed = result.status == "completed"
        self.renderer.json("Execution Result", result.raw)

    def reject(self, reason: str) -> None:
        if self.state.proposal_id is None:
            self.renderer.error("No proposal to reject.")
            return
        if self.state.executed:
            self.renderer.error("Cannot reject an executed proposal.")
            return
        proposal = self.client.reject_proposal(self.state.proposal_id, reason)
        self.state.rejected = True
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
        self.renderer.json("Cancellation Requested", task.raw)

    def run(self, inputs: Iterable[str] | None = None) -> int:
        self.connect()
        self.create_agent()
        self.renderer.section("Help", HELP_TEXT)
        if inputs is not None:
            for line in inputs:
                if not self.handle_line(line):
                    break
            return 0
        while True:
            try:
                line = input("agentclient> ")
            except EOFError:
                break
            try:
                if not self.handle_line(line):
                    break
            except AgentCoreError as exc:
                self.renderer.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - interactive clients should not crash on bad commands.
                self.renderer.error(str(exc))
                if self.renderer.debug:
                    raise
        return 0

    def _collect_task_events(self) -> None:
        if self.state.task_id is None:
            return
        for event in self.client.stream_task_events(self.state.task_id):
            self.state.events.append(event)
            self.renderer.event(event)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote AgentCore CLI.")
    parser.add_argument("--server", default=None, help="AgentCore server URL.")
    parser.add_argument("--workspace", default=None, help="Server-side workspace path.")
    parser.add_argument("--system-prompt", default=None, help="System prompt for the remote agent.")
    parser.add_argument("--config", default=None, help="Client config TOML path.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--debug", action="store_true", help="Raise unexpected command errors.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_client_config(args.config).with_overrides(
        server_url=args.server,
        default_workspace=args.workspace,
        system_prompt=args.system_prompt,
        color=False if args.no_color else None,
    )
    renderer = Renderer(color=config.color, debug=args.debug)
    with AgentCoreClient(config.server_url, timeout=config.request_timeout_sec) as client:
        return RemoteAgentCLI(client, config=config, renderer=renderer).run()


def _proposal_from_events(events: Iterable[AgentEvent]) -> dict[str, Any] | None:
    for event in events:
        if event.event_type == "plan.proposed":
            proposal = event.payload.get("proposal")
            if isinstance(proposal, dict):
                return proposal
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
