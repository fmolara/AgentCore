from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentEvent, AgentLab
from a100_agent_lab.tasks import TaskStatus

try:
    from rich.console import Console
    from rich.panel import Panel
except Exception:  # pragma: no cover - exercised when rich is not installed
    Console = None
    Panel = None


class TerminalEventSink:
    def __init__(self, renderer: "Renderer") -> None:
        self.renderer = renderer
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        self.renderer.event(event)


class Renderer:
    def __init__(self) -> None:
        self.console = Console() if Console is not None else None

    def info(self, text: str) -> None:
        if self.console is not None:
            self.console.print(text)
        else:
            print(text)

    def section(self, title: str, body: str) -> None:
        if self.console is not None and Panel is not None:
            self.console.print(Panel(body, title=title))
        else:
            print(f"\n== {title} ==\n{body}")

    def json(self, title: str, data: Any) -> None:
        self.section(title, json.dumps(data, indent=2, sort_keys=True))

    def event(self, event: AgentEvent) -> None:
        if event.event_type == "assistant.started":
            self.info("Assistant response:")
            return
        if event.event_type == "assistant.delta":
            delta = event.payload.get("delta", "")
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
            self.info(f"[assistant.failed] {event.payload.get('error', event.summary)}")
            return
        payload = event.payload
        extra = ""
        if event.event_type == "workspace.modified":
            extra = " " + ", ".join(payload.get("files_changed", []))
        elif event.event_type == "action.failed":
            action = payload.get("action", {})
            if isinstance(action, dict) and action.get("error"):
                extra = f" error={action['error']}"
        line = f"[{event.event_type}] {event.summary}{extra}"
        if self.console is not None:
            self.console.print(line)
        else:
            print(line)

    def plan(self, proposal) -> None:
        self.json("Proposed ActionPlan", proposal.action_plan.as_dict())

    def approvals(self, proposal) -> None:
        requirements = [item.as_dict() for item in proposal.approval_requirements]
        self.json("Approval Requirements", requirements)

    def report(self, task) -> None:
        self.json("Task Report", task.report().as_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive AgentCore local CLI.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument("--workspace", default=str(ROOT / "workspace" / "agentcore-cli"), help="Workspace root.")
    parser.add_argument(
        "--system-prompt",
        default="You are a concise coding assistant. Propose safe AgentCore ActionPlans only.",
        help="System prompt for the agent session.",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum tokens for planning responses.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip runtime warmup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    renderer = Renderer()
    sink = TerminalEventSink(renderer)

    renderer.info("AgentCore CLI")
    renderer.info("Git commits are never automatic. Shell access is not available.")

    lab = AgentLab.from_config(args.config)
    lab.start()
    try:
        if not args.no_warmup:
            renderer.info("Warming up runtime...")
            lab.warmup(max_tokens=8)

        agent = lab.create_agent(
            system_prompt=args.system_prompt,
            workspace_root=args.workspace,
            workspace_metadata={"purpose": "agentcore cli"},
            event_sink=sink,
        )
        if not agent.git.is_repo():
            agent.git.init()

        instruction = input("Task> ").strip()
        if not instruction:
            renderer.info("No task entered.")
            return 1

        task = agent.create_task(title=instruction[:80], description=instruction)
        planner_result = consume_proposal_stream(agent.propose_plan_stream(
            task,
            instruction=instruction,
            max_tokens=args.max_tokens,
            temperature=0,
        ))
        if planner_result.proposal is None:
            renderer.json("Planner Failed", planner_result.as_dict())
            return 1

        proposal = planner_result.proposal
        renderer.plan(proposal)
        renderer.approvals(proposal)
        renderer.info("Use /approve to execute, /reject to reject, /plan to inspect, /quit to exit.")

        executed = False
        while True:
            try:
                command = input("agentcore> ").strip()
            except EOFError:
                command = "/quit"
            if not command:
                continue
            if command == "/status":
                renderer.json("Status", {"task": task.as_dict(), "proposal": proposal.as_dict()})
            elif command == "/plan":
                renderer.plan(proposal)
                renderer.approvals(proposal)
            elif command == "/approve":
                if executed:
                    renderer.info("Proposal already executed.")
                    continue
                result = agent.execute_proposal(task, proposal, approved=True)
                renderer.json("Execution Result", result.as_dict())
                executed = result.status == "completed"
            elif command == "/reject":
                if executed:
                    renderer.info("Cannot reject an executed proposal.")
                    continue
                reason = input("Reject reason> ").strip() or "rejected by user"
                agent.reject_proposal(task, proposal, reason)
                renderer.info("Proposal rejected.")
            elif command == "/diff":
                renderer.section("Git Diff", agent.git.diff().stdout or "(empty)")
            elif command == "/checkpoint":
                label = input("Checkpoint label> ").strip() or "manual checkpoint"
                checkpoint = task.create_checkpoint(label)
                agent.emit_event(
                    "checkpoint.created",
                    f"Checkpoint created: {checkpoint.label}",
                    task=task,
                    payload={"checkpoint": checkpoint.as_dict()},
                )
                renderer.json("Checkpoint", checkpoint.as_dict())
            elif command == "/report":
                renderer.report(task)
            elif command == "/abort":
                if task.status in {TaskStatus.CREATED, TaskStatus.RUNNING}:
                    task.request_cancellation("aborted by user")
                    agent.emit_event(
                        "cancellation.requested",
                        f"Cancellation requested for task: {task.title}",
                        task=task,
                        payload={"reason": "aborted by user"},
                    )
                    if task.status == TaskStatus.CREATED:
                        task.cancel("aborted by user")
                        agent.emit_event(
                            "cancellation.completed",
                            f"Cancellation completed for task: {task.title}",
                            task=task,
                            payload={"reason": "aborted by user", "actions": 0},
                        )
                    renderer.info("Cancellation requested.")
                else:
                    renderer.info(f"Task is already {task.status.value}.")
            elif command == "/quit":
                break
            else:
                renderer.info(f"Unknown command: {command}")

        renderer.report(task)
        renderer.section("Git Diff", agent.git.diff().stdout or "(empty)")
        return 0
    finally:
        lab.shutdown()


def consume_proposal_stream(stream):
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            return stop.value


if __name__ == "__main__":
    raise SystemExit(main())
