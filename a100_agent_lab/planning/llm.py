from __future__ import annotations

import json
from typing import Any

from a100_agent_lab.events import AgentEvent
from a100_agent_lab.executor import ActionPlan, ApprovalPolicy, PlanProposal
from a100_agent_lab.generation.result import GenerationMetrics
from a100_agent_lab.planning.planner import PlannerResult
from a100_agent_lab.tasks import Task

PLANNER_SYSTEM_PROMPT = """You are AgentCore's plan proposal generator.

Return exactly one JSON object representing an ActionPlan.
Do not return Markdown, prose, comments, or code fences.

Allowed action types and schemas:
- read_file: {"type":"read_file","path":"relative/path"}
- write_file: {"type":"write_file","path":"relative/path","content":"text"}
- replace_text: {"type":"replace_text","path":"relative/path","old":"text","new":"text","count":-1}
- create_checkpoint: {"type":"create_checkpoint","label":"short label","description":"optional text"}
- git_status: {"type":"git_status"}
- git_diff: {"type":"git_diff"}
- task_report: {"type":"task_report"}

Forbidden:
- shell commands
- arbitrary git commands
- network operations
- absolute paths
- files outside the workspace
- action types outside the known schema
- automatic execution
- automatic commits

The JSON object must have this shape:
{
  "title": "short plan title",
  "description": "short plan description",
  "actions": [
    {"type": "..."}
  ],
  "metadata": {
    "planner": "simple_llm"
  }
}
"""


class SimpleLLMPlanner:
    def __init__(self, *, max_tokens: int = 1024, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def propose(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None = None,
        **generation_options: Any,
    ) -> PlannerResult:
        prompt = self._prompt(agent, task, instruction)
        options = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        options.update(generation_options)

        result = agent.ask(prompt, **options)
        raw_text = result.text.strip()
        try:
            data = _parse_json_object(raw_text)
            plan = ActionPlan.from_dict(data)
            proposal = PlanProposal.from_action_plan(
                task_id=task.id,
                action_plan=plan,
                summary=f"LLM-generated proposal for: {instruction}",
                approval_policy=approval_policy,
                metadata={
                    "planner": "simple_llm",
                    "instruction": instruction,
                },
            )
        except Exception as exc:
            return PlannerResult.failed(error=str(exc), raw_text=raw_text, metrics=result.metrics)

        return PlannerResult.proposed(proposal=proposal, raw_text=raw_text, metrics=result.metrics)

    def propose_stream(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None = None,
        **generation_options: Any,
    ):
        prompt = self._prompt(agent, task, instruction)
        options = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        options.update(generation_options)

        chunks: list[str] = []
        completed_text = ""
        metrics: GenerationMetrics | None = None
        for event in agent.stream(prompt, task=task, **options):
            if event.event_type == "assistant.delta":
                delta = event.payload.get("delta")
                if isinstance(delta, str):
                    chunks.append(delta)
            elif event.event_type == "assistant.completed":
                text = event.payload.get("text")
                if isinstance(text, str):
                    completed_text = text
                metrics = _metrics_from_payload(event.payload.get("metrics"))
            yield event

        raw_text = (completed_text or "".join(chunks)).strip()
        try:
            data = _parse_json_object(raw_text)
            plan = ActionPlan.from_dict(data)
            proposal = PlanProposal.from_action_plan(
                task_id=task.id,
                action_plan=plan,
                summary=f"LLM-generated proposal for: {instruction}",
                approval_policy=approval_policy,
                metadata={
                    "planner": "simple_llm",
                    "instruction": instruction,
                },
            )
        except Exception as exc:
            failed = AgentEvent(
                event_type="assistant.failed",
                summary="Planner output validation failed",
                task_id=task.id,
                session_id=agent.session.id,
                payload={"error": str(exc)},
            )
            agent._emit_existing_event(failed)
            yield failed
            return PlannerResult.failed(error=str(exc), raw_text=raw_text, metrics=metrics)

        result = PlannerResult.proposed(proposal=proposal, raw_text=raw_text, metrics=metrics)
        proposed = AgentEvent(
            event_type="plan.proposed",
            summary=f"Plan proposed: {proposal.title}",
            task_id=task.id,
            session_id=agent.session.id,
            payload={
                "proposal": proposal.as_dict(),
                "approval_requirements": [requirement.as_dict() for requirement in proposal.approval_requirements],
            },
        )
        agent._emit_existing_event(proposed)
        yield proposed
        return result

    def _prompt(self, agent, task: Task, instruction: str) -> str:
        workspace_listing = ", ".join(agent.workspace.list(".")) if agent.workspace.exists(".") else ""
        git_status = agent.git.status().stdout if agent.git.is_repo() else ""
        return (
            PLANNER_SYSTEM_PROMPT
            + "\nTask:\n"
            + f"- id: {task.id}\n"
            + f"- title: {task.title}\n"
            + f"- description: {task.description}\n"
            + "\nWorkspace:\n"
            + f"- files: {workspace_listing}\n"
            + "\nGit status:\n"
            + (git_status or "(not a git repository or clean status)")
            + "\n\nUser instruction:\n"
            + instruction
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("planner output must be a JSON object")
    return data


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("planner output is not valid JSON")
    return text[start : end + 1]


def _metrics_from_payload(data: Any) -> GenerationMetrics | None:
    if not isinstance(data, dict):
        return None
    try:
        return GenerationMetrics(
            prompt_tokens=int(data["prompt_tokens"]),
            generated_tokens=int(data["generated_tokens"]),
            ttft_sec=data.get("ttft_sec"),
            tokens_per_sec=float(data["tokens_per_sec"]),
            wall_sec=float(data["wall_sec"]),
        )
    except Exception:
        return None
