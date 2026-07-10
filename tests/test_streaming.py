from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from a100_agent_lab import AgentLab, ListEventSink, StreamChunk
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions import Session, SessionStore


class FakeStreamingRuntime(Runtime):
    def __init__(self, response: str):
        self.response = response
        self.loaded = True

    def load(self) -> None:
        self.loaded = True

    def shutdown(self) -> None:
        self.loaded = False

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "fake", "ready": self.ready()}

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        return self.generate(session, prompt or "warmup", max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        completed = None
        for chunk in self.stream(session, prompt, **kwargs):
            if chunk.chunk_type == "completed":
                completed = chunk
        assert completed is not None and completed.metrics is not None
        return GenerationResult(text=completed.text, metrics=completed.metrics)

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        session.add_user_message(prompt)
        metrics = GenerationMetrics(
            prompt_tokens=1,
            generated_tokens=max(1, len(self.response.split())),
            ttft_sec=0.01,
            tokens_per_sec=100.0,
            wall_sec=0.02,
        )
        yield StreamChunk.started(metadata={"prompt_tokens": 1, "runtime": "fake"})
        mid = max(1, len(self.response) // 2)
        first = self.response[:mid]
        second = self.response[mid:]
        if first:
            yield StreamChunk.delta(first)
        if second:
            yield StreamChunk.delta(second)
        session.add_assistant_message(self.response)
        yield StreamChunk.completed(text=self.response, metrics=metrics, metadata={"runtime": "fake"})

    def tokenize(self, text_or_messages: Any) -> int:
        return 1


def make_lab(workspace_root: Path, *, response: str) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = workspace_root.parent
    lab.runtime = FakeStreamingRuntime(response)
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def plan_json() -> str:
    return json.dumps(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
            ],
            "metadata": {"source": "fake-model"},
        }
    )


def prepare_agent(tmp_path, *, response: str):
    sink = ListEventSink()
    lab = make_lab(tmp_path / "workspace", response=response)
    agent = lab.create_agent(event_sink=sink)
    agent.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    task = agent.create_task(title="Edit parser")
    return agent, task, sink


def consume_generator_with_result(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value


def test_agent_stream_emits_normalized_assistant_events(tmp_path) -> None:
    agent, task, sink = prepare_agent(tmp_path, response="visible answer")

    events = list(agent.stream("Explain C.", task=task))

    assert [event.event_type for event in events] == [
        "assistant.started",
        "assistant.delta",
        "assistant.delta",
        "assistant.completed",
    ]
    assert "".join(event.payload.get("delta", "") for event in events) == "visible answer"
    assert events[-1].payload["metrics"]["generated_tokens"] >= 1
    assert [event.event_type for event in sink.events][-4:] == [event.event_type for event in events]
    forbidden = {"chain_of_thought", "reasoning", "thoughts", "hidden_reasoning"}
    for event in events:
        assert forbidden.isdisjoint(event.as_dict())
        assert forbidden.isdisjoint(event.payload)


def test_streamed_proposal_reconstructs_valid_json(tmp_path) -> None:
    agent, task, _ = prepare_agent(tmp_path, response=plan_json())

    events, result = consume_generator_with_result(
        agent.propose_plan_stream(task, instruction="Replace return 0 with return 1 in parser.c")
    )

    assert result.ok
    assert result.proposal is not None
    assert events[-1].event_type == "plan.proposed"
    assert result.proposal.approval_requirements[0].action_type == "replace_text"


def test_invalid_streamed_json_fails_cleanly(tmp_path) -> None:
    agent, task, _ = prepare_agent(tmp_path, response="not json")

    events, result = consume_generator_with_result(agent.propose_plan_stream(task, instruction="Plan this."))

    assert result.status == "failed"
    assert result.proposal is None
    assert events[-1].event_type == "assistant.failed"
    assert "valid JSON" in result.error
