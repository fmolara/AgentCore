from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from agentcore_server import Agent, AgentLab
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self) -> None:
        self.loaded = False
        self.last_kwargs: dict[str, Any] = {}

    def load(self) -> None:
        self.loaded = True

    def shutdown(self) -> None:
        self.loaded = False

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "fake", "ready": self.ready()}

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        return self.generate(session, prompt or "warmup", max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        self.last_kwargs = dict(kwargs)
        session.add_user_message(prompt)
        text = f"answer: {prompt}"
        session.add_assistant_message(text)
        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(text.split()),
                ttft_sec=0.05,
                tokens_per_sec=42.0,
                wall_sec=0.2,
            ),
        )

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self.generate(session, prompt, **kwargs).text

    def tokenize(self, text_or_messages: Any) -> int:
        return len(str(text_or_messages).split())

    def statistics(self) -> dict[str, Any]:
        return self.health()


def make_lab(workspace_root: Path) -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(workspace_root)}}
    lab.project_root = Path(".")
    lab.runtime = FakeRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def test_agent_ask_tracks_session_metrics_and_workspace(tmp_path) -> None:
    lab = make_lab(tmp_path / "workspace")
    lab.start()

    agent = lab.create_agent(
        system_prompt="You are concise.",
        max_tokens=32,
        temperature=0,
    )

    assert isinstance(agent, Agent)

    result = agent.ask("Explain pointer arithmetic.", max_tokens=16)
    stats = agent.statistics()

    assert result.text == "answer: Explain pointer arithmetic."
    assert lab.runtime.last_kwargs == {"max_tokens": 16, "temperature": 0}
    assert agent.session.turn_count == 1
    assert stats["conversation_turns"] == 1
    assert stats["prompt_tokens"] == 3
    assert stats["generated_tokens"] == 4
    assert stats["last_ttft_sec"] == 0.05
    assert stats["last_tokens_per_sec"] == 42.0
    assert stats["generation_options"] == {"max_tokens": 32, "temperature": 0}
    assert stats["workspace"]["root"] == str(tmp_path / "workspace")
    agent.workspace.write_text("notes.txt", "workspace state\n")
    assert agent.workspace.read_text("notes.txt") == "workspace state\n"


def test_agent_reset_clears_conversation_and_metrics(tmp_path) -> None:
    lab = make_lab(tmp_path / "workspace")
    lab.start()
    agent = lab.create_agent(system_prompt="Keep context.")

    agent.ask("First question.")
    assert agent.statistics()["conversation_turns"] == 1

    agent.reset()

    assert agent.session.turn_count == 0
    assert agent.session.transcript() == [{"role": "system", "content": "Keep context."}]
    assert agent.statistics()["prompt_tokens"] == 0
    assert agent.statistics()["generated_tokens"] == 0
    assert agent.statistics()["last_ttft_sec"] is None


def test_two_agents_keep_independent_sessions(tmp_path) -> None:
    lab = make_lab(tmp_path / "workspace")
    lab.start()

    first = lab.create_agent(system_prompt="Agent one.")
    second = lab.create_agent(system_prompt="Agent two.")

    first.ask("Use C.")
    second.ask("Use Python.")

    assert first.session.id != second.session.id
    assert first.session.turn_count == 1
    assert second.session.turn_count == 1
    assert "Use C." in "\n".join(message["content"] for message in first.session.transcript())
    assert "Use Python." not in "\n".join(message["content"] for message in first.session.transcript())
    assert "Use Python." in "\n".join(message["content"] for message in second.session.transcript())
