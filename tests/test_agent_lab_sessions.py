from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from agentcore_server.api.client import AgentLab
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore


class FakeRuntime(Runtime):
    def __init__(self) -> None:
        self.loaded = False

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
        session.add_user_message(prompt)
        text = f"reply to {prompt}"
        session.add_assistant_message(text)
        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(text.split()),
                ttft_sec=0.01,
                tokens_per_sec=100.0,
                wall_sec=0.02,
            ),
        )

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self.generate(session, prompt, **kwargs).text

    def tokenize(self, text_or_messages: Any) -> int:
        return len(str(text_or_messages).split())

    def statistics(self) -> dict[str, Any]:
        return self.health()


def make_lab() -> AgentLab:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake"}
    lab.project_root = Path(".")
    lab.runtime = FakeRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab


def test_agent_lab_manages_multiple_independent_sessions() -> None:
    lab = make_lab()
    lab.start()

    first = lab.create_session(system_prompt="First session.")
    second = lab.create_session(system_prompt="Second session.")

    assert first.id != second.id
    assert lab.get_session(first.id) is first
    assert lab.get_session(second.id) is second
    assert [session.id for session in lab.list_sessions()] == [first.id, second.id]

    first_result = lab.generate(first, "Explain malloc.")
    second_result = lab.generate(second, "Explain fork.")

    assert first_result.text == "reply to Explain malloc."
    assert second_result.text == "reply to Explain fork."
    assert first.turn_count == 1
    assert second.turn_count == 1
    assert first.transcript()[0] == {"role": "system", "content": "First session."}
    assert second.transcript()[0] == {"role": "system", "content": "Second session."}

    lab.reset_session(first.id)

    assert first.turn_count == 0
    assert second.turn_count == 1
    assert lab.delete_session(first.id) is True
    assert [session.id for session in lab.list_sessions()] == [second.id]

    lab.shutdown()
    assert not lab.ready()
