from __future__ import annotations

from typing import Any, Iterator

from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.generation.stream import StreamChunk
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.sessions.session import Session


class FakeRuntime(Runtime):
    def __init__(self) -> None:
        self.loaded = False
        self.shutdown_called = False
        self.generated = 0
        self.warmups = 0

    def load(self) -> None:
        self.loaded = True
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.loaded = False
        self.shutdown_called = True

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {
            "runtime_name": "fake",
            "backend_type": "in_process",
            "model_path": "fake-model",
            "ready": self.ready(),
            "server_ready_time_sec": None,
            "warmup_wall_sec": 0.01 if self.warmups else None,
            "gpu_name": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "process_pid": None,
            "endpoint": None,
            "last_error": None,
        }

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        self.warmups += 1
        session = self.create_session()
        return self.generate(session, prompt or "warmup", max_tokens=max_tokens)

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        if not self.ready():
            raise RuntimeError("runtime is not loaded")
        self.generated += 1
        session.add_user_message(prompt)
        text = f"fake response {self.generated}"
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

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        self.generated += 1
        session.add_user_message(prompt)
        text = f"fake response {self.generated}"
        metrics = GenerationMetrics(
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(text.split()),
            ttft_sec=0.01,
            tokens_per_sec=100.0,
            wall_sec=0.02,
        )
        yield StreamChunk.started(metadata={"prompt_tokens": metrics.prompt_tokens})
        yield StreamChunk.delta(text)
        session.add_assistant_message(text)
        yield StreamChunk.completed(text=text, metrics=metrics)

    def tokenize(self, text_or_messages: Any) -> int:
        if isinstance(text_or_messages, list):
            return sum(len(message["content"].split()) for message in text_or_messages)
        return len(str(text_or_messages).split())

    def statistics(self) -> dict[str, Any]:
        stats = self.health()
        stats["generated_requests"] = self.generated
        return stats


def test_runtime_contract_lifecycle_and_generation() -> None:
    runtime = FakeRuntime()

    assert not runtime.ready()
    runtime.load()
    assert runtime.ready()

    health = runtime.health()
    assert health["runtime_name"] == "fake"
    assert health["backend_type"] == "in_process"
    assert health["ready"] is True
    assert health["last_error"] is None

    session = runtime.create_session(system_prompt="Be concise.")
    result = runtime.generate(session, "Explain malloc.", max_tokens=8)

    assert result.text == "fake response 1"
    assert result.metrics.generated_tokens == 3
    assert session.transcript() == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain malloc."},
        {"role": "assistant", "content": "fake response 1"},
    ]

    streamed = list(runtime.stream(session, "Explain free.", max_tokens=8))
    assert [chunk.chunk_type for chunk in streamed] == ["started", "delta", "completed"]
    assert streamed[1].text_delta == "fake response 2"
    assert streamed[-1].text == "fake response 2"

    assert runtime.tokenize("one two three") == 3
    assert runtime.statistics()["generated_requests"] == 2

    warmup = runtime.warmup(max_tokens=4)
    assert warmup.text == "fake response 3"
    assert runtime.health()["warmup_wall_sec"] == 0.01

    runtime.shutdown()
    assert not runtime.ready()
    assert runtime.shutdown_called is True
