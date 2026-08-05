from __future__ import annotations

import json

from agentcore_server.generation.config import GenerationConfig
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.logging.events import generation_event
from agentcore_server.logging.writer import JsonlWriter
from agentcore_server.sessions.session import Message, Session


def test_message_as_dict() -> None:
    message = Message(role="user", content="hello")

    assert message.as_dict() == {"role": "user", "content": "hello"}


def test_session_tracks_transcript_and_reset() -> None:
    session = Session(system_prompt="Be concise.")

    assert session.transcript() == [{"role": "system", "content": "Be concise."}]
    original_updated_at = session.updated_at

    session.add_user_message("Explain C pointers.")
    session.add_assistant_message("Pointers store addresses.")

    assert session.turn_count == 1
    assert session.transcript() == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain C pointers."},
        {"role": "assistant", "content": "Pointers store addresses."},
    ]
    assert session.updated_at >= original_updated_at

    session.reset()

    assert session.transcript() == [{"role": "system", "content": "Be concise."}]
    assert session.turn_count == 0


def test_generation_config_from_dict_and_override() -> None:
    config = GenerationConfig.from_dict(
        {
            "temperature": "0.2",
            "max_tokens": "32",
            "top_p": "0.9",
            "top_k": "20",
            "repetition_penalty": "1.05",
            "enable_thinking": True,
        }
    )

    assert config == GenerationConfig(
        temperature=0.2,
        max_tokens=32,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.05,
        enable_thinking=True,
    )

    overridden = config.override(max_tokens=8, temperature=None, unknown_field="ignored")

    assert overridden.max_tokens == 8
    assert overridden.temperature == 0.2
    assert not hasattr(overridden, "unknown_field")


def test_generation_metrics_as_dict() -> None:
    metrics = GenerationMetrics(
        prompt_tokens=10,
        generated_tokens=4,
        ttft_sec=0.1,
        tokens_per_sec=20.0,
        wall_sec=0.3,
    )

    assert metrics.as_dict() == {
        "prompt_tokens": 10,
        "generated_tokens": 4,
        "ttft_sec": 0.1,
        "tokens_per_sec": 20.0,
        "wall_sec": 0.3,
    }


def test_generation_event_and_jsonl_writer(tmp_path) -> None:
    session = Session(system_prompt="Be concise.")
    session.add_user_message("Ping")
    session.add_assistant_message("Pong")
    result = GenerationResult(
        text="Pong",
        metrics=GenerationMetrics(
            prompt_tokens=2,
            generated_tokens=1,
            ttft_sec=0.01,
            tokens_per_sec=100.0,
            wall_sec=0.02,
        ),
    )
    health = {
        "runtime_name": "fake",
        "backend_type": "in_process",
        "ready": True,
    }

    event = generation_event("fake", session, result, health, event_type="generation")

    assert event["runtime"] == "fake"
    assert event["event_type"] == "generation"
    assert event["session_id"] == session.id
    assert event["turn"] == 1
    assert event["metrics"]["generated_tokens"] == 1
    assert event["health"] == health
    assert event["status"] == "ok"

    path = tmp_path / "logs" / "events.jsonl"
    writer = JsonlWriter(path)
    writer.write(event)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    decoded = json.loads(rows[0])
    assert decoded["runtime"] == "fake"
    assert decoded["metrics"]["tokens_per_sec"] == 100.0
