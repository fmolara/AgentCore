from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcore_server.generation import ToolCallDelta
from agentcore_server.runtime.sglang import SGLangRuntime, ToolTurnContextCapacityError
from agentcore_server.sessions import Session


class FakeTokenizer:
    def __init__(self):
        self.rendered_messages = None
        self.rendered_tools = None

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages = messages
        self.rendered_tools = kwargs.get("tools")
        return json.dumps({"messages": messages, "tools": kwargs.get("tools")}, sort_keys=True)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class FakeServer:
    def __init__(self, events):
        self.events = events
        self.payload = None

    def ready(self):
        return True

    def stream_chat_events(self, payload):
        self.payload = payload
        yield from self.events


class FixedTokenTokenizer(FakeTokenizer):
    def __init__(self, prompt_tokens: int):
        super().__init__()
        self.prompt_tokens = prompt_tokens

    def apply_chat_template(self, messages, **kwargs):
        super().apply_chat_template(messages, **kwargs)
        return "exact rendered native request"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "exact rendered native request":
            return list(range(self.prompt_tokens))
        return text.split()


def chunk(delta=None, finish=None, usage=None):
    return {
        "choices": [] if delta is None and finish is None else [{
            "delta": delta or {},
            "finish_reason": finish,
        }],
        "usage": usage,
    }


def runtime_with(events):
    runtime = SGLangRuntime.__new__(SGLangRuntime)
    runtime.config = {
        "model": {"path": "qwen"},
        "generation": {"temperature": 0, "max_tokens": 128, "enable_thinking": False},
    }
    runtime.tokenizer = FakeTokenizer()
    runtime.server = FakeServer(events)
    runtime.log_writer = None
    return runtime


def test_native_tool_deltas_assemble_ids_indices_names_and_arguments() -> None:
    events = [
        chunk({"content": "Inspecting. ", "tool_calls": None}),
        chunk({"tool_calls": [{
            "id": "call_a", "index": 1, "function": {"name": "read_file", "arguments": ""}
        }]}),
        chunk({"tool_calls": [{
            "id": "call_b", "index": 0, "function": {"name": "git_status", "arguments": "{}"}
        }]}),
        chunk({"tool_calls": [{
            "id": None, "index": 1, "function": {"name": None, "arguments": "{\"path\":"}
        }]}),
        chunk({"tool_calls": [{
            "id": None, "index": 1, "function": {"name": None, "arguments": "\"src/parser.c\"}"}
        }]}),
        chunk({}, finish="tool_calls"),
        chunk(usage={"prompt_tokens": 42, "completion_tokens": 17}),
    ]
    runtime = runtime_with(events)
    session = Session(system_prompt="system")
    session.add_user_message("inspect")
    schema = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    streamed = list(runtime.stream_tool_turn(session, schema))

    deltas = [item.tool_call_delta for item in streamed if item.tool_call_delta]
    assert all(isinstance(item, ToolCallDelta) for item in deltas)
    completed = streamed[-1].turn
    assert completed is not None
    assert completed.text == "Inspecting. "
    assert [item.index for item in completed.tool_calls] == [0, 1]
    assert completed.tool_calls[0].id == "call_b"
    assert completed.tool_calls[1].id == "call_a"
    assert completed.tool_calls[1].arguments == {"path": "src/parser.c"}
    assert completed.finish_reason == "tool_calls"
    assert completed.metrics.prompt_tokens == 42
    assert completed.metrics.generated_tokens == 17
    assert runtime.server.payload["tools"] == schema
    assert runtime.server.payload["parallel_tool_calls"] is True


def test_role_tool_result_is_sent_with_matching_id_on_next_turn() -> None:
    first = runtime_with([
        chunk({"tool_calls": [{
            "id": "call_1", "index": 0,
            "function": {"name": "read_file", "arguments": '{"path":"a.c"}'},
        }]}),
        chunk({}, finish="tool_calls"),
    ])
    session = Session(system_prompt="system")
    session.add_user_message("inspect")
    completed = list(first.stream_tool_turn(session, []))[-1].turn
    assert completed is not None
    from agentcore_server.generation import ToolResult

    session.add_tool_result(ToolResult("call_1", "read_file", True, '{"content":"x"}'))
    second = runtime_with([chunk({"content": "done"}), chunk({}, finish="stop")])

    result = list(second.stream_tool_turn(session, []))[-1].turn

    assert result is not None and result.text == "done"
    assert second.server.payload["messages"][-1] == {
        "role": "tool",
        "content": '{"content":"x"}',
        "tool_call_id": "call_1",
    }


def test_tool_turn_clamps_output_to_remaining_context() -> None:
    runtime = runtime_with([chunk({"content": "done"}), chunk({}, finish="stop")])
    runtime.config["context"] = {"max_context_tokens": 16384}
    runtime.config["generation"]["max_tokens"] = 2048
    runtime.tokenizer = FixedTokenTokenizer(14552)
    session = Session(system_prompt="system")
    session.add_user_message("task")

    streamed = list(runtime.stream_tool_turn(
        session,
        [{"type": "function", "function": {"name": "read_file"}}],
        context_safety_margin_tokens=128,
        minimum_output_tokens=256,
    ))

    assert streamed[0].metadata == {
        "runtime": "sglang",
        "context_limit": 16384,
        "exact_prompt_tokens": 14552,
        "prompt_tokens": 14552,
        "configured_max_tokens": 2048,
        "safety_margin_tokens": 128,
        "available_tokens": 1704,
        "effective_max_tokens": 1704,
        "minimum_output_tokens": 256,
        "sufficient": True,
    }
    assert runtime.server.payload["max_tokens"] == 1704


def test_tool_turn_keeps_configured_output_when_request_fits() -> None:
    runtime = runtime_with([chunk({"content": "done"}), chunk({}, finish="stop")])
    runtime.config["context"] = {"max_context_tokens": 16384}
    runtime.config["generation"]["max_tokens"] = 2048
    runtime.tokenizer = FixedTokenTokenizer(1000)
    session = Session(system_prompt="system")
    session.add_user_message("task")

    list(runtime.stream_tool_turn(session, []))

    assert runtime.server.payload["max_tokens"] == 2048


def test_tool_turn_below_minimum_is_not_sent() -> None:
    runtime = runtime_with([])
    runtime.config["context"] = {"max_context_tokens": 4096}
    runtime.config["generation"]["max_tokens"] = 2048
    runtime.tokenizer = FixedTokenTokenizer(3800)
    session = Session(system_prompt="system")
    session.add_user_message("task")
    streamed = runtime.stream_tool_turn(
        session,
        [],
        context_safety_margin_tokens=128,
        minimum_output_tokens=256,
    )

    started = next(streamed)
    assert started.metadata["available_tokens"] == 168
    assert started.metadata["sufficient"] is False
    with pytest.raises(ToolTurnContextCapacityError) as caught:
        next(streamed)

    assert caught.value.diagnostics == started.metadata
    assert runtime.server.payload is None


def test_exact_prompt_count_uses_transcript_and_native_tool_schemas() -> None:
    runtime = runtime_with([chunk({"content": "done"}), chunk({}, finish="stop")])
    session = Session(system_prompt="system")
    session.add_user_message("inspect")
    schema = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    streamed = list(runtime.stream_tool_turn(session, schema))

    assert runtime.tokenizer.rendered_messages == runtime.server.payload["messages"]
    assert runtime.tokenizer.rendered_tools == runtime.server.payload["tools"]
    rendered = json.dumps(
        {"messages": runtime.server.payload["messages"], "tools": schema},
        sort_keys=True,
    )
    assert streamed[0].metadata["exact_prompt_tokens"] == len(rendered.split())


def test_sglang_launch_command_includes_qwen_tool_parser(tmp_path: Path) -> None:
    runtime = SGLangRuntime(
        {
            "model": {"path": "qwen", "dtype": "bfloat16"},
            "server": {"tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"},
        },
        project_root=tmp_path,
    )

    command = runtime._launch_command()

    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
