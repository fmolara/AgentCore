from __future__ import annotations

from pathlib import Path

import pytest

from agentcore_server.api.client import AgentLab
from agentcore_server.generation.config import GenerationConfig
from agentcore_server.local.cli import build_parser
from agentcore_server.runtime.vllm import VLLMRuntime
from agentcore_server.tool_agent.protocols import (
    HarmonyToolProtocol,
    MistralToolProtocol,
    QwenToolProtocol,
    protocol_from_config,
)


def assistant_tool_message() -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "src/parser.c"},
            },
        }],
    }


def test_protocol_selection_is_explicit_and_qwen_compatible_by_default() -> None:
    assert isinstance(protocol_from_config(None), QwenToolProtocol)
    assert isinstance(protocol_from_config({"protocol": "qwen"}), QwenToolProtocol)
    assert isinstance(protocol_from_config({"protocol": "mistral"}), MistralToolProtocol)
    assert isinstance(protocol_from_config({"protocol": "harmony"}), HarmonyToolProtocol)
    with pytest.raises(ValueError, match="unsupported tool protocol"):
        protocol_from_config({"protocol": "unknown"})


def test_qwen_protocol_retains_existing_message_and_template_shape() -> None:
    message = assistant_tool_message()
    protocol = QwenToolProtocol()
    generation = GenerationConfig(enable_thinking=False)

    assert protocol.encode_messages([message]) == [message]
    assert protocol.template_options(generation) == {"enable_thinking": False}
    assert protocol.request_options(generation) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_mistral_protocol_uses_openai_string_tool_arguments() -> None:
    encoded = MistralToolProtocol().encode_messages([assistant_tool_message()])

    assert encoded[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path":"src/parser.c"}'
    )
    assert assistant_tool_message()["tool_calls"][0]["function"]["arguments"] == {
        "path": "src/parser.c"
    }


def test_harmony_protocol_maps_developer_role_and_hides_reasoning_delta() -> None:
    protocol = HarmonyToolProtocol(reasoning_effort="high")
    messages = [
        {"role": "system", "content": "safe coding agent"},
        assistant_tool_message(),
    ]
    generation = GenerationConfig()

    encoded = protocol.encode_messages(messages)

    assert encoded[0] == {"role": "developer", "content": "safe coding agent"}
    assert isinstance(
        encoded[1]["tool_calls"][0]["function"]["arguments"], str
    )
    assert protocol.request_options(generation) == {"reasoning_effort": "high"}
    delta = protocol.decode_delta({
        "reasoning": "private reasoning channel",
        "content": "visible",
        "tool_calls": [{"index": 0}],
    })
    assert delta.visible_text == "visible"
    assert delta.tool_calls == ({"index": 0},)


def vllm_config() -> dict:
    return {
        "runtime": "vllm",
        "model": {
            "path": "/models/example",
            "name": "example",
            "dtype": "bfloat16",
            "trust_remote_code": True,
        },
        "gpu": {"device": 2, "memory_fraction": 0.85},
        "context": {"max_context_tokens": 32768},
        "tool_agent": {"protocol": "mistral"},
        "server": {
            "python": "/venv/bin/python",
            "host": "127.0.0.1",
            "port": 31234,
            "tool_call_parser": "mistral",
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
            "enforce_eager": True,
        },
    }


def test_vllm_runtime_has_bounded_trusted_launch_shape(tmp_path: Path) -> None:
    runtime = VLLMRuntime(vllm_config(), project_root=tmp_path)

    command = runtime._launch_command()

    assert command[:3] == [
        "/venv/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert command[command.index("--model") + 1] == "/models/example"
    assert command[command.index("--tool-call-parser") + 1] == "mistral"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert "--enable-auto-tool-choice" in command
    assert "--enforce-eager" in command
    assert runtime._server_env_updates() == {
        "CUDA_VISIBLE_DEVICES": "2",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def test_agent_lab_constructs_shared_vllm_tool_runtime(tmp_path: Path) -> None:
    lab = AgentLab(vllm_config(), project_root=tmp_path)

    assert isinstance(lab.runtime, VLLMRuntime)
    assert lab.runtime.tool_protocol.name == "mistral"


def test_harmony_reasoning_effort_is_strict() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        HarmonyToolProtocol(reasoning_effort="extreme")


def test_tool_loop_cli_name_and_qwen_alias_are_both_available() -> None:
    parser = build_parser()
    common = ["--config", "config.yaml", "--workspace", "."]

    assert parser.parse_args([*common, "--agent", "tool-loop"]).agent == "tool-loop"
    assert parser.parse_args([*common, "--agent", "qwen-tools"]).agent == "qwen-tools"
