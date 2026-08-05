from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from agentcore_server.generation.config import GenerationConfig
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.generation.stream import StreamChunk
from agentcore_server.generation.tools import (
    AssistantTurn,
    ToolCall,
    ToolCallDelta,
    ToolTurnChunk,
)
from agentcore_server.logging.events import generation_event
from agentcore_server.logging.writer import JsonlWriter
from agentcore_server.runtime.base import Runtime
from agentcore_server.runtime.health import normalized_health, query_gpu
from agentcore_server.runtime.server_process import RuntimeStreamError, ServerProcess
from agentcore_server.sessions.session import Session


class ToolTurnContextCapacityError(RuntimeError):
    """Raised before a native tool request that cannot reserve safe output."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Qwen tool turn has insufficient context capacity: "
            f"available={diagnostics['available_tokens']}, "
            f"effective={diagnostics['effective_max_tokens']}, "
            f"minimum={diagnostics['minimum_output_tokens']}"
        )


class SGLangIncompleteStreamError(RuntimeError):
    """Raised when a native tool stream has no usable terminal data."""


class SGLangRuntime(Runtime):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: Path,
        log_writer: JsonlWriter | None = None,
    ):
        self.config = config
        self.project_root = project_root
        self.log_writer = log_writer
        self.tokenizer = None
        self.api_base = self._api_base()
        self.last_warmup_wall_sec: float | None = None
        self.server = ServerProcess(
            runtime_name="sglang",
            api_base=self.api_base,
            project_root=project_root,
            log_dir=self._log_dir(),
            log_prefix="sglang-server",
        )

    def load(self) -> None:
        if self.tokenizer is None:
            model_cfg = self.config["model"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_cfg["path"],
                trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            )

        if self.ready():
            return

        server_cfg = self.config.get("server", {})
        self.server.start(
            self._launch_command(),
            timeout=float(server_cfg.get("startup_timeout_sec", 180)),
            env_updates={"CUDA_VISIBLE_DEVICES": str(self.config.get("gpu", {}).get("device", 0))},
            path_prefix=server_cfg.get("path_prefix"),
        )

    def shutdown(self) -> None:
        self.server.shutdown()

    def ready(self) -> bool:
        return self.server.ready()

    def health(self) -> dict[str, Any]:
        server_health = self.server.health()
        return normalized_health(
            runtime_name="sglang",
            backend_type="server",
            model_path=self.config.get("model", {}).get("path"),
            ready=server_health["ready"],
            server_ready_time_sec=server_health["ready_sec"],
            warmup_wall_sec=self.last_warmup_wall_sec,
            process_pid=server_health["process_pid"],
            endpoint=server_health["api_base"],
            last_error=server_health["last_error"],
            gpu=query_gpu(self.config.get("gpu", {}).get("device", 0)),
            extra={"server_log_path": server_health["server_log_path"]},
        )

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        session = self.create_session()
        result = self._collect_stream(
            session,
            prompt or "Say ready.",
            event_type="warmup",
            max_tokens=max_tokens,
        )
        self.last_warmup_wall_sec = result.metrics.wall_sec
        return result

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        return self._collect_stream(session, prompt, event_type="generation", **kwargs)

    def _collect_stream(
        self,
        session: Session,
        prompt: str,
        *,
        event_type: str,
        **kwargs: Any,
    ) -> GenerationResult:
        completed: GenerationResult | None = None
        for chunk in self._stream(session, prompt, event_type=event_type, **kwargs):
            if chunk.chunk_type == "failed":
                raise RuntimeError(chunk.error or "generation failed")
            if chunk.chunk_type == "completed" and chunk.metrics is not None:
                completed = GenerationResult(text=chunk.text, metrics=chunk.metrics)
        if completed is None:
            raise RuntimeError("generation did not complete")
        return completed

    def _stream(
        self,
        session: Session,
        prompt: str,
        *,
        event_type: str,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        if not self.ready():
            raise RuntimeError("SGLang server is not ready")

        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        session.add_user_message(prompt)
        messages = session.transcript()
        prompt_tokens = self.tokenize(messages, generation=generation)
        yield StreamChunk.started(metadata={"prompt_tokens": prompt_tokens, "runtime": "sglang"})

        payload = {
            "model": self.config.get("model", {}).get("name", self.config["model"]["path"]),
            "messages": messages,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": generation.enable_thinking},
        }
        self._add_optional_sampling(payload, generation)

        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        for chunk in self.server.stream_chat(payload):
            if chunk and first_token_at is None:
                first_token_at = time.perf_counter()
            if chunk:
                yield StreamChunk.delta(chunk)
            chunks.append(chunk)
        end = time.perf_counter()

        text = "".join(chunks)
        generated_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        ttft = None if first_token_at is None else first_token_at - start
        decode_sec = None if first_token_at is None else max(end - first_token_at, 1e-9)
        tokens_per_sec = 0.0 if decode_sec is None else generated_tokens / decode_sec
        session.add_assistant_message(text)

        result = GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                ttft_sec=ttft,
                tokens_per_sec=tokens_per_sec,
                wall_sec=end - start,
            ),
        )

        if self.log_writer is not None:
            self.log_writer.write(generation_event("sglang", session, result, self.health(), event_type=event_type))

        yield StreamChunk.completed(text=text, metrics=result.metrics, metadata={"runtime": "sglang"})

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        yield from self._stream(session, prompt, event_type="generation", **kwargs)

    def stream_tool_turn(
        self,
        session: Session,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ToolTurnChunk]:
        if not self.ready():
            raise RuntimeError("SGLang server is not ready")
        safety_margin_tokens = int(kwargs.pop("context_safety_margin_tokens", 128))
        minimum_output_tokens = int(kwargs.pop("minimum_output_tokens", 256))
        if safety_margin_tokens < 0:
            raise ValueError("context_safety_margin_tokens must not be negative")
        if minimum_output_tokens <= 0:
            raise ValueError("minimum_output_tokens must be positive")
        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        messages = session.transcript()
        prompt_tokens = self.tokenize(messages, generation=generation, tools=tools)
        context_limit = int(self.config.get("context", {}).get("max_context_tokens", 4096))
        configured_max_tokens = generation.max_tokens
        available_tokens = context_limit - prompt_tokens - safety_margin_tokens
        effective_max_tokens = max(0, min(configured_max_tokens, available_tokens))
        capacity = {
            "runtime": "sglang",
            "context_limit": context_limit,
            "exact_prompt_tokens": prompt_tokens,
            "prompt_tokens": prompt_tokens,
            "configured_max_tokens": configured_max_tokens,
            "safety_margin_tokens": safety_margin_tokens,
            "available_tokens": available_tokens,
            "effective_max_tokens": effective_max_tokens,
            "minimum_output_tokens": minimum_output_tokens,
            "sufficient": effective_max_tokens >= minimum_output_tokens,
        }
        yield ToolTurnChunk.started(metadata=capacity)
        if effective_max_tokens < minimum_output_tokens:
            raise ToolTurnContextCapacityError(capacity)
        payload = {
            "model": self.config.get("model", {}).get("name", self.config["model"]["path"]),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": effective_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": generation.enable_thinking},
        }
        self._add_optional_sampling(payload, generation)

        start = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        call_parts: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        for event in self.server.stream_chat_events(payload):
            stream_error = RuntimeStreamError.from_event("sglang", event)
            if stream_error is not None:
                raise stream_error
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(content)
                    yield ToolTurnChunk.text(content)
                for raw_call in delta.get("tool_calls") or []:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    index = raw_call.get("index")
                    if not isinstance(index, int) or isinstance(index, bool):
                        raise RuntimeError("SGLang returned a tool call without a valid index")
                    current = call_parts.setdefault(
                        index, {"id": None, "name": None, "arguments": []}
                    )
                    function = raw_call.get("function") or {}
                    if raw_call.get("id"):
                        current["id"] = raw_call["id"]
                    if function.get("name"):
                        current["name"] = function["name"]
                    argument_delta = function.get("arguments") or ""
                    current["arguments"].append(argument_delta)
                    yield ToolTurnChunk.tool_delta(
                        ToolCallDelta(
                            index=index,
                            id=raw_call.get("id"),
                            function_name=function.get("name"),
                            arguments_delta=argument_delta,
                        )
                    )
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        end = time.perf_counter()

        if not text_parts and not call_parts and finish_reason is None:
            raise SGLangIncompleteStreamError(
                "SGLang native tool stream ended without content, tool calls, "
                "a finish reason, or an explicit error"
            )

        calls = tuple(self._assemble_tool_call(index, item) for index, item in sorted(call_parts.items()))
        text = "".join(text_parts)
        generated_tokens = (
            int(usage.get("completion_tokens", 0))
            if usage is not None
            else len(
                self.tokenizer.encode(
                    text + "".join(call.argument_text for call in calls),
                    add_special_tokens=False,
                )
            )
        )
        if usage is not None:
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
        ttft = None if first_token_at is None else first_token_at - start
        decode_sec = None if first_token_at is None else max(end - first_token_at, 1e-9)
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_sec=ttft,
            tokens_per_sec=0.0 if decode_sec is None else generated_tokens / decode_sec,
            wall_sec=end - start,
        )
        turn = AssistantTurn(
            text=text,
            tool_calls=calls,
            finish_reason=finish_reason,
            metrics=metrics,
        )
        if calls:
            session.add_assistant_tool_message(text, calls)
        else:
            session.add_assistant_message(text)
        if self.log_writer is not None:
            result = GenerationResult(text=text, metrics=metrics)
            self.log_writer.write(
                generation_event("sglang", session, result, self.health(), event_type="tool_turn")
            )
        yield ToolTurnChunk.completed(turn)

    @staticmethod
    def _add_optional_sampling(
        payload: dict[str, Any], generation: GenerationConfig
    ) -> None:
        if generation.top_k is not None:
            payload["top_k"] = generation.top_k
        if generation.repetition_penalty is not None:
            payload["repetition_penalty"] = generation.repetition_penalty

    @staticmethod
    def _assemble_tool_call(index: int, item: dict[str, Any]) -> ToolCall:
        call_id = item.get("id")
        name = item.get("name")
        argument_text = "".join(item.get("arguments", []))
        error: str | None = None
        arguments: dict[str, Any] | None = None
        if not isinstance(call_id, str) or not call_id:
            error = "tool call ID is missing"
            call_id = f"invalid_{index}"
        elif not isinstance(name, str) or not name:
            error = "tool function name is missing"
            name = ""
        else:
            try:
                parsed = json.loads(argument_text)
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must decode to an object")
                arguments = parsed
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"invalid tool arguments: {exc}"
        return ToolCall(
            id=call_id,
            index=index,
            function_name=name,
            argument_text=argument_text,
            arguments=arguments,
            parsing_error=error,
        )

    def tokenize(
        self,
        text_or_messages: Any,
        *,
        generation: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        if self.tokenizer is None:
            raise RuntimeError("runtime is not loaded")
        if isinstance(text_or_messages, list):
            if generation is None:
                generation = GenerationConfig.from_dict(self.config.get("generation", {}))
            try:
                text = self.tokenizer.apply_chat_template(
                    text_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=generation.enable_thinking,
                    tools=tools,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    text_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(self.tokenizer.encode(str(text_or_messages), add_special_tokens=False))

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def _launch_command(self) -> list[str]:
        server_cfg = self.config.get("server", {})
        model_cfg = self.config["model"]
        context_cfg = self.config.get("context", {})
        gpu_cfg = self.config.get("gpu", {})

        cmd = [
            server_cfg.get("python", "python"),
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_cfg["path"],
            "--host",
            str(server_cfg.get("host", "127.0.0.1")),
            "--port",
            str(server_cfg.get("port", 31000)),
            "--dtype",
            str(model_cfg.get("dtype", "bfloat16")),
            "--context-length",
            str(context_cfg.get("max_context_tokens", 4096)),
            "--mem-fraction-static",
            str(gpu_cfg.get("memory_fraction", 0.80)),
            "--log-level",
            str(server_cfg.get("log_level", "info")),
        ]
        if model_cfg.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")
        reasoning_parser = server_cfg.get("reasoning_parser")
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", str(reasoning_parser)])
        tool_call_parser = server_cfg.get("tool_call_parser")
        if tool_call_parser:
            cmd.extend(["--tool-call-parser", str(tool_call_parser)])
        model_name = model_cfg.get("name")
        if model_name:
            cmd.extend(["--served-model-name", str(model_name)])
        return cmd

    def _api_base(self) -> str:
        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", 31000)
        return f"http://{host}:{port}"

    def _log_dir(self) -> Path:
        log_cfg = self.config.get("logging", {})
        log_dir = Path(log_cfg.get("path", "experiments/logs"))
        if not log_dir.is_absolute():
            log_dir = self.project_root / log_dir
        return log_dir
