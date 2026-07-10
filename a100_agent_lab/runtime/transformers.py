from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from a100_agent_lab.generation.config import GenerationConfig
from a100_agent_lab.generation.result import GenerationMetrics, GenerationResult
from a100_agent_lab.generation.stream import StreamChunk
from a100_agent_lab.logging.events import generation_event
from a100_agent_lab.logging.writer import JsonlWriter
from a100_agent_lab.runtime.base import Runtime
from a100_agent_lab.runtime.health import normalized_health, query_gpu
from a100_agent_lab.sessions.session import Session


class TransformersRuntime(Runtime):
    def __init__(self, config: dict[str, Any], *, log_writer: JsonlWriter | None = None):
        self.config = config
        self.log_writer = log_writer
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.load_sec: float | None = None
        self.last_warmup_wall_sec: float | None = None
        self.last_error: str | None = None
        self.device = config.get("gpu", {}).get("device", "cuda:0")

    def load(self) -> None:
        import torch

        self.torch = torch
        model_cfg = self.config["model"]
        model_path = model_cfg["path"]
        dtype = self._torch_dtype(model_cfg.get("dtype", "bfloat16"))
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": {"": self.device},
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                **load_kwargs,
            )
        self.model.eval()
        self.load_sec = time.perf_counter() - start

    def shutdown(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def health(self) -> dict[str, Any]:
        gpu = {}
        if self.torch is not None and self.torch.cuda.is_available():
            gpu = {
                "cuda_available": True,
                "device": self.device,
                "name": self.torch.cuda.get_device_name(0),
                "memory_allocated_mib": round(self.torch.cuda.memory_allocated() / 1024 / 1024, 2),
                "memory_reserved_mib": round(self.torch.cuda.memory_reserved() / 1024 / 1024, 2),
            }
        extra = {"load_time_sec": self.load_sec}
        if gpu:
            extra.update(
                {
                    "gpu_memory_allocated_mib": gpu["memory_allocated_mib"],
                    "gpu_memory_reserved_mib": gpu["memory_reserved_mib"],
                }
            )
        return normalized_health(
            runtime_name="transformers",
            backend_type="in_process",
            model_path=self.config.get("model", {}).get("path"),
            ready=self.ready(),
            warmup_wall_sec=self.last_warmup_wall_sec,
            last_error=self.last_error,
            gpu=query_gpu(self.device),
            extra=extra,
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
            raise RuntimeError("runtime is not loaded")

        generation = GenerationConfig.from_dict(self.config.get("generation", {})).override(**kwargs)
        session.add_user_message(prompt)
        messages = session.transcript()
        encoded = self._encode_messages(messages, generation)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        yield StreamChunk.started(metadata={"prompt_tokens": prompt_tokens, "runtime": "transformers"})

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs = {
            **encoded,
            "max_new_tokens": generation.max_tokens,
            "do_sample": generation.temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if generation.temperature > 0:
            generate_kwargs["temperature"] = generation.temperature
            generate_kwargs["top_p"] = generation.top_p

        error: list[BaseException] = []

        def run_generate() -> None:
            try:
                with self.torch.inference_mode():
                    self.model.generate(**generate_kwargs)
            except BaseException as exc:  # propagate errors from the worker thread
                error.append(exc)

        start = time.perf_counter()
        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        first_token_at: float | None = None
        chunks: list[str] = []
        for chunk in streamer:
            if chunk and first_token_at is None:
                first_token_at = time.perf_counter()
            if chunk:
                yield StreamChunk.delta(chunk)
            chunks.append(chunk)

        thread.join()
        if error:
            raise RuntimeError("generation failed") from error[0]

        end = time.perf_counter()
        text = "".join(chunks)
        generated_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        ttft = None if first_token_at is None else first_token_at - start
        decode_sec = None if first_token_at is None else max(end - first_token_at, 1e-9)
        tokens_per_sec = 0.0 if decode_sec is None else generated_tokens / decode_sec
        session.add_assistant_message(text)

        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_sec=ttft,
            tokens_per_sec=tokens_per_sec,
            wall_sec=end - start,
        )
        result = GenerationResult(text=text, metrics=metrics)

        if self.log_writer is not None:
            self.log_writer.write(
                generation_event("transformers", session, result, self.health(), event_type=event_type)
            )

        yield StreamChunk.completed(text=text, metrics=metrics, metadata={"runtime": "transformers"})

    def stream(self, session: Session, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        yield from self._stream(session, prompt, event_type="generation", **kwargs)

    def tokenize(self, text_or_messages: Any) -> int:
        if self.tokenizer is None:
            raise RuntimeError("runtime is not loaded")
        if isinstance(text_or_messages, list):
            return int(self._encode_messages(text_or_messages)["input_ids"].shape[-1])
        return len(self.tokenizer.encode(str(text_or_messages), add_special_tokens=False))

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def _torch_dtype(self, name: str):
        if self.torch is None:
            raise RuntimeError("torch is not loaded")
        mapping = {
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        return mapping[name.lower()]

    def _encode_messages(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig | None = None,
    ):
        if generation is None:
            generation = GenerationConfig.from_dict(self.config.get("generation", {}))
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=generation.enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if hasattr(encoded, "items") and not hasattr(encoded, "shape"):
            return {key: value.to(self.device) for key, value in encoded.items()}
        return {"input_ids": encoded.to(self.device)}
