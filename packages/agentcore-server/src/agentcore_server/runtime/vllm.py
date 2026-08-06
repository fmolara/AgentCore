from __future__ import annotations

from agentcore_server.runtime.sglang import SGLangRuntime


class VLLMRuntime(SGLangRuntime):
    """vLLM's OpenAI endpoint with the shared native tool-turn protocol path."""

    def _launch_command(self) -> list[str]:
        server = self.config.get("server", {})
        model = self.config["model"]
        context = self.config.get("context", {})
        gpu = self.config.get("gpu", {})
        cmd = [
            str(server.get("python", "python")),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model["path"]),
            "--host",
            str(server.get("host", "127.0.0.1")),
            "--port",
            str(server.get("port", 31000)),
            "--dtype",
            str(model.get("dtype", "auto")),
            "--max-model-len",
            str(context.get("max_context_tokens", 4096)),
            "--gpu-memory-utilization",
            str(gpu.get("memory_fraction", 0.90)),
            "--enable-auto-tool-choice",
        ]
        parser = server.get("tool_call_parser")
        if parser:
            cmd.extend(["--tool-call-parser", str(parser)])
        name = model.get("name")
        if name:
            cmd.extend(["--served-model-name", str(name)])
        if model.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")
        if server.get("enforce_eager", False):
            cmd.append("--enforce-eager")
        for key, flag in (
            ("tokenizer_mode", "--tokenizer-mode"),
            ("config_format", "--config-format"),
            ("load_format", "--load-format"),
            ("kv_cache_dtype", "--kv-cache-dtype"),
        ):
            value = server.get(key)
            if value:
                cmd.extend([flag, str(value)])
        return cmd

    def _server_env_updates(self) -> dict[str, str]:
        updates = super()._server_env_updates()
        updates["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        return updates
