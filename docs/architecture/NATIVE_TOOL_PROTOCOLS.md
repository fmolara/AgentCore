# Native Tool Protocols

AgentCore keeps one workspace-safe `ToolLoopAgent` and a deliberately small
protocol boundary for three model families exercised against real runtimes.
The adapter controls only message roles, assistant tool-call encoding,
model-specific request options, streamed delta decoding, and visible final
text. Tool validation, previews, approvals, execution, limits, events, and
TaskReport lifecycle remain shared.

| Protocol | Runtime parser | Adapter behavior |
| --- | --- | --- |
| Qwen | SGLang `qwen3_coder` | Preserves the existing transcript and passes `enable_thinking` through chat-template options. |
| Mistral | vLLM `mistral` | Uses OpenAI string-encoded assistant tool arguments and the model's native Mistral template. |
| Harmony | vLLM `openai` | Encodes the host safety instruction as `developer`, sends bounded `reasoning_effort`, and excludes reasoning deltas from visible output and public traces. |

Configuration selects `tool_agent.protocol`; there is no provider discovery
or generic multi-provider framework. SGLang and vLLM share the same normalized
OpenAI streaming assembly, call-ID handling, context preflight, and error
propagation. vLLM launch options are individually allowlisted in configuration;
no arbitrary server arguments or model-visible shell tool are introduced.

Qwen's prior `--agent qwen-tools` entry point and public Python names remain
aliases for compatibility. New local deployments should use `--agent
tool-loop`. Distributed exposure remains deferred.

The implementation is behaviorally inspired by antirez/ds4's persistent
incremental tool loop. No DS4 source code is copied, and AgentCore is not
affiliated with or endorsed by antirez.
