# Core Platform Architecture

Milestone 1 freezes the core AgentCore platform architecture. The project is a
runtime-independent orchestration layer for persistent AI agents. It is not an
LLM runtime and does not implement model kernels, schedulers, or inference
engines.

## Stable Public Surface

Normal user code should interact with:

- `AgentLab`
- `Agent`
- `Session`

Runtime adapters, server process helpers, logging internals, and benchmark
scripts are implementation details unless a contributor is extending the
platform.

## Agent

`Agent` represents one AI agent backed by one persistent `Session`.

Responsibilities:

- own a conversation session;
- expose `ask()` for conversational generation;
- expose `stream()` through the underlying runtime API;
- provide `reset()`;
- report agent-level statistics such as turns, last TTFT, last generation
  speed, prompt tokens, and generated tokens.

Dependencies:

- depends on `AgentLab` for runtime access;
- depends on `Session` for persistent history;
- depends on runtime-independent `GenerationResult` metrics.

`Agent` does not implement tools, planning, memory, filesystem access, Git
operations, sandboxing, or function calling.

## AgentLab

`AgentLab` is the application entry point.

Responsibilities:

- load configuration;
- construct the selected runtime adapter;
- start and stop runtime lifecycle;
- provide health and statistics;
- run warmup;
- create and manage sessions;
- create `Agent` objects;
- forward generation calls to the active runtime.

Dependencies:

- YAML configuration;
- one runtime adapter;
- `SessionStore`;
- JSONL logging writer when enabled.

`AgentLab` deliberately does not expose backend-specific server APIs to user
code.

## Runtime Abstraction

The `Runtime` interface defines the contract shared by all backends:

- `load()`
- `shutdown()`
- `ready()`
- `health()`
- `warmup()`
- `create_session()`
- `generate()`
- `stream()`
- `tokenize()`
- `statistics()`

Runtime adapters own backend-specific concerns such as process launch,
OpenAI-compatible HTTP calls, tokenizer behavior, and response parsing.

## Session

`Session` stores persistent conversation state.

Responsibilities:

- system prompt;
- ordered messages;
- transcript export;
- reset;
- `created_at` and `updated_at`;
- derived `turn_count`.

`SessionStore` adds first-class multi-session management:

- create;
- get by id;
- list;
- reset;
- delete;
- clear.

Sessions are in-memory only in Milestone 1.

## Runtime Adapters

Implemented adapters:

- `TransformersRuntime`
- `SGLangRuntime`
- `LMDeployRuntime`

Server runtimes share common lifecycle code through `ServerProcess`:

- subprocess launch;
- environment and PATH handling;
- server log files;
- readiness polling;
- clean shutdown;
- OpenAI-compatible streaming POST helper.

Backend-specific code remains in each adapter:

- command-line arguments;
- configuration interpretation;
- tokenizer details;
- response parsing;
- backend health extras.

## Health

Runtime health is normalized across backends.

Common fields include:

- runtime name;
- backend type;
- model path;
- readiness;
- server ready time when applicable;
- warmup wall time when available;
- GPU name;
- GPU memory used and total;
- process PID and endpoint for server runtimes;
- last error.

Runtime-specific extras are allowed but should not be required by normal user
code.

## Warmup

Warmup is explicit and runtime-independent:

```python
lab.start()
lab.warmup()
```

Warmup performs a short generation request and records metrics separately from
normal generation events.

## Benchmark

Milestone 1 includes a persistent 10-turn benchmark script that uses only the
public `AgentLab` API. It is intended for runtime comparison and regression
checks, not as a production workload.

The benchmark records:

- prompt tokens;
- generated tokens;
- TTFT;
- tokens/sec;
- wall time;
- runtime health;
- JSONL log path.

## Logging

Generation and warmup events are written as JSONL when configured.

Each event records:

- timestamp;
- event type;
- runtime;
- session id;
- turn;
- generation metrics;
- normalized health;
- status.

Logs are stored under `experiments/logs/` by default and are not committed.

## Configuration

Runtime selection and generation defaults are YAML-driven.

Current configs:

- `config/transformers-a100.yaml`
- `config/sglang-a100.yaml`
- `config/lmdeploy-a100.yaml`

Configuration covers:

- runtime name;
- model path;
- dtype;
- GPU selection;
- context length;
- generation defaults;
- server launch settings;
- logging.

## Tests

Milestone 1 includes lightweight tests that do not load the 27B model and do
not require a GPU:

- data structures;
- runtime contract with a fake runtime;
- session store;
- multi-session behavior;
- agent behavior.

Runtime integration tests are opt-in through:

```bash
A100_AGENT_LAB_RUN_INTEGRATION=1 python -m pytest -m integration
```

## Intentionally Not Implemented

Milestone 1 intentionally does not include:

- tool calling;
- filesystem tools;
- Git tools;
- sandboxing;
- planning;
- long-term memory;
- vector stores;
- multi-agent orchestration;
- authentication;
- production HTTP API;
- queueing;
- multi-user scheduling;
- model/runtime optimization;
- custom inference kernels.

These belong to later milestones.
