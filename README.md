# AgentCore

AgentCore is an experimental platform for building persistent coding-agent
systems on an NVIDIA A100 80GB PCIe server.

This project is not an LLM runtime. It is the application and orchestration
layer that sits on top of existing runtimes such as:

- SGLang
- LMDeploy
- HuggingFace Transformers

The goal is to provide stable Python abstractions for runtime management,
persistent conversations, generation metrics, structured logging, managed
workspaces, local Git workflows, reviewable plans, and approved task execution.

Current runtime roles:

- Primary runtime: SGLang
- Secondary runtime: LMDeploy
- Reference runtime: HuggingFace Transformers

The platform currently includes runtime adapters for all three backends while
keeping the public `AgentLab` API runtime-independent.

## Current Status

Milestone 1, Core Platform, is complete. It established the runtime-independent
foundation:

- `AgentLab`, `Agent`, and `Session`;
- Transformers, SGLang, and LMDeploy runtime adapters;
- warmup and health reporting;
- persistent benchmarks;
- structured JSONL logging;
- runtime contract tests.

Milestone 2, Coding Workspace, is complete. AgentCore can now:

- manage workspace-scoped files;
- run constrained local Git workflows;
- create tasks;
- propose validated action plans;
- require explicit approval for mutating plans;
- execute approved plans;
- report task state and Git diffs;
- create checkpoints;
- plan and execute restores;
- expose an interactive CLI;
- expose a localhost HTTP API;
- stream visible assistant output and operational events through SSE;
- request cooperative cancellation.

AgentCore still does not expose shell access, arbitrary subprocess execution,
network Git operations, automatic Git commits, production authentication, or
persistent server storage.

## Public API

Normal application code should interact with:

- `AgentLab`
- `Agent`
- `Session`

```python
from agentcore_server import AgentLab

lab = AgentLab.from_config("config/transformers-a100.yaml")
lab.start()

session = lab.create_session(system_prompt="You are a concise coding agent.")
result = lab.generate(session, "Explain pointers in C.", max_tokens=64)

print(result.text)
print(result.metrics)

lab.shutdown()
```

The higher-level `Agent` abstraction owns one persistent session and exposes a
simple conversational API:

```python
from agentcore_server import AgentLab

lab = AgentLab.from_config("config/sglang-a100.yaml")
lab.start()
lab.warmup()

agent = lab.create_agent(system_prompt="You are a concise coding assistant.")
reply = agent.ask("Explain pointer arithmetic.")

print(reply.text)
print(agent.statistics())

lab.shutdown()
```

Each agent owns one managed workspace. Workspace operations are restricted to
the workspace root:

```python
agent = lab.create_agent(
    system_prompt="You are a concise coding assistant.",
    workspace_root="workspace/project-a",
)

agent.workspace.mkdir("notes")
agent.workspace.write_text("notes/pointers.txt", "Pointer arithmetic scales by element size.\n")

print(agent.workspace.list("notes"))
print(agent.workspace.read_text("notes/pointers.txt"))
```

Safe file editing primitives are available through `agent.files`, scoped to the
same workspace:

```python
agent.files.write_text("src/main.c", "int answer(void) { return 1; }\n")
agent.files.replace_text("src/main.c", "return 1", "return 42")
agent.files.append_text("README.md", "Notes from the agent.\n")

print(agent.files.read_lines("src/main.c", start=0, end=1))
```

Local Git operations are also scoped to the same workspace root. No clone,
fetch, pull, or push API is exposed:

```python
agent.git.init()
agent.workspace.write_text("notes/pointers.txt", "Pointer arithmetic scales by element size.\n")

print(agent.git.status().stdout)

agent.git.add(["notes/pointers.txt"])
agent.git.commit("Add pointer notes")

print(agent.git.log(limit=3).stdout)
```

Tasks are execution containers owned by an agent. They track status and metadata
but do not perform planning or tool calling:

```python
task = agent.create_task(
    title="Refactor parser",
    description="Replace placeholder token parsing logic.",
)

task.start()
agent.files.replace_text("src/parser.c", "return 0;", "return 1;")
print(task.report().as_dict())
task.complete()
```

Multiple independent sessions can be managed by id:

```python
first = lab.create_session(system_prompt="You are session one.")
second = lab.create_session(system_prompt="You are session two.")

lab.generate(first, "Explain malloc.")
lab.generate(second, "Explain fork.")

assert first.id != second.id
assert first.turn_count == 1
assert second.turn_count == 1

same_first = lab.get_session(first.id)
all_sessions = lab.list_sessions()
lab.reset_session(first.id)
lab.delete_session(second.id)
```

## Smoke Test

```bash
python scripts/smoke_transformers.py
```

Generation events are written as JSONL under `experiments/logs/`.

## HTTP Server

AgentCore also exposes a small localhost HTTP API with Server-Sent Events for
task execution traces:

```bash
agentcore-server \
  --config config/sglang-a100.yaml \
  --host 127.0.0.1 \
  --port 8080
```

Minimal lifecycle:

```bash
curl -s http://127.0.0.1:8080/health
```

Create an agent, create a task, request a proposal, approve explicitly, then
execute. Mutating plans are never executed automatically and Git commits are
never created automatically.

For streamed planning, use `POST /v1/tasks/{task_id}/proposals/stream`. The
stream exposes only visible assistant text plus structured operational events.
Cancellation is cooperative via `POST /v1/tasks/{task_id}/cancel`.

See [HTTP API](docs/architecture/HTTP_API.md) for endpoint details and SSE
format.

## Local Mode

AgentCore can also run orchestration directly in one process, without FastAPI,
HTTP, SSE, or `agentclient`:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project
```

The local composition reuses the same planner, proposal, approval,
`TaskExecutor`, workspace, and runtime implementations as distributed mode.
Proposal-only diagnostics are available with `--proposal-only` and
`--trace-file`. Execution still requires explicit approval, and Git commits
remain manual.

See [Local Mode](docs/architecture/LOCAL_MODE.md) for topology, commands,
diagnostics, and exit codes.

## Package Layout

AgentCore is now organized as a monorepo with separate installable packages:

- `packages/agentcore-protocol`: shared HTTP/SSE schemas, errors, event model,
  sync/async clients, and protocol version metadata.
- `packages/agentcore-server`: server/domain/runtime package with `AgentLab`,
  `Agent`, workspaces, tasks, planners, runtime adapters, FastAPI HTTP API, and
  the `agentcore-server` console command.
- `packages/agentclient`: lightweight remote terminal client. It talks only to
  an AgentCore server over HTTP/SSE and does not import server/runtime internals.

The old `a100_agent_lab` import path is retained only as a deprecated
compatibility shim.

Remote client usage:

```bash
agentclient --server http://127.0.0.1:8080
agentclient --server http://192.168.1.20:8080 --workspace /srv/workspaces/demo
```

The `--workspace` value is interpreted by the server. The client does not read
or write that path locally.

## Tests

Lightweight tests do not load the 27B model and do not require a GPU:

```bash
python -m pytest
```

Runtime integration tests are opt-in because they start real runtimes and may
load the model:

```bash
A100_AGENT_LAB_RUN_INTEGRATION=1 \
  python -m pytest -m integration
```

## Architecture

Milestone 1 architecture documents:

- [Core Platform](docs/architecture/CORE_PLATFORM.md)
- [Runtime Support](docs/architecture/RUNTIME_SUPPORT.md)
- [Coding Workspace](docs/architecture/CODING_WORKSPACE.md)
- [Local Mode](docs/architecture/LOCAL_MODE.md)
- [Task Workflow](docs/architecture/TASK_WORKFLOW.md)
- [HTTP API](docs/architecture/HTTP_API.md)
- [Roadmap](docs/architecture/ROADMAP.md)
