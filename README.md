# AgentCore

AgentCore is an experimental platform for building persistent coding-agent
systems on an NVIDIA A100 80GB PCIe server.

This project is not an LLM runtime. It is the application and orchestration
layer that sits on top of existing runtimes such as:

- SGLang
- vLLM
- LMDeploy
- HuggingFace Transformers

The goal is to provide stable Python abstractions for runtime management,
persistent conversations, native model tool calls, generation metrics,
structured traces, managed workspaces, local Git inspection, and explicitly
approved coding operations.

Current runtime roles:

- Primary runtime: SGLang
- Native multi-protocol tool runtime: vLLM
- Secondary runtime: LMDeploy
- Reference runtime: HuggingFace Transformers

The platform currently includes runtime adapters for all four backends while
keeping the public `AgentLab` API runtime-independent.

## Current Status

AgentCore's primary coding workflow is a persistent, incremental native-tool
loop. `ToolLoopAgent` lets the model inspect the real workspace, request one or
more structured tools, receive each concrete result, recover from failures,
run configured checks, inspect Git diff, and then provide a final response.
Qwen, Mistral, and OpenAI Harmony protocols share the same host-side tool and
safety implementation.

The platform provides:

- `AgentLab`, `Agent`, persistent `Session` transcripts, and `TaskReport`;
- Transformers, SGLang, LMDeploy, and vLLM runtime adapters;
- workspace-confined, bounded read, search, exact-edit, and write operations;
- complete pre-mutation previews and approval for every edit, write, and check;
- symbolic allowlisted checks with trusted argument vectors and `shell=False`;
- local Git status and diff inspection without model-visible commit or push;
- structured JSONL traces, cooperative cancellation, and bounded agent loops;
- a local tool-agent CLI plus the existing HTTP server and remote client;
- a frozen daily-coding qualification manifest.

The previous ActionPlan/PlanProposal workflow remains available as a legacy
compatibility mode. It is not the recommended coding-agent path.

AgentCore still does not expose model-visible shell access, arbitrary
subprocess execution, network Git operations, automatic Git commits,
production authentication, or persistent server storage.

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

The HTTP API retains the legacy proposal workflow for compatibility. Mutating
plans are never executed automatically and Git commits are never created
automatically. Native `ToolLoopAgent` orchestration is currently local-first
and is not exposed through the distributed API.

For streamed planning, use `POST /v1/tasks/{task_id}/proposals/stream`. The
stream exposes only visible assistant text plus structured operational events.
Cancellation is cooperative via `POST /v1/tasks/{task_id}/cancel`.

See [HTTP API](docs/architecture/HTTP_API.md) for endpoint details and SSE
format.

## Local Tool Agent

Run the recommended incremental coding workflow directly in one process,
without FastAPI, HTTP, SSE, or `agentclient`:

```bash
agentcore-local \
  --agent tool-loop \
  --config config/sglang-qwen-tools.yaml \
  --workspace workspace/project \
  --prompt-file task.txt \
  --trace-file trace.jsonl
```

The model receives native tools rather than a request for a complete plan.
Read-only calls run under policy; every `edit`, `write_file`, and `run_check`
call displays its complete AgentCore-generated effect and requires a unique
approval tied to the call ID and preview digest. `qwen-tools` remains a CLI
compatibility alias.

For normal daily use, install the two explicit local profiles and run Qwen by
default:

```bash
agentcore-local \
  --agent tool-loop \
  --workspace workspace/project \
  --prompt-file task.txt

agentcore-local \
  --agent tool-loop \
  --profile strong \
  --workspace workspace/project \
  --prompt-file task.txt
```

`fast` is the implicit tool-loop profile and selects Qwen3.6/SGLang. `strong`
selects gpt-oss/vLLM/Harmony explicitly. Public profile starting points are
`config/sglang-qwen-tools.yaml` and `config/vllm-harmony-tools.yaml`; actual
model paths and runtime executables belong in ignored local profile files.
Configured local runs automatically retain an ordered trace and append one
privacy-bounded aggregate metrics record after each terminal task.

The legacy planner remains available explicitly:

```bash
agentcore-local \
  --config config/sglang-a100-iterative.yaml \
  --planner iterative \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file result.jsonl
```

Planner modes produce ActionPlan proposals and use `TaskExecutor`. They are
supported for serialized-plan and API compatibility, not recommended for new
coding-agent work. Build and test execution remains available only through
trusted symbolic `run_check` configuration and explicit approval.

See [Local Mode](docs/architecture/LOCAL_MODE.md) for topology, commands,
previews, diagnostics, and exit codes; [Native Tool Agent](docs/architecture/QWEN_TOOL_AGENT.md)
for loop safety; and [Native Tool Protocols](docs/architecture/NATIVE_TOOL_PROTOCOLS.md)
for adapter responsibilities. [Planner v2](docs/architecture/PLANNER_V2.md)
documents the legacy planner mode.

## Model Positioning

- **Qwen3.6-27B:** default fast local coding model using SGLang and the native
  Qwen tool parser. It is the normal choice for interactive daily work.
- **gpt-oss-120b:** optional strong experimental profile for controlled,
  broader, or reasoning-heavy tasks using vLLM/Harmony. It is not
  production-proven and has materially higher latency and VRAM use.
- **Qwen3-Coder-30B-A3B-Instruct and Devstral Small 2 24B:** evaluated
  experimental options, not primary recommendations.

On the frozen daily-coding milestone suite, gpt-oss completed 9/10 tasks and
Qwen3.6 completed 5/5 tasks in a preselected representative subset. These are
qualification outcomes, not statistically universal success rates; every
accepted result also received independent human-equivalent review. See the
[milestone decision](docs/milestones/DS4_STYLE_TOOL_AGENT.md) and
[benchmark manifest](benchmarks/daily-coding/manifest.yaml).

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

Current architecture documents:

- [Core Platform](docs/architecture/CORE_PLATFORM.md)
- [Runtime Support](docs/architecture/RUNTIME_SUPPORT.md)
- [Coding Workspace](docs/architecture/CODING_WORKSPACE.md)
- [Local Mode](docs/architecture/LOCAL_MODE.md)
- [Native Tool Agent](docs/architecture/QWEN_TOOL_AGENT.md)
- [Native Tool Protocols](docs/architecture/NATIVE_TOOL_PROTOCOLS.md)
- [DS4-Style Tool Agent Milestone](docs/milestones/DS4_STYLE_TOOL_AGENT.md)
- [Daily Local Operation](docs/operations/DAILY_USE.md)
- [Planner v2 (legacy compatibility)](docs/architecture/PLANNER_V2.md)
- [Task Workflow](docs/architecture/TASK_WORKFLOW.md)
- [HTTP API](docs/architecture/HTTP_API.md)
- [Roadmap](docs/architecture/ROADMAP.md)
