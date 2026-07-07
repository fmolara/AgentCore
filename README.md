# AgentCore

AgentCore is an experimental platform for building persistent coding-agent
systems on an NVIDIA A100 80GB PCIe server. The Python package is currently
named `a100_agent_lab`.

This project is not an LLM runtime. It is the application and orchestration
layer that sits on top of existing runtimes such as:

- SGLang
- LMDeploy
- HuggingFace Transformers

The goal is to provide stable Python abstractions for runtime management,
persistent conversations, generation metrics, structured logging, and future
agent capabilities such as tools, filesystem access, and Git integration.

Current runtime roles:

- Primary runtime: SGLang
- Secondary runtime: LMDeploy
- Reference runtime: HuggingFace Transformers

The platform currently includes runtime adapters for all three backends while
keeping the public `AgentLab` API runtime-independent.

## Public API

Normal application code should interact with:

- `AgentLab`
- `Agent`
- `Session`

```python
from a100_agent_lab import AgentLab

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
from a100_agent_lab import AgentLab

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
- [Roadmap](docs/architecture/ROADMAP.md)
