# A100 Agent Lab

A100 Agent Lab is an experimental platform for building persistent coding-agent
systems on an NVIDIA A100 80GB PCIe server.

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

The first implementation includes only the minimal platform skeleton and a
working `TransformersRuntime` reference backend. SGLang and LMDeploy process
management will be added after the public API is stable.

## Public API

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

## Smoke Test

```bash
cd /home/federico.molara/a100-agent-lab
/home/federico.molara/venv/a100-runtime/bin/python scripts/smoke_transformers.py
```

Generation events are written as JSONL under `experiments/logs/`.
