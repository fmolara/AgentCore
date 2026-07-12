# agentcore-server

`agentcore-server` contains the AgentCore server/domain implementation:

- `AgentLab`, `Agent`, `Session`;
- managed workspaces, safe file editing, and local Git helpers;
- tasks, reports, checkpoints, restore planning, and restore execution;
- action plans, approval policies, plan proposals, and task execution;
- runtime adapters for Transformers, SGLang, and LMDeploy;
- FastAPI HTTP API and Server-Sent Events;
- structured logging and runtime health/statistics.

The package depends on `agentcore-protocol` for wire-level DTOs, events, errors,
and client/server protocol version metadata. It does not depend on
`agentclient`.

## Install

Install the base server package plus HTTP server dependencies:

```bash
pip install "agentcore-server[server]"
```

For the in-process Transformers reference runtime:

```bash
pip install "agentcore-server[server,transformers]"
```

SGLang and LMDeploy installations are externally managed. AgentCore launches
those runtimes through configured executables or virtual environments; it does
not vendor or install them.

The `sglang` and `lmdeploy` extras only install Python-side helper dependencies
used by the adapters, such as tokenizers. They do not install SGLang or
LMDeploy themselves.

## Run

```bash
agentcore-server \
  --config config/sglang-a100.yaml \
  --host 127.0.0.1 \
  --port 8080
```

The default bind host remains `127.0.0.1`. AgentCore currently has no production
authentication layer, so do not expose the server on an untrusted network.
