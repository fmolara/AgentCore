# agentcore-server

`agentcore-server` contains the AgentCore server/domain implementation:

- `AgentLab`, `Agent`, `Session`;
- managed workspaces, safe file editing, and local Git helpers;
- tasks, reports, checkpoints, restore planning, and restore execution;
- incremental native `ToolLoopAgent` orchestration with Qwen, Mistral, and
  Harmony protocol adapters;
- complete per-call approval previews, exact edits, and symbolic checks;
- legacy action plans, approval policies, plan proposals, and task execution;
- runtime adapters for Transformers, SGLang, vLLM, and LMDeploy;
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

## Local orchestration

The same distribution also provides a single-process orchestration command:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project
```

The recommended local workflow is the native tool loop. With an installed
`fast` profile it defaults to Qwen3.6/SGLang; `strong` selects
gpt-oss/vLLM/Harmony explicitly:

```bash
agentcore-local --agent tool-loop --workspace workspace/project
agentcore-local --agent tool-loop --profile strong --workspace workspace/project
```

Local mode reuses the server package's domain, approval, workspace, and runtime
implementations. It does not start FastAPI, use HTTP or SSE, or import
`agentclient`. The legacy planner/executor path remains available only with an
explicit configuration.

For a legacy proposal-only diagnostic run:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file result.jsonl
```

Without `--proposal-only` or `--approve`, `--prompt` and `--prompt-file` seed
the task and then enter the normal interactive command loop. The file is read
as UTF-8 and preserved exactly. Use `/approve` or `/reject` after reviewing the
proposal. The two prompt options are mutually exclusive.

Mutating actions require explicit approval, and non-interactive execution
always requires `--approve`. Git commits are never automatic.

Bounded workspace exploration is opt-in:

```bash
agentcore-local \
  --config config/sglang-a100-iterative.yaml \
  --planner iterative \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file result.jsonl
```

The model may request only bounded `list_directory`, `search_files`, and
`read_file` exploration. Configured build/test commands are exposed later as
symbolic `run_check` actions and still require explicit approval.
