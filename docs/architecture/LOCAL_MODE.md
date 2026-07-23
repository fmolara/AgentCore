# Local AgentCore Mode

AgentCore supports a local orchestration topology in addition to its existing
HTTP server topology. Local mode belongs to the `agentcore-server`
distribution; it is not a separate package.

## Topologies

### 1. Local AgentCore Process

Implemented in this phase.

One process owns AgentCore orchestration, including:

- runtime lifecycle;
- `Agent` and `Session`;
- `Task`;
- `SimpleLLMPlanner`;
- `PlanProposal` and `ApprovalPolicy`;
- `TaskExecutor`;
- workspace, files, and local Git;
- terminal rendering and ordered JSONL traces.

The process does not run FastAPI, use HTTP, emulate SSE, import `agentclient`,
or launch an AgentCore server subprocess. An inference backend such as SGLang
may still be an externally managed process launched through the existing
runtime adapter.

### 2. AgentCore Server and Remote Client

Implemented by `agentcore-server` plus the lightweight `agentclient` package.
The client communicates over HTTP and SSE using `agentcore-protocol`. This
topology is appropriate when the workspace and inference runtime live on a
server that is separate from the operator's machine.

### 3. Local AgentCore with Remote Inference

Future topology. AgentCore orchestration and the workspace would be local,
while model inference would be provided by a remote runtime endpoint. This
phase does not add that transport or alter runtime adapters.

## Shared Domain Path

Local and distributed modes use the same domain implementations:

```text
AgentLab
  -> Agent
  -> SimpleLLMPlanner
  -> PlanProposal / ApprovalPolicy
  -> TaskExecutor
  -> Workspace / Files / Git
  -> TaskReport
```

Local mode adds only a composition root, terminal approval flow, renderer, and
an `AgentEvent` sink. It does not contain local copies of planner, executor,
task, or workspace behavior.

`AgentLab` currently lives in `agentcore_server.api.client` even though it is a
topology-neutral composition object. Moving it is deferred to avoid combining
a namespace refactor with local mode.

## Interactive Use

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project
```

The same command is available as:

```bash
python -m agentcore_server.local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project
```

Interactive commands:

```text
/status
/plan
/approve
/reject
/diff
/report
/abort
/quit
/help
```

Every proposal is shown before execution. `/approve` is explicit, `/reject`
does not mutate the workspace, and `/abort` requests cooperative cancellation
between atomic actions. AgentCore never creates a Git commit automatically.

## Non-Interactive Use

Pass a task directly:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project \
  --prompt "Replace return 0 with return 1 in parser.c." \
  --approve
```

Or read the exact task from a file:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only
```

`--proposal-only` never executes actions and requires no approval. Any
non-interactive execution requires the visible `--approve` flag, including a
plan that contains only read-only actions.

## Proposal Diagnostics

An ordered public trace can be written as JSONL:

```bash
agentcore-local \
  --config config/sglang-a100.yaml \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file result.jsonl
```

The trace may include:

- the sanitized effective planner prompt;
- visible assistant text;
- raw final visible model text available to the planner;
- parsed `PlannerResult`, `PlanProposal`, and `ActionPlan`;
- structural validation and approval-policy results;
- execution events, `TaskReport`, and Git diff.

It does not request, expose, or store hidden chain-of-thought.

## Exit Codes

```text
0   success
2   CLI or configuration error
10  runtime unavailable
20  structurally invalid proposal
21  proposal rejected
22  explicit approval required
23  task failed
24  task cancelled
70  unexpected local error
```

A structurally valid but semantically poor proposal is not classified as
invalid. It remains visible and can be rejected by the operator.

## Runtime Configuration

Local mode uses the same runtime configuration as `agentcore-server`.
Machine-specific paths belong in ignored local configuration files rather than
public examples. For an externally managed SGLang installation, configure the
model path and executable path prefix in that local file, then pass it through
`--config`.

Runtime shutdown occurs in cleanup after success, rejection, cancellation, or
an operational error.
