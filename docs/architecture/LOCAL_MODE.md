# Local AgentCore Mode

Local mode runs AgentCore orchestration directly in one process without
FastAPI, HTTP, SSE, or `agentclient`. The recommended coding path is the
incremental native tool workflow selected with `--agent tool-loop`.
`--agent qwen-tools` remains a compatibility alias. Legacy planner modes remain
available for ActionPlan and serialized-plan compatibility.

## Topologies

### Local Tool Agent

One process owns:

- runtime lifecycle and the persistent model transcript;
- `Agent`, `Session`, and `Task`;
- the topology-neutral `ToolLoopAgent`;
- workspace, exact edits, complete side-effect previews, approval, checks,
  Git inspection, event trace, and the authoritative `TaskReport`.

Inference may still be supplied by an externally managed SGLang or vLLM
server. The local process does not import `agentclient` or start an AgentCore
HTTP server subprocess.

```text
AgentLab / runtime
  -> ToolProtocolAdapter (Qwen, Mistral, or Harmony)
  -> ToolLoopAgent
  -> Workspace / Approval / run_check / Git
  -> Task / TaskReport / Event trace
```

The protocol adapter only encodes and decodes model-native messages. Tool
validation, approval, execution, limits, cancellation, and lifecycle are
shared. There are no per-model copies of the tool loop.

### Server And Remote Client

`agentcore-server` plus the lightweight `agentclient` package provide the
distributed HTTP/SSE topology. It retains the planner/proposal workflow for
compatibility. Native `ToolLoopAgent` HTTP exposure is deferred; local and
distributed algorithms are not duplicated.

### Legacy Local Planner

The local runner can still compose `SimpleLLMPlanner` or
`IterativeLLMPlanner`, `PlanProposal`, `ApprovalPolicy`, and `TaskExecutor`.
This path supports existing CLI and serialized-plan behavior but is not the
recommended coding-agent architecture.

## Tool-Agent Use

```bash
agentcore-local \
  --agent tool-loop \
  --config config/sglang-qwen-tools.yaml \
  --workspace workspace/project \
  --prompt-file task.txt \
  --trace-file trace.jsonl
```

The module form is equivalent:

```bash
python -m agentcore_server.local \
  --agent tool-loop \
  --config config/sglang-qwen-tools.yaml \
  --workspace workspace/project
```

When local profiles are installed, `--config` can be omitted. Native tool-loop
mode defaults to the `fast` Qwen profile; the strong profile is always selected
explicitly:

```bash
agentcore-local --agent tool-loop --workspace workspace/project
agentcore-local --agent tool-loop --profile strong --workspace workspace/project
```

Profiles are resolved from `AGENTCORE_PROFILE_DIR`, then
`~/.config/agentcore/profiles`, then `config/local` under the current project.
Legacy planner mode still requires an explicit `--config`.

Tool-agent commands are:

```text
/status
/diff
/report
/preview
/approve
/reject [reason]
/abort
/quit
/help
```

Read-only tools may execute automatically under policy. Every `edit`,
`write_file`, and `run_check` pauses for a decision on that exact native
tool-call ID. AgentCore computes the effect from the real workspace and shows
the complete preview; approval binds the call ID and preview digest. The effect
is recomputed immediately before execution, so a stale preview cannot mutate
the workspace. `/preview` reopens the retained complete artifact.

A rejected call performs no mutation and becomes a normal `role=tool` result,
allowing the model to recover within configured rejection and loop limits.
Check failures are handled the same way. AgentCore never creates a Git commit
automatically.

`--agent tool-loop` is mutually exclusive with planner-only flags such as
`--planner`, `--proposal-only`, and `--approve`. Blanket approval is not
available for the tool loop.

## Protocol Configuration

Select `tool_agent.protocol` as one of:

- `qwen` for native Qwen function calls, normally through SGLang
  `qwen3_coder`;
- `mistral` for native Mistral function calls through vLLM;
- `harmony` for gpt-oss through vLLM's OpenAI/Harmony parser.

Checked-in examples use relative placeholder model paths. Copy one to an
ignored local file and set the actual model path, Python/runtime executable,
port, context capacity, and trusted checks. The current starting points are:

- `config/sglang-qwen-tools.yaml` for Qwen3.6-27B;
- `config/vllm-harmony-tools.yaml` for gpt-oss-120b;
- `config/vllm-native-tools.yaml` for a Mistral-compatible model.

Machine-specific paths and trusted check environment values must not be added
to public examples.

The public examples also configure `logging.trace_path` and
`logging.metrics_path`. If no `--trace-file` is supplied, local tool-loop mode
creates a unique trace in `trace_path`. At terminal completion it appends one
aggregate metrics record containing model/profile identity, status, turns,
tool/check/approval counts, token totals, first-token latency, generation and
wall time, changed files, and trace location. It contains no prompts, tool
arguments, source text, diffs, process environment, or hidden reasoning.

## Safety And Capacity

The model-visible tools are limited to bounded directory/search/read and Git
inspection plus exact unique edit, whole-file write, and symbolic checks.
There is no arbitrary shell, network, package installation, commit, or push
tool. Workspace traversal and symlink escapes are rejected.

Before each model request, AgentCore tokenizes the exact rendered transcript
and native tool schemas, reserves a safety margin, and clamps output to the
remaining configured context. Requests below the minimum output reserve fail
visibly. Tool results and trace payloads are bounded; the transcript is not
rebuilt into a planner EvidencePack.

## Traces And Exit Codes

`--trace-file` writes ordered JSONL containing visible assistant output,
structured tool lifecycle events, preview identifiers and digests, decisions,
bounded results, and one terminal `TaskReport`. Hidden Harmony reasoning is not
rendered as assistant text or public trace output.

```text
0   success
2   CLI or configuration error
10  runtime unavailable
20  structurally invalid legacy proposal
21  legacy proposal rejected
22  explicit approval required
23  task failed
24  task cancelled
70  unexpected local error
```

## Legacy Planner Commands

Proposal-only diagnostics remain available for compatibility:

```bash
agentcore-local \
  --config config/sglang-a100-iterative.yaml \
  --planner iterative \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file proposal.jsonl
```

Legacy planner execution still requires explicit approval. See
[PLANNER_V2.md](PLANNER_V2.md); it is compatibility documentation, not the
recommended local coding workflow.
