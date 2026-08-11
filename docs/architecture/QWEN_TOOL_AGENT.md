# Native Tool Agent

## Scope

`ToolLoopAgent` is a local-first coding workflow for native tool-capable models.
The concrete protocols are Qwen function calling, Mistral function calling,
and OpenAI Harmony through an OpenAI-compatible SGLang or vLLM endpoint.
`QwenToolAgent` remains a compatibility alias. This is an incremental
native-tool loop, not a Planner v2 or ActionPlan mode. Existing planners remain
available and unchanged.

The design is behaviorally inspired by the native agent in
[`antirez/ds4`](https://github.com/antirez/ds4), particularly its persistent
conversation, tool-result feedback, exact editing, recoverable tool failures,
steering between turns, and concise final answer. AgentCore does not copy DS4
source code and is not affiliated with or endorsed by antirez.

## Conversation

One model-native chat transcript is retained for the task:

1. AgentCore appends the user instruction.
2. The runtime returns visible assistant text and structured native tool calls.
3. AgentCore validates and executes each call in index order.
4. A `role=tool` result with the matching call ID is appended.
5. The model receives the updated transcript and chooses another tool or a final
   assistant response.

Tool calls are obtained from structured `delta.tool_calls`; they are not
parsed from prose and are never represented as an ActionPlan or PlanProposal.
The Qwen adapter supplies `enable_thinking`, the Mistral adapter uses OpenAI
string tool arguments, and the Harmony adapter maps the safety prompt to
`developer` while keeping reasoning-channel text out of visible events.
Visible `assistant.*` events contain only assistant text; tool lifecycle events
remain separate.

Unlike DS4's native KV-session persistence, this first implementation sends
the ordinary text transcript on each request. It does not synthesize an
EvidencePack or rebuild a complete task plan.

## Tools And Safety

The initial read-only tools are `list_directory`, `search_files`, `read_file`,
`git_status`, and `git_diff`. Results are deterministic and bounded. Hidden
directories and `.git` are skipped during discovery, binary content searches
are skipped, and path resolution retains AgentCore's traversal and symlink
escape checks.

The side-effecting tools are `edit`, `write_file`, and `run_check`:

- `edit` performs one exact unique replacement. Zero or multiple matches fail
  without mutation and the failure is returned to the model.
- `write_file` writes complete contents and is intended primarily for new
  files.
- `run_check` accepts only a trusted symbolic check configured with a fixed
  argv. It uses `shell=False`, the workspace root as cwd, bounded output, a
  timeout, and the existing controlled environment.

Every side-effecting call requires a new explicit decision for its concrete
tool-call ID. Approval is never inherited by a later call. Rejection performs
no side effect and becomes a structured tool result so the model can reconsider.
There is no arbitrary shell, network, package-install, commit, or push tool.

Before approval, AgentCore computes the concrete effect against the current
workspace. Exact edits must have one match and display a complete prospective
unified diff. Whole-file writes display their complete prospective diff, and
checks display the trusted argv, cwd, timeout, and `shell=False`. Each preview
has a stable digest bound to the tool-call ID. AgentCore recomputes the preview
immediately before execution and rejects it as stale if the workspace changed.
Normal events remain bounded, while local mode renders the complete preview and
can retain it in an isolated directory selected with
`--approval-preview-dir`. `/preview` displays the pending artifact again.

Concrete edit limits bound old text, new text, prospective diff bytes, changed
lines, and whole-file write bytes. Oversized edits execute nothing and tell
the model to split the operation into smaller exact edits. Operator rejections are
normal tool results. Separate total and consecutive rejection limits allow
bounded recovery without treating two rejected attempts as terminal.

## Lifecycle And Limits

A Task starts with the loop and completes only after the model returns a final
non-tool response. Ordinary failures such as a missing file, ambiguous edit,
or nonzero check are returned to the model and do not end the Task. Runtime,
workspace-safety, cancellation, and configured loop-limit failures are
terminal. One authoritative TaskReport is emitted after the terminal Task
transition. AgentCore never commits automatically.

For explicitly named configured checks and Git-diff requirements, a plain final
answer is not accepted until the checks have passed and `git_diff` has been
inspected after the latest mutation. AgentCore returns a concise completion
requirement to the persistent transcript so the model can continue; it does not
synthesize a success response.

Trusted `tool_agent` configuration bounds model turns, tool calls, consecutive
failures, individual and cumulative result bytes, read lines, directory depth,
search results, and native-turn context capacity. Before each request, the
runtime renders the exact transcript and native tool schemas with the configured
tokenizer, reserves `context_safety_margin_tokens` (default 128), and clamps
generation to the remaining context. When output capacity drops below
`context_recovery_target_tokens` (default 2048), the loop deterministically
elides the oldest replayable read-only tool result and repeats the exact
preflight before making a model request. The assistant tool call, role=tool
message, call ID, success state, byte count, and content digest remain in place;
the model can rerun the read, search, listing, status, or diff tool if it needs
the omitted data. At least `preserve_recent_tool_results` results (default 8)
remain complete, and edit, write, approval, rejection, and check results are
never elided. A request is still refused when fewer than
`minimum_output_tokens` (default 256) remain and no eligible result can recover
capacity. This is bounded native transcript maintenance, not Planner evidence
selection, task reconstruction, or an EvidencePack.

The normal model-turn limit is 40. At that boundary, the shared ToolLoopAgent
may grant one completion runway of at most 12 turns (absolute limit 52) only
when the last eight turns contain an actual workspace change followed by a
successful configured check or Git-diff inspection and are not dominated by
repeated identical discovery calls. The decision uses host-observed tool
results, never model prose or another model. Rejection limits, cancellation,
runtime failures, and context preflight remain authoritative. A runway cannot
recurse. `agent.turn_runway.granted` and passive metrics record the bounded
grant and its use without recording source or prompt content.

An OpenAI-compatible runtime may report a generation failure as a top-level
SSE `error` object even when the HTTP response status is 200. AgentCore
preserves its type, message, and code as a runtime failure. A stream with no
content, tool call, finish reason, or explicit error is rejected as incomplete
rather than treated as an empty assistant response.

## Local Use

Select `tool_agent.protocol` as `qwen`, `mistral`, or `harmony` and configure
the runtime's matching native tool parser. The local command is:

```bash
agentcore-local \
  --agent tool-loop \
  --config config/local/native-tool-model.yaml \
  --workspace /path/to/workspace \
  --prompt-file task.txt \
  --trace-file trace.jsonl \
  --no-color
```

The commands `/status`, `/diff`, `/report`, `/preview`, `/abort`, `/quit`, and
`/help` are available. `/approve` and `/reject [reason]` decide only the pending
side-effecting tool call and its exact preview digest. One plain-text steering
message can be queued between tool turns.

`--agent qwen-tools` remains a compatibility alias. Both agent names are
mutually exclusive with `--planner`, `--proposal-only`, and `--approve`;
blanket approval is intentionally absent.

## Deliberate DS4 Deviations

- Inference uses model-native function calls through an OpenAI-compatible
  SGLang or vLLM service rather than DSML.
- Arbitrary bash is replaced by configured symbolic checks.
- Workspace confinement, per-call approval, JSONL traces, and TaskReport
  lifecycle semantics remain AgentCore responsibilities.
- Anchored `[upto]` editing and native KV-cache persistence are deferred.
- Distributed HTTP exposure is deferred; the domain loop is topology-neutral
  so a later server adapter can reuse it.
