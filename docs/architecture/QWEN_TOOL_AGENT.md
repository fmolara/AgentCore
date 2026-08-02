# Qwen Native Tool Agent

## Scope

`QwenToolAgent` is a local-first coding workflow for Qwen models served by
SGLang. It is an incremental native-tool loop, not a Planner v2 or ActionPlan
mode. Existing simple and iterative planners remain available and unchanged.

The design is behaviorally inspired by the native agent in
[`antirez/ds4`](https://github.com/antirez/ds4), particularly its persistent
conversation, tool-result feedback, exact editing, recoverable tool failures,
steering between turns, and concise final answer. AgentCore does not copy DS4
source code and is not affiliated with or endorsed by antirez.

## Conversation

One Qwen chat transcript is retained for the task:

1. AgentCore appends the user instruction.
2. SGLang returns visible assistant text and structured native tool calls.
3. AgentCore validates and executes each call in index order.
4. A `role=tool` result with the matching call ID is appended.
5. Qwen receives the updated transcript and chooses another tool or a final
   assistant response.

Tool calls are obtained from SGLang's structured `delta.tool_calls` stream
using `qwen3_coder`. They are not parsed from prose and are never represented
as an ActionPlan or PlanProposal. Visible `assistant.*` events contain only
assistant text; tool lifecycle events remain separate.

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
  without mutation and the failure is returned to Qwen.
- `write_file` writes complete contents and is intended primarily for new
  files.
- `run_check` accepts only a trusted symbolic check configured with a fixed
  argv. It uses `shell=False`, the workspace root as cwd, bounded output, a
  timeout, and the existing controlled environment.

Every side-effecting call requires a new explicit decision for its concrete
tool-call ID. Approval is never inherited by a later call. Rejection performs
no side effect and becomes a structured tool result so Qwen can reconsider.
There is no arbitrary shell, network, package-install, commit, or push tool.

## Lifecycle And Limits

A Task starts with the loop and completes only after Qwen returns a final
non-tool response. Ordinary failures such as a missing file, ambiguous edit,
or nonzero check are returned to Qwen and do not end the Task. Runtime,
workspace-safety, cancellation, and configured loop-limit failures are
terminal. One authoritative TaskReport is emitted after the terminal Task
transition. AgentCore never commits automatically.

Trusted `tool_agent` configuration bounds model turns, tool calls, consecutive
failures, individual and cumulative result bytes, read lines, directory depth,
search results, and native-turn context capacity. Before each request, the
runtime renders the exact transcript and native tool schemas with the configured
tokenizer, reserves `context_safety_margin_tokens` (default 128), and clamps
generation to the remaining context. A request is not sent when fewer than
`minimum_output_tokens` (default 256) remain. This mode does not truncate the
transcript or invoke planner-style context compaction.

SGLang may report a generation failure as a top-level SSE `error` object even
when the HTTP response status is 200. AgentCore preserves its type, message,
and code as a runtime failure. A stream with no content, tool call, finish
reason, or explicit error is rejected as incomplete rather than treated as an
empty assistant response.

## Local Use

Use a Qwen/SGLang configuration with `server.tool_call_parser: qwen3_coder`,
thinking disabled, and trusted workspace checks:

```bash
agentcore-local \
  --agent qwen-tools \
  --config config/local/sglang-a100-qwen-tools.yaml \
  --workspace /path/to/workspace \
  --prompt-file task.txt \
  --trace-file trace.jsonl \
  --no-color
```

The commands `/status`, `/diff`, `/report`, `/abort`, `/quit`, and `/help` are
available. `/approve` and `/reject` decide only the pending side-effecting tool
call. One plain-text steering message can be queued between tool turns.

`--agent qwen-tools` is mutually exclusive with `--planner`,
`--proposal-only`, and `--approve`; blanket approval is intentionally absent.

## Deliberate DS4 Deviations

- Inference remains Qwen through the OpenAI-compatible SGLang service.
- AgentCore uses native Qwen function calls rather than DSML.
- Arbitrary bash is replaced by configured symbolic checks.
- Workspace confinement, per-call approval, JSONL traces, and TaskReport
  lifecycle semantics remain AgentCore responsibilities.
- Anchored `[upto]` editing and native KV-cache persistence are deferred.
- Distributed HTTP exposure is deferred; the domain loop is topology-neutral
  so a later server adapter can reuse it.
