# Coding Workspace

Milestone 2 makes AgentCore a coding-workspace platform. It does not make the
agent autonomous. The platform can represent work, propose reviewable plans,
execute approved actions, track changes, stream observable events, and recover
workspace state through checkpoints.

The current architecture is intentionally bounded:

- no shell access;
- no automatic Git commits;
- no network Git operations;
- no production authentication;
- no persistent server storage;
- no autonomous long-running planner.

## Workspace

`Workspace` represents the external project state owned by an agent.

Responsibilities:

- define a root directory;
- track the current working directory;
- enforce read-only or read-write mode;
- validate every path;
- block path traversal and symlink escape;
- expose metadata.

`Session` remains conversation state. `Workspace` is the filesystem boundary.
An `Agent` owns both, but they solve different problems.

## Files

`agent.files` exposes safe file editing primitives backed by the workspace path
validator.

Supported operations include:

- read text;
- write text;
- append text;
- replace text;
- read line ranges;
- write lines;
- apply unified diffs.

Mutating operations return structured results with changed paths and byte/line
counts. They do not execute commands and cannot escape the workspace root.

## Git

`agent.git` is a constrained local Git helper scoped to the workspace root.

Supported operations:

- `is_repo()`;
- `init()`;
- `status()`;
- `diff()`;
- `add(paths)`;
- `commit(message)`;
- `log(limit)`;
- `current_branch()`.

Git commands always run with `cwd = workspace.root`. Paths passed to `add()` are
validated by the workspace. No clone, fetch, pull, push, arbitrary Git argument
passthrough, or shell API is exposed.

AgentCore never creates a Git commit automatically. Commits are explicit API
calls only.

## Task

`Task` is the execution container.

It tracks:

- id;
- title;
- description;
- status;
- timestamps;
- metadata;
- checkpoints;
- cancellation state.

Task statuses are:

- `created`;
- `running`;
- `completed`;
- `failed`;
- `cancelled`.

Task does not plan, call the model, or decide which actions to run.

## TaskExecutor

`TaskExecutor` executes an explicit sequence of actions for one task.

Responsibilities:

- start the task if needed;
- execute actions sequentially;
- record every action result;
- emit structured events;
- stop immediately on failure;
- check cooperative cancellation between actions;
- produce a final `TaskReport`.

The executor owns no planning policy and performs no retries or concurrency.
Each action is treated as an atomic operation.

## ActionPlan

`ActionPlan` is a serialized, validated list of explicit actions.

Current action types:

- `read_file`;
- `write_file`;
- `replace_text`;
- `create_checkpoint`;
- `git_status`;
- `git_diff`;
- `task_report`.

Unknown actions and malformed schemas are rejected. Execution-time path safety is
delegated to the workspace APIs.

## PlanProposal

`PlanProposal` wraps an `ActionPlan` for review.

It records:

- task id;
- title;
- summary;
- action plan;
- approval requirements;
- status;
- timestamps;
- metadata.

Proposal statuses are:

- `proposed`;
- `approved`;
- `rejected`;
- `executed`.

The LLM may propose a plan, but it cannot execute it. Execution requires the
proposal object to pass validation and approval policy checks.

## ApprovalPolicy

`ApprovalPolicy` classifies actions before execution.

The default policy:

- allows read-only actions;
- requires approval for mutating file/checkpoint actions;
- does not require approval for local Git status/diff;
- supports allowlists and denylists.

Mutating plans cannot execute automatically. Approval and execution are separate
operations in both Python and HTTP APIs.

## Checkpoints

Task checkpoints capture task-local workspace state metadata.

Each checkpoint records:

- id;
- task id;
- label;
- description;
- timestamp;
- Git branch;
- Git status;
- Git diff;
- metadata;
- restorable file snapshots when available.

Checkpoints are not Git commits. They are AgentCore task artifacts.

## Restore

Restore is explicit and safety checked.

`task.plan_restore_checkpoint(...)` is read-only and reports:

- target checkpoint;
- current Git status and diff;
- checkpoint diff;
- files that would be modified;
- files that would be overwritten;
- warnings;
- `safe_to_restore`.

`task.restore_checkpoint(...)` requires a safe plan unless `force=True` is
provided. Restores use workspace-bound file APIs and do not commit to Git.

## TaskReport

`TaskReport` is the current structured view of a task.

It includes:

- task identity and status;
- timestamps;
- failure/cancellation fields;
- current Git branch;
- current Git status;
- current Git diff;
- files changed when derivable;
- task metadata.

Reports are available through Python, CLI, and HTTP.

## Event Model

`AgentEvent` is the common observable event schema.

Each event includes:

- timestamp;
- event type;
- task id when available;
- session id when available;
- human-readable summary;
- structured payload.

Events are designed for traceability. They do not expose raw hidden reasoning or
chain-of-thought.

Operational events include:

- task lifecycle;
- plan proposal/approval/rejection;
- execution lifecycle;
- action start/completion/failure;
- workspace modification;
- checkpoint creation;
- Git diff capture;
- cancellation.

## Streaming

Runtime token streaming is normalized through `StreamChunk`.

`Agent.stream()` converts chunks into user-visible assistant events:

- `assistant.started`;
- `assistant.delta`;
- `assistant.completed`;
- `assistant.failed`.

Planner streaming uses the same events. The visible assistant response is
collected and parsed as an `ActionPlan` only after completion. Invalid JSON or
invalid action schemas fail cleanly and do not create executable proposals.

HTTP streams these events through Server-Sent Events.

## Cancellation

Cancellation is cooperative.

`Task.request_cancellation(...)` marks intent. `TaskExecutor` checks that flag
before starting an action and after each completed action. The current atomic
action is allowed to finish safely.

Cancellation emits:

- `cancellation.requested`;
- `cancellation.completed`.

AgentCore does not guarantee interruption of an already-running GPU request,
filesystem operation, or runtime call.

## Interaction Summary

The normal workflow is:

1. Create an `Agent`.
2. Create a workspace-bound `Task`.
3. Ask the LLM for a `PlanProposal`.
4. Review the proposal and approval requirements.
5. Approve or reject explicitly.
6. Execute through `TaskExecutor`.
7. Observe events through CLI, JSONL, or SSE.
8. Inspect `TaskReport` and Git diff.
9. Optionally create checkpoints or restore them explicitly.

## Outside Current Architecture

Milestone 2 intentionally does not include:

- autonomous planning loops;
- long-term memory;
- tool registry;
- shell access;
- arbitrary subprocess execution;
- production authentication;
- persistent HTTP server storage;
- multi-user deployment;
- distributed workers;
- automatic Git commits;
- network Git operations.
