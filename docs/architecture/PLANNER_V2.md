# Planner v2

Planner v2 adds bounded workspace discovery before producing a reviewable
`ActionPlan`. It is available as `planner.mode: iterative`; `simple` remains
the compatibility default.

## State Machine

Every model response selects one explicit phase:

```text
EXPLORE
  -> bounded read-only observations
  -> EXPLORE, FINAL, or CANNOT_PLAN

FINAL
  -> ActionPlan validation
  -> PlanProposal
  -> explicit operator approval
  -> existing TaskExecutor
```

`max_rounds` bounds executed exploration rounds. After the final exploration
round, one terminal model call may return only `FINAL` or `CANNOT_PLAN`; it
cannot execute more discovery. `CANNOT_PLAN` and exhausted limits produce visible planning failures. An
exploration response never becomes a `PlanProposal`, never invokes
`TaskExecutor`, and never changes task execution status.

## Task Context

Every round uses an ephemeral, stateless planning session. Its effective
request contains the complete immutable task instruction exactly once,
workspace identity, accumulated observations, and remaining budget. Planner
correctness therefore does not depend on a backend preserving conversation
state between generation calls, and a request never duplicates the task
instruction.

Trace events identify the session, task-context digest, workspace root,
context mode, observation count, and remaining budget for each round.

## Exploration Actions

Planner v2 accepts only these internal discovery actions:

- `list_directory`: recursively lists sorted entries, identifies files,
  directories, and symlinks, excludes `.git`, and applies depth/count limits.
- `search_files`: applies a glob to basenames, returns sorted paths, optionally
  performs a bounded literal UTF-8 content search, and reports binary,
  undecodable, symlink, and truncation cases.
- `read_file`: reads a regular UTF-8 file with explicit line and byte limits;
  directories, missing files, binary files, and encoding failures become
  bounded observations.

All actions resolve through the existing `Workspace` boundary. Absolute paths,
`..`, and symlink escapes are hard failures. The complete round is validated
before any action runs. Once validated, ordinary per-action failures do not
prevent later actions in the round from producing observations. Malformed
actions, security violations, and global budget exhaustion stop planning.

No exploration action writes files, runs commands, changes Git, creates a
checkpoint, or accesses the network.

## Default Limits

```yaml
planner:
  mode: iterative
  exploration:
    max_rounds: 3
    max_actions_per_round: 8
    max_total_actions: 20
    max_directory_depth: 4
    max_files_returned: 100
    max_single_file_bytes: 65536
    max_total_observation_bytes: 262144
    max_observation_text_per_action: 65536
```

Results are deterministic: traversal and matches are sorted, file reads are
bounded, and truncation is explicit.

Before each model call, the planner tokenizes the complete stateless request
and caps `max_tokens` to the configured context window minus a small safety
margin. The effective budget is emitted as `planning.generation_budget`. If
fewer than 64 output tokens remain, planning fails visibly instead of sending
a request that cannot produce a structured response.

## Final Plan Validation

The final response uses the existing executable action schema. AgentCore
rejects malformed actions, unknown actions, workspace escapes, directory
targets for `read_file`, unknown check names, empty plans, and a final plan
that remains only a collection of discovery reads after exploration.

This validation does not claim to judge code quality. A structurally valid but
weak executable plan remains visible and can be rejected by the operator.
Each final response creates a new proposal ID and a new approval decision.

## Allowlisted Checks

Build and test execution is represented by a symbolic action:

```json
{"type": "run_check", "check": "test"}
```

Trusted configuration supplies the immutable command:

```yaml
workspace:
  checks:
    build:
      argv: ["make"]
      timeout_sec: 30
    test:
      argv: ["make", "test"]
      timeout_sec: 60
```

The model cannot supply or alter `argv`. Checks use `shell=False`, the
workspace root as `cwd`, no stdin, bounded stdout/stderr collectors, and a
timeout. On POSIX, timeout cleanup terminates the process group so descendants
do not remain running; other platforms fall back to terminating the child
process.

The environment inherits only:

- `PATH`, `HOME`, locale, and `TMPDIR`;
- common compiler/build variables (`CC`, `CXX`, flags, `MAKEFLAGS`, and
  `PKG_CONFIG_PATH`);
- additional variables explicitly trusted in the check configuration.

Environment values are not included in action results or traces. Normal exit,
nonzero exit, timeout, launch failure, and output truncation are distinct.
Nonzero and timed-out checks fail task execution, not planning transport.

`run_check` always requires explicit approval. It is never available during
exploration.

## Events

Planning lifecycle uses structured `AgentEvent` names such as:

```text
planning.started
exploration.round.started
exploration.plan.generated
exploration.action.started
exploration.action.completed
exploration.action.failed
exploration.observations.ready
exploration.round.completed
replan.started
planning.final_plan.generated
planning.failed
plan.proposed
```

`assistant.started`, `assistant.delta`, and `assistant.completed` contain only
visible model output. Planning events are never serialized as assistant text.
Local JSONL traces and distributed SSE carry the same logical events.

## Topologies

Both `LocalAgentCoreApp` and `AgentCoreServerState` select planners through the
same factory and configuration. The planner, explorer, proposal, approval,
executor, workspace, and check implementations are shared. Only event
rendering or transport differs.

Local proposal-only diagnosis:

```bash
agentcore-local \
  --config config/sglang-a100-iterative.yaml \
  --planner iterative \
  --workspace workspace/project \
  --prompt-file task.txt \
  --proposal-only \
  --trace-file result.jsonl
```

The distributed server reads the same `planner.mode` configuration. No
arbitrary shell execution or automatic Git commit is supported.
