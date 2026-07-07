# Task Workflow

Task is the execution container used by AgentCore for coding-agent work.
It is not a planner, memory system, shell, or tool-calling framework.

The current Task workflow provides a structured way to track work performed
through the existing Workspace, Git, and file-editing APIs.

## Responsibilities

A Task records:

- a unique id;
- title and description;
- status;
- timestamps;
- metadata;
- reports;
- explicit checkpoints;
- checkpoint comparisons;
- restore plans;
- explicit checkpoint restores.

An Agent owns multiple Tasks. A Session remains conversation state only.
Workspace represents the external filesystem state.

## Lifecycle

Task status values are:

- `created`
- `running`
- `completed`
- `failed`
- `cancelled`

Typical usage:

```python
task = agent.create_task(
    title="Refactor parser",
    description="Replace placeholder token parsing logic.",
)

task.start()
# perform workspace/file operations
task.complete()
```

Invalid transitions raise errors. Completing, failing, or cancelling a task
does not commit Git changes.

## Task Report

`task.report()` returns a structured, JSON-serializable report.

When the task is workspace-bound through an Agent, the report includes:

- task id, title, description, status;
- timestamps;
- current Git branch;
- current Git status;
- current Git diff;
- files changed when derivable;
- task metadata.

The report is read-only. It does not modify files and does not run arbitrary
commands.

## Checkpoints

Checkpoints are explicit task snapshots:

```python
checkpoint = task.create_checkpoint(
    "parser return value",
    "Parser now returns one token.",
)
```

A checkpoint records:

- checkpoint id;
- task id;
- timestamp;
- label;
- description;
- Git branch;
- Git status;
- Git diff;
- metadata.

For Agent-owned tasks, checkpoints also store internal file snapshots for files
changed at checkpoint time. These snapshots make explicit restore possible
without using shell access or Git reset.

Checkpoint accessors:

```python
task.checkpoints()
task.latest_checkpoint()
```

## Checkpoint Comparison

Checkpoint comparison is read-only:

```python
comparison = task.compare_checkpoints(first, second)
latest = task.compare_latest_checkpoint()
```

The comparison contains:

- checkpoint ids, labels, and timestamps;
- diff captured at checkpoint A;
- diff captured at checkpoint B;
- changed files at A and B;
- file sets added, removed, and common between the two checkpoint snapshots.

Comparison does not inspect or modify the current workspace.

## Restore Planning

Restore planning is read-only:

```python
plan = task.plan_restore_checkpoint(checkpoint)
```

The restore plan contains:

- target checkpoint id, label, and timestamp;
- current Git status;
- current Git diff;
- checkpoint diff;
- files that would be modified;
- files that would be overwritten;
- warnings;
- `safe_to_restore`.

The planner is conservative. If current uncommitted changes overlap with the
checkpoint files, `safe_to_restore` is false. If a checkpoint does not contain
restorable file snapshots, `safe_to_restore` is false.

## Restore Execution

Restore is explicit:

```python
result = task.restore_checkpoint(checkpoint)
```

If the restore plan is not safe, restore raises unless forced:

```python
result = task.restore_checkpoint(checkpoint, force=True)
```

Restore writes only files captured in the checkpoint's file snapshots. It
returns a structured result with:

- target checkpoint id and label;
- restored files;
- warnings;
- whether force was used;
- whether the original plan was safe.

Restore never commits automatically.

## Safety Model

Task workflow deliberately avoids broad execution capabilities:

- no shell access;
- no arbitrary subprocess execution;
- no automatic Git commit;
- no Git reset;
- no Git checkout;
- no network Git operations;
- no restore unless explicitly requested.

All file writes go through Workspace/File APIs. Workspace path resolution
enforces the workspace root and blocks path traversal and symlink escape.

Git interaction is local-only and workspace-bound. Task reporting, checkpoint
creation, comparison, and restore planning use read-only Git state. Restore
uses stored file snapshots, not shell commands.

## Workflow Example

```python
agent.git.init()
agent.files.write_text("src/parser.c", "int parse(void) { return 0; }\n")
agent.git.add(["src/parser.c"])
agent.git.commit("Prepare parser")

task = agent.create_task(
    title="Refactor parser",
    description="Record and restore parser edits.",
)
task.start()

agent.files.replace_text("src/parser.c", "return 0", "return 1")
print(agent.git.diff().stdout)

first = task.create_checkpoint("return one")

agent.files.replace_text("src/parser.c", "return 1", "return 2")
second = task.create_checkpoint("return two")

comparison = task.compare_checkpoints(first, second)
print(comparison.as_dict())

plan = task.plan_restore_checkpoint(first)
print(plan.as_dict())

if plan.safe_to_restore:
    result = task.restore_checkpoint(first)
else:
    result = task.restore_checkpoint(first, force=True)

print(result.as_dict())
print(agent.git.diff().stdout)
```

This sequence keeps Git history under user control. The final diff is visible,
but no commit is created by the Task workflow.
