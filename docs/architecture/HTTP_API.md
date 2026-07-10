# AgentCore HTTP API

The AgentCore HTTP API is the first network-accessible interface for the local
coding-agent platform. It is intentionally small: one process, in-memory state,
JSON commands, Server-Sent Events for task execution traces, and explicit
proposal approval.

This is not a production deployment layer yet. The default bind host is
`127.0.0.1`, there is no authentication, and state is not durable.

## Lifecycle

1. Start the server with a runtime configuration.
2. Create an agent bound to a workspace.
3. Create a task for that agent.
4. Request a plan proposal for the task.
5. Inspect the proposed action plan and approval requirements.
6. Approve or reject explicitly.
7. Execute only an approved mutating proposal.
8. Consume task events through SSE.
9. Inspect the final task report and Git diff.

Execution never implies approval, and no Git commit is created automatically.

## Launch

```bash
python scripts/agentcore_server.py \
  --config config/sglang-a100.yaml \
  --host 127.0.0.1 \
  --port 8080
```

Useful options:

- `--workspace-root PATH`: default parent directory for server-created workspaces.
- `--warmup`: warm up the runtime on startup.
- `--no-warmup`: skip runtime warmup.

## Endpoints

Health:

```http
GET /health
```

Agents:

```http
POST   /v1/agents
GET    /v1/agents
GET    /v1/agents/{agent_id}
DELETE /v1/agents/{agent_id}
```

Tasks:

```http
POST /v1/agents/{agent_id}/tasks
GET  /v1/tasks/{task_id}
GET  /v1/tasks/{task_id}/report
```

Planning:

```http
POST /v1/tasks/{task_id}/proposals
GET  /v1/proposals/{proposal_id}
POST /v1/proposals/{proposal_id}/approve
POST /v1/proposals/{proposal_id}/reject
```

Execution:

```http
POST /v1/proposals/{proposal_id}/execute
```

Workspace inspection:

```http
GET /v1/agents/{agent_id}/git/status
GET /v1/agents/{agent_id}/git/diff
```

Events:

```http
GET /v1/tasks/{task_id}/events
```

## Requests

Create an agent:

```json
{
  "system_prompt": "You are a concise coding assistant.",
  "workspace_root": "workspace/project-a",
  "workspace_mode": "read_write",
  "workspace_metadata": {},
  "generation_options": {}
}
```

Create a task:

```json
{
  "title": "Edit parser",
  "description": "Replace return 0 with return 1 in parser.c.",
  "metadata": {}
}
```

Request a proposal:

```json
{
  "instruction": "Replace return 0 with return 1 in parser.c.",
  "max_tokens": 512,
  "temperature": 0
}
```

Reject a proposal:

```json
{
  "reason": "The proposed file is not the intended target."
}
```

## Approval Semantics

Proposal generation and execution are separate operations.

Mutating plans cannot execute until an explicit approval request has succeeded.
Rejected proposals cannot execute. Duplicate execution requests are rejected.

The current mutating actions include file writes, text replacement, and
checkpoint creation. Read-only actions such as Git status, Git diff, file reads,
and task reports may execute under the default approval policy.

## SSE Format

Task events are serialized as Server-Sent Events:

```text
event: action.started
id: 7
data: {"event_type":"action.started","summary":"Action started: replace_text",...}
```

Events preserve the order in which AgentCore records them. The stream replays
existing task history first, then follows live events. A stream closes after a
terminal task execution event. During idle periods the server may send heartbeat
comments:

```text
: heartbeat
```

No raw chain-of-thought is exposed. Events contain only structured work state:
task lifecycle, plan proposal/approval/rejection, action status, workspace
modifications, checkpoints, Git diff, and final execution status.

## Security Model

This first server is intended for localhost development only:

- default bind host is `127.0.0.1`;
- no authentication is implemented yet;
- no arbitrary shell endpoint exists;
- no unrestricted Git command endpoint exists;
- no clone, fetch, pull, or push endpoint exists;
- all workspace operations continue to use AgentCore workspace validation.

Production concerns intentionally deferred:

- authentication and authorization;
- durable storage;
- multi-user isolation;
- process supervision;
- rate limits;
- distributed workers;
- remote workspace provisioning.
