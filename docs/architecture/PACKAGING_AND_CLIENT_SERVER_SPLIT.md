# Packaging And Client/Server Split

AgentCore must become easy to run from a different workstation than the A100
server. The target architecture is:

```text
agentclient on a workstation
        |
        | HTTP/SSE
        v
agentcore-server on the A100 server
        |
        v
SGLang / LMDeploy / Transformers / workspace / Git / task execution
```

The Git repository remains a monorepo, but packaging should split the code into
three installable distributions:

- `agentcore-protocol`;
- `agentcore-server`;
- `agentclient`.

This document started as a migration plan. Phase 1 and Phase 2 have extracted
`agentcore-protocol` and `agentclient`. Phase 3 extracts the server/domain code
into `agentcore-server` while preserving protocol semantics and runtime
behavior.

## Implementation Status

Current package split:

- `agentcore-protocol`: installable shared protocol/client package.
- `agentclient`: installable lightweight remote CLI package.
- `agentcore-server`: installable server/domain/runtime package.

The server implementation now lives under:

```text
packages/agentcore-server/src/agentcore_server/
```

The legacy import path:

```python
import a100_agent_lab
```

is retained only as a deprecated compatibility shim that re-exports
`agentcore_server`.

## Current Import Audit

Historically, the `a100_agent_lab` package mixed protocol, server, runtime, and
CLI concerns. The active implementation has now moved to `agentcore_server`.

### Protocol Types Mixed With Server Internals

Protocol-like types currently live in the server/runtime package:

- `a100_agent_lab.events.AgentEvent`
- `a100_agent_lab.generation.result.GenerationMetrics`
- `a100_agent_lab.generation.result.GenerationResult`
- `a100_agent_lab.generation.stream.StreamChunk`
- `a100_agent_lab.server.schemas.CreateAgentRequest`
- `a100_agent_lab.server.schemas.CreateTaskRequest`
- `a100_agent_lab.server.schemas.CreateProposalRequest`
- `a100_agent_lab.server.schemas.RejectProposalRequest`
- `a100_agent_lab.server.schemas.CancelTaskRequest`
- `a100_agent_lab.server.events.format_sse`

These belong in a shared protocol package because both server and remote client
need to serialize, deserialize, and reason about the same HTTP/SSE payloads.

Some domain objects are also returned over HTTP and should either get protocol
DTOs or explicit response schemas:

- task dictionaries from `Task.as_dict()`;
- task reports from `TaskReport.as_dict()`;
- plan proposal dictionaries from `PlanProposal.as_dict()`;
- action plan dictionaries from `ActionPlan.as_dict()`;
- action execution results from `TaskExecutionResult.as_dict()`;
- Git command result shapes.

The protocol package should not import the domain classes themselves. It should
define stable wire schemas that mirror the current HTTP representation.

### CLI Imports That Violate Client/Server Separation

The current local CLI is embedded and imports server/domain code directly:

- `scripts/agentcore_cli.py` imports `AgentLab`;
- `scripts/agentcore_cli.py` imports `AgentEvent`;
- `scripts/agentcore_cli.py` imports `TaskStatus`;
- it creates an in-process `Agent`;
- it executes proposals directly through local Python objects.

That is correct for a development CLI, but it is not a lightweight remote
client. The production client CLI must not import:

- `Agent`;
- `AgentLab`;
- `TaskExecutor`;
- `Workspace`;
- runtime adapters;
- FastAPI;
- server state;
- server application objects.

The current `scripts/demo_http_agent.py` already demonstrates remote protocol
use, but it still imports `Workspace` for local demo setup. A packaged
`agentclient` must not do that.

### Dependencies That Leak Into A Lightweight Client

The current root `pyproject.toml` installs everything together:

- `torch`;
- `transformers`;
- `PyYAML`;
- `fastapi`;
- `uvicorn`.

This is too heavy for a workstation client. It also makes a remote-only install
look like it requires CUDA/model/runtime dependencies.

The lightweight client should only need:

- `agentcore-protocol`;
- `httpx`;
- `rich` for terminal rendering.

It must not depend on:

- FastAPI;
- Uvicorn;
- Torch;
- Transformers;
- SGLang;
- LMDeploy;
- workspace/Git/task execution server code.

## Proposed Monorepo Layout

```text
packages/
  agentcore-protocol/
    pyproject.toml
    src/
      agentcore_protocol/
        __init__.py
        version.py
        errors.py
        events.py
        schemas.py
        stream.py
        client.py
        sse.py
        compatibility.py
    tests/
      test_schemas.py
      test_events.py
      test_client.py
      test_sse.py

  agentcore-server/
    pyproject.toml
    config/
    src/
      agentcore_server/
        __init__.py
        cli.py
        api/
        agents/
        executor/
        generation/
        health/
        logging/
        planner/
        planning/
        runtime/
        server/
        sessions/
        tasks/
        workspace/
    tests/
      test_http_server.py
      test_runtime_contract.py
      ...

  agentclient/
    pyproject.toml
    src/
      agentclient/
        __init__.py
        cli.py
        config.py
        rendering.py
        commands.py
    tests/
      test_cli.py
      test_fake_server.py
      test_stream_rendering.py

docs/
scripts/
config/
examples/
```

The legacy `a100_agent_lab` top-level package remains only as a deprecated
compatibility shim. It is not a second implementation.

## Package Names And Import Names

### agentcore-protocol

Distribution name:

```text
agentcore-protocol
```

Import name:

```python
import agentcore_protocol
```

Primary public API:

```python
from agentcore_protocol import AgentCoreClient
from agentcore_protocol import AgentEvent
from agentcore_protocol import API_VERSION
```

Responsibilities:

- request schemas;
- response schemas;
- event schemas;
- stream schemas;
- normalized API errors;
- HTTP client;
- SSE client;
- protocol version and compatibility checks.

Console entry points: none.

### agentcore-server

Distribution name:

```text
agentcore-server
```

Import names:

```python
import agentcore_server
import a100_agent_lab
```

`agentcore_server` is the canonical server/domain import. `a100_agent_lab` is a
temporary deprecated compatibility import that re-exports `agentcore_server`.

Console entry point:

```text
agentcore-server = agentcore_server.cli:main
```

Responsibilities:

- AgentCore domain logic;
- `Agent`;
- `Task`;
- `Workspace`;
- file/Git helpers;
- planner;
- executor;
- runtime adapters;
- FastAPI application;
- server process launcher.

### agentclient

Distribution name:

```text
agentclient
```

Import name:

```python
import agentclient
```

Console entry point:

```text
agentclient = agentclient.cli:main
```

Responsibilities:

- interactive remote CLI;
- terminal rendering;
- steering commands;
- local client configuration;
- HTTP/SSE calls through `agentcore_protocol.AgentCoreClient`.

`agentclient` must communicate only with HTTP/SSE. It must not import server
domain objects.

## Dependency Graph

```text
agentcore-protocol
  -> httpx
  -> standard library
  -> optional schema library only if shared and lightweight

agentcore-server
  -> agentcore-protocol
  -> fastapi
  -> uvicorn
  -> PyYAML
  -> torch
  -> transformers
  -> runtime-specific packages available in server environments

agentclient
  -> agentcore-protocol
  -> rich
  -> standard library
```

No dependency may point from `agentcore-protocol` to `agentcore-server` or from
`agentclient` to `agentcore-server`.

## Optional Development Extras

Root development extras should install all packages editable:

```text
.[dev]
```

Suggested package extras:

```text
agentcore-protocol[dev]
  -> pytest
  -> respx or pytest-httpx if chosen later

agentcore-server[dev]
  -> pytest
  -> httpx

agentcore-server[transformers]
  -> torch
  -> transformers

agentcore-server[sglang]
  -> no direct install by default; expect external runtime environment

agentcore-server[lmdeploy]
  -> no direct install by default; expect external runtime environment

agentclient[dev]
  -> pytest
```

The server package should avoid forcing every runtime dependency into every
install. A production A100 server can install the needed runtime extra or rely on
pre-existing SGLang/LMDeploy environments.

## Protocol Versioning

The protocol package owns version identifiers.

Suggested constants:

```python
API_VERSION = "v1"
SCHEMA_VERSION = "0.2"
MIN_CLIENT_SCHEMA_VERSION = "0.2"
MIN_SERVER_SCHEMA_VERSION = "0.2"
```

### API Version

The current HTTP paths remain:

```text
/v1/...
```

`API_VERSION` changes only when endpoint semantics or path structure become
incompatible.

### Schema Compatibility

Schema version changes when request/response/event payloads change.

Compatible changes:

- adding optional response fields;
- adding optional request fields with defaults;
- adding new event types;
- adding new endpoints.

Incompatible changes:

- removing fields;
- changing field meaning;
- making optional fields required;
- changing event ordering guarantees;
- changing approval or execution semantics.

### Compatibility Check

`GET /health` should eventually include:

```json
{
  "api_version": "v1",
  "schema_version": "0.2",
  "server_package": "agentcore-server",
  "server_version": "0.2.0"
}
```

The client should call health on startup and reject clearly if:

- API major version differs;
- server schema is older than the client minimum;
- client schema is older than the server minimum.

This check should be advisory first, then enforced once the split is stable.

## HTTP Client Responsibilities

`agentcore_protocol.AgentCoreClient` should wrap the current HTTP API:

- `health()`;
- `create_agent(...)`;
- `list_agents()`;
- `get_agent(agent_id)`;
- `delete_agent(agent_id)`;
- `create_task(agent_id, ...)`;
- `get_task(task_id)`;
- `task_report(task_id)`;
- `propose(task_id, ...)`;
- `propose_stream(task_id, ...)`;
- `get_proposal(proposal_id)`;
- `approve(proposal_id)`;
- `reject(proposal_id, reason)`;
- `execute(proposal_id)`;
- `cancel(task_id, reason)`;
- `git_status(agent_id)`;
- `git_diff(agent_id)`;
- `task_events(task_id)`.

The client should expose typed responses where practical, but must keep access
to raw JSON for forward compatibility.

## Normalized Errors

The protocol package should define errors independent of FastAPI:

- `AgentCoreError`;
- `AgentCoreHTTPError`;
- `AgentCoreConnectionError`;
- `AgentCoreProtocolError`;
- `AgentCoreCompatibilityError`.

Server-side FastAPI exceptions remain server details. The protocol client maps
HTTP status codes and response bodies into these normalized errors.

## SSE Client Responsibilities

The SSE client should:

- connect to a URL;
- parse `event:`, `id:`, `data:`;
- ignore heartbeat comments;
- deserialize data into `AgentEvent`;
- preserve event order;
- expose reconnect metadata later, but not implement complex reconnect policy in
  the first split.

It should support both:

- task event streams;
- streamed proposal responses.

## Migration Strategy

The migration should happen in small, reviewable phases.

### Phase 1: Extract Protocol Types In Place

Create `packages/agentcore-protocol` and move/copy only protocol DTOs:

- `AgentEvent`;
- `GenerationMetrics` or protocol equivalent;
- `StreamChunk` or stream DTO equivalent;
- request schemas;
- response schema definitions;
- SSE parsing/formatting;
- error types;
- protocol version constants.

Keep compatibility imports from `a100_agent_lab` temporarily:

```python
from agentcore_protocol import AgentEvent
```

then update server imports.

### Phase 2: Add HTTP/SSE Client Library

Implement `AgentCoreClient` in `agentcore-protocol`.

Refactor `scripts/demo_http_agent.py` to use it. This proves the client library
can drive the current server without importing server internals.

### Phase 3: Extract AgentClient CLI

Create `packages/agentclient`.

Move remote-only CLI logic there. The new CLI must:

- accept `--server`;
- default to `http://127.0.0.1:8080`;
- use `AgentCoreClient`;
- render `AgentEvent`;
- never import `AgentLab`, `Agent`, `Workspace`, `TaskExecutor`, or runtime
  adapters.

The current embedded `scripts/agentcore_cli.py` may remain as a development
command, but it should be renamed or clearly marked as local embedded mode.

### Phase 4: Extract Server Package

Create `packages/agentcore-server` and move server/domain code from
`a100_agent_lab` to `agentcore_server`.

Moved server scope:

- `agentcore_server.cli`;
- `agentcore_server.server`;
- `agentcore_server.api`;
- domain packages for agents, sessions, tasks, workspace, executor, planner,
  runtime adapters, health/statistics, and logging;
- configs and examples as package data or repo-level examples.

The old `a100_agent_lab` import path is kept only as a warning-emitting shim.

### Phase 5: Root Monorepo Tooling

Add root tooling for:

- editable install of all packages;
- test all packages;
- package smoke tests;
- optional build checks.

Avoid a broad source rename during this phase.

### Phase 6: Deprecation And Cleanup

Deprecate direct public imports that are no longer part of the remote client
surface.

Document:

- server install path;
- client install path;
- dev install path;
- compatibility matrix.

## Preserving Git History

Prefer `git mv` when moving complete files into package directories.

Recommended preservation strategy:

- move protocol-like files first with minimal edits;
- keep compatibility shims where old imports existed;
- avoid formatting-only rewrites during moves;
- split moves and behavior changes into separate commits;
- run tests after each phase.

Large renames should be isolated from behavior changes. The
`a100_agent_lab` to `agentcore_server` rename is tracked with `git mv` and
compatibility shims.

## Tests

### Protocol Package Tests

- event serialization/deserialization;
- request schema defaults;
- response schema compatibility;
- normalized error mapping;
- SSE parsing:
  - `event`;
  - `id`;
  - `data`;
  - heartbeat comments;
  - ordering.

### AgentClient Tests

Use a fake HTTP/SSE server. Do not start AgentCore server or load models.

Test:

- health check;
- create agent;
- create task;
- streamed proposal;
- approval/rejection;
- execution;
- cancellation;
- task events;
- rendering receives only protocol events;
- client install does not import server dependencies.

### Server Tests

Current FastAPI tests should be updated to use shared protocol schemas.

Test:

- endpoint compatibility;
- response fields match protocol DTOs;
- SSE output parses with protocol SSE parser;
- approval semantics unchanged;
- cancellation semantics unchanged.

### Packaging Smoke Tests

Run in isolated virtual environments:

```bash
python -m pip install dist/agentcore_protocol-*.whl
python -c "import agentcore_protocol"
python -c "import torch"  # must fail unless installed independently
```

```bash
python -m pip install dist/agentclient-*.whl
agentclient --help
python -c "import fastapi"       # must fail unless installed independently
python -c "import transformers"  # must fail unless installed independently
```

```bash
python -m pip install dist/agentcore_server-*.whl
agentcore-server --help
```

The key acceptance criterion is that `agentclient` installs and runs without
server/runtime dependencies.

## Compatibility Concerns

Risks:

- accidentally changing HTTP payload names while extracting schemas;
- creating circular dependencies between protocol and server packages;
- keeping hidden server imports in the client CLI;
- making `agentclient` pull CUDA/runtime dependencies through transitive imports;
- breaking existing scripts before replacement commands exist;
- version skew between a new client and old server.

Mitigations:

- keep `/v1` endpoint semantics unchanged;
- add protocol golden JSON fixtures;
- test client against fake HTTP/SSE server;
- test server output with protocol parser;
- add import-guard tests for `agentclient`;
- keep compatibility imports during transition;
- add health-based compatibility checks before strict enforcement.

## First Implementation Target

The first code change should not move the whole project. It should extract only
the protocol layer and add a remote client library.

Smallest useful first PR:

1. create `packages/agentcore-protocol`;
2. define `AgentEvent`, stream/event DTOs, request DTOs, errors, version
   constants;
3. implement `AgentCoreClient` and SSE parser;
4. update `scripts/demo_http_agent.py` to use `AgentCoreClient`;
5. add package smoke test proving no FastAPI/Torch/Transformers dependency.

Only after this should `agentclient` become a separate package.
