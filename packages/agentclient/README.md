# agentclient

`agentclient` is the lightweight remote terminal client for AgentCore.

It talks to an AgentCore server over HTTP and Server-Sent Events only. It does
not import server internals, runtime adapters, CUDA, Torch, Transformers,
SGLang, LMDeploy, Workspace, Git, TaskExecutor, or planner implementations.

## Install

Install the client package together with `agentcore-protocol`:

```bash
pip install agentcore-protocol agentclient
```

When using this monorepo directly, build both wheels and install them into a
small client-side virtual environment.

## Connect

The default endpoint is localhost:

```bash
agentclient
```

Connect to a remote server:

```bash
agentclient --server http://192.168.1.20:8080
```

HTTPS URLs are accepted:

```bash
agentclient --server https://agentcore.example.internal
```

The `--workspace` argument is server-side request metadata. The client never
reads or writes that path locally.

## SSH tunnel

AgentCore currently has no production authentication layer. Prefer a private
network or an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 user@a100-server
agentclient --server http://127.0.0.1:8080
```

## Commands

```text
/status   Show current remote task/proposal state
/plan     Request or re-request a streamed plan proposal
/approve  Explicitly approve and execute the current proposal
/reject   Reject the current proposal
/diff     Fetch Git diff from the remote workspace
/report   Fetch the current task report
/abort    Request cooperative cancellation
/quit     Exit
/help     Show command help
```

Mutating proposals are never executed automatically. Git commits are never
created automatically.

## Security

The AgentCore HTTP server is still a development interface. It should bind to
localhost by default and should not be exposed on an untrusted network without
an authentication and authorization layer.
