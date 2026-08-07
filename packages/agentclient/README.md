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

For example, a server reachable directly through a trusted VPN may use:

```bash
agentclient --server http://10.121.0.10:8080
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

`agentclient` does not configure or manage the VPN or SSH tunnel. It reports
the connection failure observed at the configured URL. In particular, a
closed local forwarded port is reported as `CONNECTION_REFUSED`.

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

## Exit codes

`agentclient` exposes stable process exit codes for scripts and automation:

| Code | Name | Meaning |
|---:|---|---|
| 0 | `SUCCESS` | The invocation completed normally. |
| 2 | `CLI_USAGE_ERROR` | Invalid arguments, configuration, or non-interactive command usage. |
| 10 | `NETWORK_UNREACHABLE` | DNS, routing, connect timeout, or initial-response timeout. |
| 11 | `CONNECTION_REFUSED` | The operating system reported `ECONNREFUSED`. |
| 12 | `TLS_ERROR` | TLS negotiation or certificate validation failed. |
| 13 | `HTTP_ERROR` | The server returned an unexpected HTTP error. |
| 14 | `PROTOCOL_INCOMPATIBLE` | Protocol major versions differ or the protocol response is unusable. |
| 15 | `SERVER_NOT_READY` | The AgentCore server responded but is not ready. |
| 16 | `RUNTIME_NOT_READY` | The inference runtime or model is unavailable or unhealthy. |
| 20 | `TASK_FAILED` | The remote task failed. |
| 21 | `TASK_CANCELLED` | The remote task was cancelled, including `/abort`. |
| 22 | `PROPOSAL_REJECTED` | The user explicitly rejected the proposal. |
| 23 | `APPROVAL_REQUIRED` | A non-interactive invocation ended with required approval pending. |
| 24 | `STREAM_ERROR` | An HTTP/SSE stream was interrupted, malformed, or lacked a terminal result. |
| 70 | `INTERNAL_CLIENT_ERROR` | An unexpected client-side error occurred. |

An initial HTTP read timeout is classified as `NETWORK_UNREACHABLE`; a timeout
after SSE events have started is `STREAM_ERROR`. Classification is based on
the exception type and operating-system error code. Some platforms expose
less detail for transport failures; in that case the client returns the
nearest safe category without interpreting localized error messages.

Inspect the result from a direct VPN connection:

```bash
python3 -m agentclient.cli \
  --server http://10.121.0.10:8080 \
  --workspace /server/path

rc=$?
printf 'agentclient exit code: %s\n' "$rc"
```

The same pattern works through an SSH tunnel by using
`http://127.0.0.1:8080`. A shell automation example:

```bash
agentclient --server http://127.0.0.1:8080 --workspace /server/path
rc=$?

case "$rc" in
  0)  echo "AgentCore session completed" ;;
  10) echo "Network path or initial response unavailable" >&2 ;;
  11) echo "No listener on the configured host and port" >&2 ;;
  14) echo "Client/server protocol mismatch" >&2 ;;
  15|16) echo "Server or inference runtime is not ready" >&2 ;;
  20|21|22) echo "Task did not complete: agentclient status $rc" >&2 ;;
  23) echo "Explicit approval is required" >&2 ;;
  24) echo "SSE stream did not produce a valid terminal result" >&2 ;;
  *)  echo "AgentClient failed with status $rc" >&2 ;;
esac
```

Exit codes describe what the client observed. They do not prove that a VPN or
SSH tunnel is configured correctly. `--debug` adds a sanitized exception
traceback while preserving the same exit code.

## Security

The AgentCore HTTP server is still a development interface. It should bind to
localhost by default and should not be exposed on an untrusted network without
an authentication and authorization layer.
