from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab.workspace import Workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate the AgentCore HTTP/SSE API.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentCore runtime config.")
    parser.add_argument("--host", default="127.0.0.1", help="AgentCore server host.")
    parser.add_argument("--port", type=int, default=8080, help="AgentCore server port.")
    parser.add_argument("--server-url", default=None, help="Use an already running AgentCore server.")
    parser.add_argument("--no-start-server", action="store_true", help="Do not start a local server subprocess.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip server warmup when starting a subprocess.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.server_url or f"http://{args.host}:{args.port}"
    server: subprocess.Popen | None = None
    log_file = None

    with tempfile.TemporaryDirectory(prefix="agentcore-http-demo-") as tmp:
        workspace_root = Path(tmp) / "workspace"
        prepare_workspace(workspace_root)

        try:
            if not args.no_start_server:
                log_path = Path(tmp) / "agentcore-server.log"
                log_file = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "agentcore_server.py"),
                    "--config",
                    args.config,
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ]
                if args.no_warmup:
                    command.append("--no-warmup")
                server = subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
                wait_for_health(base_url)

            agent = post_json(
                base_url,
                "/v1/agents",
                {
                    "system_prompt": "You are a concise coding assistant.",
                    "workspace_root": str(workspace_root),
                },
            )
            agent_id = agent["id"]
            task = post_json(
                base_url,
                f"/v1/agents/{agent_id}/tasks",
                {
                    "title": "Replace parser return value",
                    "description": "Replace return 0 with return 1 in parser.c.",
                },
            )["task"]
            task_id = task["id"]

            proposal_response = post_json(
                base_url,
                f"/v1/tasks/{task_id}/proposals",
                {
                    "instruction": "Replace return 0 with return 1 in parser.c.",
                    "max_tokens": 512,
                    "temperature": 0,
                },
            )
            proposal = proposal_response["proposal"]
            proposal_id = proposal["id"]

            print("Proposed plan:")
            print(json.dumps(proposal["action_plan"], indent=2, sort_keys=True))
            print("\nApproval requirements:")
            print(json.dumps(proposal["approval_requirements"], indent=2, sort_keys=True))

            events: list[dict[str, Any]] = []
            sse_thread = threading.Thread(
                target=collect_sse,
                args=(base_url, task_id, events),
                daemon=True,
            )
            sse_thread.start()
            time.sleep(0.2)

            post_json(base_url, f"/v1/proposals/{proposal_id}/approve", {})
            execution = post_json(base_url, f"/v1/proposals/{proposal_id}/execute", {})
            sse_thread.join(timeout=20)

            report = get_json(base_url, f"/v1/tasks/{task_id}/report")["report"]
            diff = get_json(base_url, f"/v1/agents/{agent_id}/git/diff")["stdout"]
            parser_text = (workspace_root / "parser.c").read_text(encoding="utf-8")
            log_lines = [line for line in Workspace(workspace_root).git.log(limit=10).stdout.splitlines() if line]

            if "return 1;" not in parser_text:
                raise RuntimeError("parser.c was not modified as expected")
            if len(log_lines) != 1:
                raise RuntimeError("demo expected no automatic Git commit")

            print("\nExecution status:", execution["status"])
            print("\nSSE trace:")
            for event in events:
                print(f"- {event['event_type']}: {event['summary']}")
            print("\nTask report status:", report["status"])
            print("\nGit diff:")
            print(diff.rstrip())
            print("\nVerified: parser.c contains return 1 and no Git commit was created automatically.")
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            if log_file is not None:
                log_file.close()


def prepare_workspace(root: Path) -> None:
    workspace = Workspace(root)
    workspace.git.init()
    subprocess.run(["git", "config", "user.name", "AgentCore Demo"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "agentcore-demo@example.invalid"], cwd=root, check=True)
    workspace.files.write_text("parser.c", "int parse(void) {\n    return 0;\n}\n")
    workspace.git.add(["parser.c"])
    result = workspace.git.commit("Initial parser")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "initial Git commit failed")


def wait_for_health(base_url: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = get_json(base_url, "/health")
            if health.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - demo diagnostics should preserve the last error.
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def get_json(base_url: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(base_url + path, method="GET")
    return _json_response(request)


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return _json_response(request)


def _json_response(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def collect_sse(base_url: str, task_id: str, events: list[dict[str, Any]]) -> None:
    request = urllib.request.Request(base_url + f"/v1/tasks/{task_id}/events", method="GET")
    current_data: list[str] = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\n")
            if not line:
                if current_data:
                    events.append(json.loads("\n".join(current_data)))
                    current_data.clear()
                continue
            if line.startswith("data: "):
                current_data.append(line[6:])


if __name__ == "__main__":
    main()
