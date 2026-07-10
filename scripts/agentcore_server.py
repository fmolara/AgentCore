from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn

from a100_agent_lab.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AgentCore HTTP API server.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default is localhost.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument("--workspace-root", default=None, help="Default root for server-created workspaces.")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", action="store_true", default=True, help="Warm up runtime on server startup.")
    warmup.add_argument("--no-warmup", action="store_false", dest="warmup", help="Skip runtime warmup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        config_path=args.config,
        workspace_root=args.workspace_root,
        warmup=args.warmup,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
