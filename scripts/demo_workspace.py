from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agentcore-server" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "agentcore-protocol" / "src"))
sys.path.insert(0, str(ROOT))

from agentcore_server import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore workspace operations.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "demo"),
        help="Workspace root for the demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lab = AgentLab.from_config(args.config)
    agent = lab.create_agent(
        system_prompt="You are a concise coding assistant.",
        workspace_root=args.workspace,
        workspace_metadata={"purpose": "workspace demo"},
    )

    agent.workspace.mkdir("notes")
    agent.workspace.write_text(
        "notes/pointer_arithmetic.txt",
        "Pointer arithmetic advances by sizeof(*ptr), not by one byte.\n",
    )

    print("workspace:")
    print(agent.workspace.as_dict())
    print()
    print("root entries:")
    print(agent.workspace.list("."))
    print()
    print("notes entries:")
    print(agent.workspace.list("notes"))
    print()
    print("read back:")
    print(agent.workspace.read_text("notes/pointer_arithmetic.txt").strip())
    print()
    print(f"exists: {agent.workspace.exists('notes/pointer_arithmetic.txt')}")

    try:
        agent.workspace.read_text("../outside.txt")
    except ValueError:
        print("path traversal blocked: yes")


if __name__ == "__main__":
    main()
