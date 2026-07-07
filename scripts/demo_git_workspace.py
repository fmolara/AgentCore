from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate local Git operations in an AgentCore workspace.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "git-demo"),
        help="Workspace root for the demo.",
    )
    return parser.parse_args()


def ensure_demo_git_identity() -> None:
    os.environ.setdefault("GIT_AUTHOR_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "agentcore-demo@example.invalid")
    os.environ.setdefault("GIT_COMMITTER_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "agentcore-demo@example.invalid")


def print_result(title: str, text: str) -> None:
    print(title)
    print(text.strip() or "(empty)")
    print()


def main() -> None:
    args = parse_args()
    ensure_demo_git_identity()

    lab = AgentLab.from_config(args.config)
    agent = lab.create_agent(
        system_prompt="You are a concise coding assistant.",
        workspace_root=args.workspace,
        workspace_metadata={"purpose": "git workspace demo"},
    )

    if not agent.git.is_repo():
        init = agent.git.init()
        print_result("git init:", init.stdout + init.stderr)

    agent.workspace.mkdir("notes")
    stamp = datetime.now(timezone.utc).isoformat()
    agent.workspace.write_text(
        "notes/demo.txt",
        f"AgentCore Git workspace demo\nupdated_at={stamp}\n",
    )

    print_result("status after write:", agent.git.status().stdout)

    agent.git.add(["notes/demo.txt"])
    commit = agent.git.commit("Update Git workspace demo")
    print_result("commit:", commit.stdout + commit.stderr)

    agent.workspace.write_text(
        "notes/demo.txt",
        f"AgentCore Git workspace demo\nupdated_at={stamp}\nworking_tree_change=yes\n",
    )

    print_result("diff after modification:", agent.git.diff().stdout)
    print_result("recent log:", agent.git.log(limit=5).stdout)


if __name__ == "__main__":
    main()
