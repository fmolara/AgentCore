from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore task reporting.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "task-report-demo"),
        help="Workspace root for the demo.",
    )
    return parser.parse_args()


def ensure_demo_git_identity() -> None:
    os.environ.setdefault("GIT_AUTHOR_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "agentcore-demo@example.invalid")
    os.environ.setdefault("GIT_COMMITTER_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "agentcore-demo@example.invalid")


def main() -> None:
    args = parse_args()
    ensure_demo_git_identity()

    lab = AgentLab.from_config(args.config)
    agent = lab.create_agent(
        system_prompt="You are a concise coding assistant.",
        workspace_root=args.workspace,
        workspace_metadata={"purpose": "task report demo"},
    )

    if not agent.git.is_repo():
        agent.git.init()

    baseline = (
        "int parse_token(const char *input) {\n"
        "    (void)input;\n"
        "    return 0;\n"
        "}\n"
    )
    agent.files.write_text("parser.c", baseline)
    if agent.git.status().stdout.strip():
        agent.git.add(["parser.c"])
        agent.git.commit("Prepare parser task report demo")

    task = agent.create_task(
        title="Refactor parser",
        description="Replace placeholder return value with a token count.",
    )
    task.start()
    agent.files.replace_text("parser.c", "return 0;", "return 1;")

    print(json.dumps(task.report().as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
