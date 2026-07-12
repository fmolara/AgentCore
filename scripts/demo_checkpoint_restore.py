from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agentcore-server" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "agentcore-protocol" / "src"))
sys.path.insert(0, str(ROOT))

from agentcore_server import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore checkpoint restore.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "checkpoint-restore-demo"),
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
        workspace_metadata={"purpose": "checkpoint restore demo"},
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
        agent.git.commit("Prepare checkpoint restore demo")

    task = agent.create_task(
        title="Restore parser checkpoint",
        description="Restore parser.c to a prior explicit checkpoint.",
    )
    task.start()

    agent.files.replace_text("parser.c", "return 0;", "return 1;")
    checkpoint = task.create_checkpoint("return one", "Parser returns one token.")

    agent.files.replace_text("parser.c", "return 1;", "return 2;")
    result = task.restore_checkpoint(checkpoint, force=True)

    print("restore result:")
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    print()
    print("parser.c:")
    print(agent.files.read_text("parser.c").rstrip())
    print()
    print("git diff:")
    print(agent.git.diff().stdout.rstrip())


if __name__ == "__main__":
    main()
