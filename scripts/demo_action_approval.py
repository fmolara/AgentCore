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

from agentcore_server import ActionPlan, AgentLab, TaskExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore action approval policy.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "action-approval-demo"),
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
        workspace_metadata={"purpose": "action approval demo"},
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
        agent.git.commit("Prepare action approval demo")

    readonly_plan = ActionPlan.from_dict(
        {
            "title": "Inspect parser",
            "actions": [
                {"type": "read_file", "path": "parser.c"},
                {"type": "git_status"},
                {"type": "git_diff"},
                {"type": "task_report"},
            ],
        }
    )
    mutating_plan = ActionPlan.from_dict(
        {
            "title": "Edit parser",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
                {"type": "git_diff"},
            ],
        }
    )

    executor = TaskExecutor(agent)

    readonly_task = agent.create_task(title=readonly_plan.title)
    readonly_result = executor.execute_plan(readonly_task, readonly_plan)

    refused_task = agent.create_task(title=mutating_plan.title)
    refused_result = executor.execute_plan(refused_task, mutating_plan)

    approved_task = agent.create_task(title=mutating_plan.title)
    approved_result = executor.execute_plan(approved_task, mutating_plan, approved=True)

    print("readonly result:")
    print(json.dumps(readonly_result.as_dict(), indent=2, sort_keys=True))
    print()
    print("mutating without approval:")
    print(json.dumps(refused_result.as_dict(), indent=2, sort_keys=True))
    print()
    print("mutating with approval:")
    print(json.dumps(approved_result.as_dict(), indent=2, sort_keys=True))
    print()
    print("git diff:")
    print(agent.git.diff().stdout.rstrip())


if __name__ == "__main__":
    main()
