from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import ActionPlan, AgentLab, TaskExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore serializable action plans.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "action-plan-demo"),
        help="Workspace root for the demo.",
    )
    parser.add_argument(
        "--plan",
        default=str(ROOT / "examples" / "action_plans" / "simple_edit.json"),
        help="ActionPlan JSON/YAML path.",
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
        workspace_metadata={"purpose": "action plan demo"},
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
        agent.git.commit("Prepare action plan demo")

    plan = ActionPlan.load(args.plan)
    task = agent.create_task(title=plan.title, description=plan.description, metadata={"plan_id": plan.id})
    result = TaskExecutor(agent).execute_plan(task, plan)

    print("action timeline:")
    print(json.dumps([action.as_dict() for action in result.actions], indent=2, sort_keys=True))
    print()
    print("task report:")
    print(json.dumps(result.report.as_dict() if result.report else None, indent=2, sort_keys=True))
    print()
    print("git diff:")
    print(agent.git.diff().stdout.rstrip())


if __name__ == "__main__":
    main()
