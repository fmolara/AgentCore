from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import ActionPlan, AgentLab, PlanProposal, TaskExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate AgentCore plan proposals.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "plan-proposal-demo"),
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
        workspace_metadata={"purpose": "plan proposal demo"},
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
        agent.git.commit("Prepare plan proposal demo")

    task = agent.create_task(
        title="Update parser return value",
        description="Apply a reviewed edit proposal to parser.c.",
    )
    plan = ActionPlan.from_dict(
        {
            "title": "Edit parser",
            "description": "Replace parser return value.",
            "actions": [
                {"type": "replace_text", "path": "parser.c", "old": "return 0;", "new": "return 1;"},
                {"type": "git_diff"},
                {"type": "task_report"},
            ],
        }
    )
    proposal = PlanProposal.from_action_plan(
        task_id=task.id,
        action_plan=plan,
        summary="Change parse_token() to return a successful parse result.",
    )
    executor = TaskExecutor(agent)

    print("approval requirements:")
    print(json.dumps([requirement.as_dict() for requirement in proposal.approval_requirements], indent=2))
    print()

    refused = proposal.execute(executor, task)
    print("attempt without approval:")
    print(json.dumps(refused.as_dict(), indent=2, sort_keys=True))
    print()

    proposal.approve()
    executed = proposal.execute(executor, task)
    print("after approval:")
    print(json.dumps(executed.as_dict(), indent=2, sort_keys=True))
    print()

    print("proposal:")
    print(json.dumps(proposal.as_dict(), indent=2, sort_keys=True))
    print()

    print("task report:")
    print(json.dumps(task.report().as_dict(), indent=2, sort_keys=True))
    print()

    print("git diff:")
    print(agent.git.diff().stdout.rstrip())


if __name__ == "__main__":
    main()
