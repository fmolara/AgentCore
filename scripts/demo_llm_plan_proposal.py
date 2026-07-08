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
    parser = argparse.ArgumentParser(description="Demonstrate LLM-assisted ActionPlan proposal generation.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "llm-plan-proposal-demo"),
        help="Workspace root for the demo.",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum tokens for the planning response.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip runtime warmup.")
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
    lab.start()
    try:
        if not args.no_warmup:
            lab.warmup(max_tokens=8)

        agent = lab.create_agent(
            system_prompt="You are a concise coding assistant.",
            workspace_root=args.workspace,
            workspace_metadata={"purpose": "llm plan proposal demo"},
        )
        if not agent.git.is_repo():
            agent.git.init()

        agent.files.write_text(
            "parser.c",
            "int parse_token(const char *input) {\n"
            "    (void)input;\n"
            "    return 0;\n"
            "}\n",
        )
        task = agent.create_task(
            title="Update parser return value",
            description="Ask the model to propose a safe edit plan.",
        )

        result = agent.propose_plan(
            task,
            instruction="Replace return 0 with return 1 in parser.c. Do not execute anything.",
            max_tokens=args.max_tokens,
            temperature=0,
        )

        print("planner result:")
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        print()
        if result.proposal is not None:
            print("approval requirements:")
            print(json.dumps([item.as_dict() for item in result.proposal.approval_requirements], indent=2))
            print()
            print("proposal JSON:")
            print(json.dumps(result.proposal.as_dict(), indent=2, sort_keys=True))
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()
