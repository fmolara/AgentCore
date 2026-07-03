from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal AgentCore agent demo.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--system-prompt",
        default="You are a concise coding assistant.",
        help="System prompt for the demo agent.",
    )
    parser.add_argument("--prompt", default="Explain pointer arithmetic in C.", help="Prompt to ask the agent.")
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum generated tokens.")
    return parser.parse_args()


def log_path(lab: AgentLab) -> str | None:
    writer = getattr(lab.runtime, "log_writer", None)
    path = getattr(writer, "path", None)
    return str(path) if path else None


def main() -> None:
    args = parse_args()
    lab = AgentLab.from_config(args.config)

    lab.start()
    try:
        lab.warmup(max_tokens=16)
        agent = lab.create_agent(
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=0,
        )
        reply = agent.ask(args.prompt)

        print(reply.text.strip())
        print()
        print("agent statistics:")
        print(json.dumps(agent.statistics(), indent=2, sort_keys=True))
        print(f"JSONL log path: {log_path(lab)}")
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()
