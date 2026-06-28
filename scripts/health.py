from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print normalized AgentLab runtime health.")
    parser.add_argument("--config", required=True, help="AgentLab YAML configuration path.")
    parser.add_argument("--warmup", action="store_true", help="Run a short warmup before printing statistics.")
    parser.add_argument("--max-tokens", type=int, default=16, help="Warmup token limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lab = AgentLab.from_config(args.config)
    warmup_metrics = None

    lab.start()
    try:
        if args.warmup:
            warmup = lab.warmup(max_tokens=args.max_tokens)
            warmup_metrics = warmup.metrics.as_dict()
        print(
            json.dumps(
                {
                    "health": lab.health(),
                    "statistics": lab.statistics(),
                    "warmup": warmup_metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()
