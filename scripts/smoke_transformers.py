from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def main() -> None:
    lab = AgentLab.from_config(ROOT / "config" / "transformers-a100.yaml")
    lab.start()
    try:
        session = lab.create_session(system_prompt="You are a concise coding agent.")
        result = lab.generate(session, "Explain pointers in C.", max_tokens=64)
        print(result.text.strip())
        print(json.dumps(result.metrics.as_dict(), indent=2))
        print(json.dumps(lab.health(), indent=2))
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()

