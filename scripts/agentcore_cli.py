from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agentclient" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "agentcore-protocol" / "src"))

from agentclient.cli import main


if __name__ == "__main__":
    print(
        "scripts/agentcore_cli.py is deprecated; use the remote `agentclient` command instead.",
        file=sys.stderr,
    )
    raise SystemExit(main(sys.argv[1:]))
