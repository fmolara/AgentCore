from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agentcore-server" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "agentcore-protocol" / "src"))
sys.path.insert(0, str(ROOT))

from agentcore_server import AgentLab


SYSTEM_A = "You are session A, a concise C systems tutor."
SYSTEM_B = "You are session B, a concise Python tooling tutor."

TURNS = [
    ("A1", "A", "In one paragraph, explain malloc in C."),
    ("B1", "B", "In one paragraph, explain pytest fixtures."),
    ("A2", "A", "Now explain why free must match malloc."),
    ("B2", "B", "Now explain why parametrized tests are useful."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-session AgentLab smoke test.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum generated tokens per turn.")
    return parser.parse_args()


def configured_log_dir(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    log_dir = Path(config.get("logging", {}).get("path", "experiments/logs"))
    if not log_dir.is_absolute():
        log_dir = config_path.parent.parent / log_dir
    return log_dir


def generation_logs(log_dir: Path) -> set[Path]:
    if not log_dir.exists():
        return set()
    return set(log_dir.glob("generation-*.jsonl"))


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def verify_isolated_transcripts(session_a, session_b) -> None:
    transcript_a = session_a.transcript()
    transcript_b = session_b.transcript()
    text_a = "\n".join(message["content"] for message in transcript_a)
    text_b = "\n".join(message["content"] for message in transcript_b)

    assert "malloc" in text_a
    assert "free must match malloc" in text_a
    assert "pytest fixtures" not in text_a
    assert "parametrized tests" not in text_a

    assert "pytest fixtures" in text_b
    assert "parametrized tests" in text_b
    assert "malloc" not in text_b
    assert "free must match malloc" not in text_b

    assert session_a.turn_count == 2
    assert session_b.turn_count == 2


def print_summary(rows: list[dict]) -> None:
    print()
    print("turn  session  ttft_s  tok/s   wall_s")
    print("----  -------  ------  ------  ------")
    for row in rows:
        print(
            f"{row['turn']:<4}  "
            f"{row['session']:^7}  "
            f"{format_float(row['ttft_sec']):>6}  "
            f"{format_float(row['tokens_per_sec']):>6}  "
            f"{format_float(row['wall_sec']):>6}"
        )

    ttfts = [row["ttft_sec"] for row in rows if row["ttft_sec"] is not None]
    tps = [row["tokens_per_sec"] for row in rows if row["tokens_per_sec"] is not None]
    walls = [row["wall_sec"] for row in rows if row["wall_sec"] is not None]
    print()
    print("summary")
    print(f"average TTFT: {statistics.mean(ttfts):.3f} s")
    print(f"average tokens/sec: {statistics.mean(tps):.2f}")
    print(f"total generation wall: {sum(walls):.3f} s")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    log_dir = configured_log_dir(config_path)
    logs_before = generation_logs(log_dir)

    lab = AgentLab.from_config(config_path)
    lab.start()
    try:
        warmup = lab.warmup(max_tokens=16)
        print(f"warmup: {warmup.metrics.as_dict()}")

        session_a = lab.create_session(system_prompt=SYSTEM_A)
        session_b = lab.create_session(system_prompt=SYSTEM_B)
        sessions = {"A": session_a, "B": session_b}
        rows: list[dict] = []

        for turn_name, session_name, prompt in TURNS:
            result = lab.generate(sessions[session_name], prompt, max_tokens=args.max_tokens)
            rows.append(
                {
                    "turn": turn_name,
                    "session": session_name,
                    **result.metrics.as_dict(),
                }
            )

        verify_isolated_transcripts(session_a, session_b)
        print_summary(rows)
        print()
        print(f"session A turn_count: {session_a.turn_count}")
        print(f"session B turn_count: {session_b.turn_count}")
        print("transcripts isolated: yes")
    finally:
        lab.shutdown()

    new_logs = sorted(generation_logs(log_dir) - logs_before)
    print(f"JSONL log path: {new_logs[-1] if new_logs else '-'}")


if __name__ == "__main__":
    main()
