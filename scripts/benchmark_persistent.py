from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "agentcore-server" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "agentcore-protocol" / "src"))
sys.path.insert(0, str(ROOT))

from agentcore_server import AgentLab


SYSTEM_PROMPT = "You are a concise coding agent."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a persistent multi-turn AgentLab benchmark.")
    parser.add_argument("--config", required=True, help="AgentLab YAML configuration path.")
    parser.add_argument("--prompts", required=True, help="Text file with one prompt per non-empty line.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum generated tokens per turn.")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", dest="warmup", action="store_true", help="Run explicit warmup first.")
    warmup.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip explicit warmup.")
    parser.set_defaults(warmup=True)
    return parser.parse_args()


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def log_path(lab: AgentLab) -> str | None:
    writer = getattr(lab.runtime, "log_writer", None)
    path = getattr(writer, "path", None)
    return str(path) if path else None


def format_memory(used: int | None, total: int | None) -> str:
    if used is None:
        return "-"
    if total is None:
        return f"{used} MiB"
    return f"{used}/{total} MiB"


def print_runtime_health(health: dict) -> None:
    print("runtime health")
    print(f"runtime: {health.get('runtime_name')} ({health.get('backend_type')})")
    print(f"model: {health.get('model_path')}")
    print(f"ready: {health.get('ready')}")
    print(
        "GPU: "
        f"{health.get('gpu_name') or '-'} "
        f"{format_memory(health.get('gpu_memory_used_mib'), health.get('gpu_memory_total_mib'))}"
    )
    if health.get("endpoint"):
        print(f"endpoint: {health.get('endpoint')}")
    if health.get("process_pid"):
        print(f"PID: {health.get('process_pid')}")
    if health.get("last_error"):
        print(f"last error: {health.get('last_error')}")


def print_table(rows: list[dict]) -> None:
    print()
    print("turn  prompt_tok  gen_tok  ttft_s  tok/s   wall_s")
    print("----  ----------  -------  ------  ------  ------")
    for row in rows:
        print(
            f"{row['turn']:>4}  "
            f"{row['prompt_tokens']:>10}  "
            f"{row['generated_tokens']:>7}  "
            f"{format_float(row['ttft_sec']):>6}  "
            f"{format_float(row['tokens_per_sec']):>6}  "
            f"{format_float(row['wall_sec']):>6}"
        )


def main() -> None:
    args = parse_args()
    prompts = read_prompts(Path(args.prompts))
    lab = AgentLab.from_config(args.config)

    lab.start()
    try:
        print_runtime_health(lab.health())
        if args.warmup:
            warmup = lab.warmup(max_tokens=16)
            print(f"warmup: {warmup.metrics.as_dict()}")

        session = lab.create_session(system_prompt=SYSTEM_PROMPT)
        rows: list[dict] = []
        started = time.perf_counter()
        for index, prompt in enumerate(prompts, start=1):
            result = lab.generate(session, prompt, max_tokens=args.max_tokens)
            metrics = result.metrics.as_dict()
            rows.append({"turn": index, **metrics})
        total_wall = time.perf_counter() - started

        print_table(rows)

        ttfts = [row["ttft_sec"] for row in rows if row["ttft_sec"] is not None]
        tps = [row["tokens_per_sec"] for row in rows if row["tokens_per_sec"] is not None]
        tps_excluding_turn1 = [
            row["tokens_per_sec"] for row in rows[1:] if row["tokens_per_sec"] is not None
        ]
        print()
        print("summary")
        print(f"total wall time: {total_wall:.3f} s")
        print(f"average TTFT: {statistics.mean(ttfts):.3f} s")
        print(f"average tokens/sec: {statistics.mean(tps):.2f}")
        print(f"average tokens/sec excluding turn 1: {statistics.mean(tps_excluding_turn1):.2f}")
        print()
        print_runtime_health(lab.statistics())
        print(f"log path: {log_path(lab)}")
    finally:
        lab.shutdown()


if __name__ == "__main__":
    main()
