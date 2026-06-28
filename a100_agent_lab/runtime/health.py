from __future__ import annotations

import subprocess
from typing import Any


def query_gpu(device: int | str | None = None) -> dict[str, Any]:
    index = _device_index(device)
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if index is not None:
        cmd.extend(["-i", str(index)])
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        name, used, total = [item.strip() for item in output.splitlines()[0].split(",")]
        return {
            "gpu_name": name,
            "gpu_memory_used_mib": int(used),
            "gpu_memory_total_mib": int(total),
        }
    except Exception:
        return {
            "gpu_name": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
        }


def normalized_health(
    *,
    runtime_name: str,
    backend_type: str,
    model_path: str | None,
    ready: bool,
    server_ready_time_sec: float | None = None,
    warmup_wall_sec: float | None = None,
    process_pid: int | None = None,
    endpoint: str | None = None,
    last_error: str | None = None,
    gpu: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    health = {
        "runtime_name": runtime_name,
        "backend_type": backend_type,
        "model_path": model_path,
        "ready": ready,
        "server_ready_time_sec": server_ready_time_sec,
        "warmup_wall_sec": warmup_wall_sec,
        "gpu_name": None,
        "gpu_memory_used_mib": None,
        "gpu_memory_total_mib": None,
        "process_pid": process_pid,
        "endpoint": endpoint,
        "last_error": last_error,
    }
    if gpu:
        health.update(
            {
                "gpu_name": gpu.get("gpu_name"),
                "gpu_memory_used_mib": gpu.get("gpu_memory_used_mib"),
                "gpu_memory_total_mib": gpu.get("gpu_memory_total_mib"),
            }
        )
    if extra:
        health.update(extra)
    return health


def _device_index(device: int | str | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    text = str(device)
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    try:
        return int(text)
    except ValueError:
        return None
