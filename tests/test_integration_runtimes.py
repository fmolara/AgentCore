from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentcore_server import AgentLab


pytestmark = pytest.mark.integration


RUN_INTEGRATION = os.environ.get("A100_AGENT_LAB_RUN_INTEGRATION") == "1"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set A100_AGENT_LAB_RUN_INTEGRATION=1 to run")
@pytest.mark.parametrize(
    "config_path",
    [
        ROOT / "config" / "transformers-a100.yaml",
        ROOT / "config" / "sglang-a100.yaml",
        ROOT / "config" / "lmdeploy-a100.yaml",
    ],
)
def test_runtime_integration_smoke(config_path: Path) -> None:
    lab = AgentLab.from_config(config_path)

    lab.start()
    try:
        assert lab.ready()
        health = lab.health()
        assert health["ready"] is True
        assert health["runtime_name"] in {"transformers", "sglang", "lmdeploy"}
        assert health["model_path"]

        warmup = lab.warmup(max_tokens=4)
        assert warmup.metrics.generated_tokens >= 0

        session = lab.create_session(system_prompt="You are concise.")
        result = lab.generate(session, "Say OK.", max_tokens=4)
        assert result.text
        assert result.metrics.prompt_tokens > 0
        assert result.metrics.wall_sec >= 0
    finally:
        lab.shutdown()
