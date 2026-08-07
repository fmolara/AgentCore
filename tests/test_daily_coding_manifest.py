from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "daily-coding" / "manifest.yaml"


def test_daily_coding_manifest_is_complete_and_diverse():
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    tasks = data["tasks"]

    assert data["schema_version"] == 1
    assert len(tasks) == 10
    assert len({task["id"] for task in tasks}) == 10
    assert set(data["qwen_baseline_subset"]) == {"C01", "C02", "C04", "C05", "C08"}
    assert sum(len(task["primary_files"]) > 1 for task in tasks) >= 6
    assert sum(task["initial_test_status"] == "fail" for task in tasks) >= 2
    assert any(task["language"] == "Python" for task in tasks)
    assert any(task["category"] == "cmake-build-system" for task in tasks)
    assert any(task["source"]["kind"] == "historical-public" for task in tasks)

    for task in tasks:
        assert len(task["baseline"]) == 40
        assert task["prompt"].count("configured build") == 1
        assert "test check" in task["prompt"]
        assert "Git diff" in task["prompt"]
        assert task["checks"]["build"]
        assert task["checks"]["test"]
        assert task["acceptance"]
