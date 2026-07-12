from __future__ import annotations

import ast
import sys
import tomllib
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = ROOT / "packages" / "agentcore-server"


def test_agentcore_server_imports_without_agentclient() -> None:
    import agentcore_server

    assert agentcore_server.AgentLab.__name__ == "AgentLab"


def test_agentcore_server_depends_on_protocol_not_agentclient() -> None:
    pyproject = tomllib.loads((SERVER_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "agentcore-protocol>=1.0.0" in pyproject["project"]["dependencies"]
    assert all("agentclient" not in dependency for dependency in pyproject["project"]["dependencies"])


def test_agentcore_server_package_does_not_import_agentclient() -> None:
    for path in (SERVER_ROOT / "src" / "agentcore_server").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] != "agentclient", f"{path} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] != "agentclient", f"{path} imports {node.module}"


def test_deprecated_a100_agent_lab_import_warns() -> None:
    sys.modules.pop("a100_agent_lab", None)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        import a100_agent_lab
        from a100_agent_lab.sessions import Session

    assert a100_agent_lab.AgentLab.__name__ == "AgentLab"
    assert Session.__name__ == "Session"
    assert any("a100_agent_lab is deprecated" in str(item.message) for item in captured)
