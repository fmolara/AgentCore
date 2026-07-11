from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_http_demo_uses_protocol_client_for_http_operations() -> None:
    path = ROOT / "scripts" / "demo_http_agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    forbidden_prefixes = (
        "a100_agent_lab.server",
        "a100_agent_lab.api",
        "a100_agent_lab.executor",
        "a100_agent_lab.runtime",
        "urllib",
    )

    assert "AgentCoreClient" in imported_names
    assert any(module == "agentcore_protocol" for module in imported_modules)
    assert not any(module == prefix or module.startswith(prefix + ".") for module in imported_modules for prefix in forbidden_prefixes)
