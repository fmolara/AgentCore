from __future__ import annotations

import pytest

from a100_agent_lab.workspace import Workspace


def test_workspace_read_write_list_exists_and_mkdir(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project", metadata={"name": "test"})

    workspace.mkdir("src")
    written = workspace.write_text("src/main.c", "int main(void) { return 0; }\n")

    assert written == len("int main(void) { return 0; }\n")
    assert workspace.exists("src/main.c")
    assert workspace.read_text("src/main.c") == "int main(void) { return 0; }\n"
    assert workspace.list(".") == ["src"]
    assert workspace.list("src") == ["main.c"]
    assert workspace.as_dict()["metadata"] == {"name": "test"}


def test_workspace_can_use_relative_cwd(tmp_path) -> None:
    root = tmp_path / "project"
    workspace = Workspace(root, cwd="pkg")

    workspace.write_text("module.py", "VALUE = 1\n")

    assert workspace.exists("module.py")
    assert (root / "pkg" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert workspace.as_dict()["cwd"] == "pkg"


def test_workspace_rejects_path_traversal(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")

    with pytest.raises(ValueError):
        workspace.read_text("../outside.txt")

    with pytest.raises(ValueError):
        workspace.write_text(tmp_path / "outside.txt", "no")

    with pytest.raises(ValueError):
        workspace.exists("../outside.txt")


def test_workspace_read_only_blocks_mutations(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")
    workspace = Workspace(root, mode=Workspace.READ_ONLY)

    assert workspace.read_text("note.txt") == "hello"

    with pytest.raises(PermissionError):
        workspace.write_text("note.txt", "changed")

    with pytest.raises(PermissionError):
        workspace.mkdir("new")


def test_workspace_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)
    workspace = Workspace(root)

    with pytest.raises(ValueError):
        workspace.read_text("link/secret.txt")
