from __future__ import annotations

import pytest

from agentcore_server import AgentLab
from agentcore_server.sessions import Session, SessionStore
from agentcore_server.workspace import FileEditResult, Workspace


class FakeRuntime:
    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)


def test_file_workspace_read_write_append(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")

    result = workspace.files.write_text("notes.txt", "one\n")
    append = workspace.files.append_text("notes.txt", "two\n")

    assert isinstance(result, FileEditResult)
    assert result.operation == "write_text"
    assert result.path == "notes.txt"
    assert result.bytes_written == 4
    assert result.lines_written == 1
    assert append.operation == "append_text"
    assert workspace.files.read_text("notes.txt") == "one\ntwo\n"


def test_file_workspace_replace_success(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    workspace.files.write_text("main.c", "int value(void) { return 1; }\n")

    result = workspace.files.replace_text("main.c", "return 1", "return 2")

    assert result.operation == "replace_text"
    assert result.replacements == 1
    assert "return 2" in workspace.files.read_text("main.c")


def test_file_workspace_replace_missing_text(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    workspace.files.write_text("main.c", "int value(void) { return 1; }\n")

    with pytest.raises(ValueError, match="text not found"):
        workspace.files.replace_text("main.c", "return 7", "return 2")


def test_file_workspace_unique_replace_rejects_zero_and_multiple_matches(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    workspace.files.write_text("main.c", "one one\n")

    with pytest.raises(ValueError, match="not found"):
        workspace.files.replace_text_unique("main.c", "missing", "new")
    assert workspace.files.read_text("main.c") == "one one\n"

    with pytest.raises(ValueError, match="ambiguous"):
        workspace.files.replace_text_unique("main.c", "one", "new")
    assert workspace.files.read_text("main.c") == "one one\n"


def test_file_workspace_unique_replace_reports_local_context(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    workspace.files.write_text("main.c", "zero\none\ntwo\n")

    result = workspace.files.replace_text_unique("main.c", "one", "changed")

    assert result.match_count == 1
    assert result.affected_start_line == 2
    assert "changed" in (result.context or "")
    assert workspace.files.read_text("main.c") == "zero\nchanged\ntwo\n"


def test_file_workspace_read_line_ranges_and_write_lines(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    result = workspace.files.write_lines("notes.txt", ["zero", "one", "two", "three"])

    assert result.operation == "write_lines"
    assert result.lines_written == 4
    assert workspace.files.read_lines("notes.txt", start=1, end=3) == ["one", "two"]


def test_file_workspace_apply_simple_unified_diff(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")
    workspace.files.write_text(
        "main.c",
        "int value(void) {\n"
        "    return 1;\n"
        "}\n",
    )
    diff = """--- a/main.c
+++ b/main.c
@@ -1,3 +1,3 @@
 int value(void) {
-    return 1;
+    return 2;
 }
"""

    result = workspace.files.apply_unified_diff(diff)

    assert result.operation == "apply_unified_diff"
    assert result.files_changed == ("main.c",)
    assert result.lines_written == 2
    assert workspace.files.read_text("main.c") == "int value(void) {\n    return 2;\n}\n"


def test_file_workspace_rejects_path_traversal(tmp_path) -> None:
    workspace = Workspace(tmp_path / "project")

    with pytest.raises(ValueError):
        workspace.files.write_text("../outside.txt", "no")

    diff = """--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError):
        workspace.files.apply_unified_diff(diff)


def test_file_workspace_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)
    workspace = Workspace(root)

    with pytest.raises(ValueError):
        workspace.files.read_text("link/secret.txt")

    diff = """--- a/link/secret.txt
+++ b/link/secret.txt
@@ -1 +1 @@
-secret
+changed
"""
    with pytest.raises(ValueError):
        workspace.files.apply_unified_diff(diff)


def test_agent_files_shortcut_points_to_workspace_files(tmp_path) -> None:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(tmp_path / "workspace")}}
    lab.project_root = tmp_path
    lab.runtime = FakeRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)

    agent = lab.create_agent()

    assert agent.files is agent.workspace.files
    agent.files.write_text("note.txt", "hello\n")
    assert agent.workspace.files.read_text("note.txt") == "hello\n"
