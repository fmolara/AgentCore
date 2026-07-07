from __future__ import annotations

import pytest

from a100_agent_lab import AgentLab
from a100_agent_lab.sessions import Session, SessionStore
from a100_agent_lab.workspace import GitResult, Workspace


class FakeRuntime:
    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)


def set_test_git_identity(monkeypatch) -> None:
    monkeypatch.setenv("GIT_AUTHOR_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "agentcore-test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "AgentCore Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "agentcore-test@example.invalid")


def test_git_init_creates_repo_and_reports_branch(tmp_path) -> None:
    workspace = Workspace(tmp_path / "repo")

    assert workspace.git.is_repo() is False
    result = workspace.git.init()

    assert isinstance(result, GitResult)
    assert result.ok
    assert workspace.git.is_repo() is True
    assert workspace.git.current_branch() is not None


def test_git_status_add_commit_log_and_diff(tmp_path, monkeypatch) -> None:
    set_test_git_identity(monkeypatch)
    workspace = Workspace(tmp_path / "repo")
    workspace.git.init()

    workspace.write_text("file.txt", "one\n")
    status = workspace.git.status()

    assert "?? file.txt" in status.stdout

    add = workspace.git.add(["file.txt"])
    assert add.ok

    commit = workspace.git.commit("Initial commit")
    assert commit.ok

    log = workspace.git.log(limit=1)
    assert "Initial commit" in log.stdout

    workspace.write_text("file.txt", "one\ntwo\n")
    diff = workspace.git.diff()

    assert diff.ok
    assert "+two" in diff.stdout


def test_git_add_validates_paths_and_blocks_traversal(tmp_path) -> None:
    workspace = Workspace(tmp_path / "repo")
    workspace.git.init()
    workspace.write_text("safe.txt", "safe\n")

    assert workspace.git.add("safe.txt").ok

    with pytest.raises(ValueError):
        workspace.git.add("../outside.txt")

    with pytest.raises(ValueError):
        workspace.git.add(tmp_path / "outside.txt")


def test_git_commit_requires_message(tmp_path) -> None:
    workspace = Workspace(tmp_path / "repo")
    workspace.git.init()

    with pytest.raises(ValueError):
        workspace.git.commit("   ")


def test_git_read_only_workspace_blocks_mutations(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace = Workspace(root, mode=Workspace.READ_ONLY)

    with pytest.raises(PermissionError):
        workspace.git.init()

    with pytest.raises(PermissionError):
        workspace.git.add(["file.txt"])

    with pytest.raises(PermissionError):
        workspace.git.commit("message")


def test_git_network_operations_are_not_exposed(tmp_path) -> None:
    workspace = Workspace(tmp_path / "repo")

    for name in ("clone", "fetch", "pull", "push"):
        assert not hasattr(workspace.git, name)


def test_agent_git_shortcut_points_to_workspace_git(tmp_path) -> None:
    lab = AgentLab.__new__(AgentLab)
    lab.config = {"runtime": "fake", "workspace": {"root": str(tmp_path / "workspace")}}
    lab.project_root = tmp_path
    lab.runtime = FakeRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)

    agent = lab.create_agent()

    assert agent.git is agent.workspace.git
    assert agent.git.init().ok
