from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from agentcore_server import ActionPlan, AgentLab, RunCheckAction, TaskExecutor
from agentcore_server.runtime.base import Runtime
from agentcore_server.sessions import Session, SessionStore
from agentcore_server.workspace import Workspace


class NoopRuntime(Runtime):
    def load(self):
        pass

    def shutdown(self):
        pass

    def ready(self):
        return True

    def health(self):
        return {"ready": True}

    statistics = health

    def warmup(self, prompt=None, max_tokens=16):
        raise NotImplementedError

    def create_session(self, *, system_prompt=None):
        return Session(system_prompt=system_prompt)

    def generate(self, session, prompt, **kwargs):
        raise NotImplementedError

    def stream(self, session, prompt, **kwargs):
        raise NotImplementedError

    def tokenize(self, text_or_messages):
        return 0


def _agent(
    tmp_path: Path,
    checks: dict,
    *,
    workspace_mode: str = Workspace.READ_WRITE,
):
    lab = AgentLab.__new__(AgentLab)
    lab.config = {
        "runtime": "fake",
        "workspace": {"root": str(tmp_path), "checks": checks},
    }
    lab.project_root = tmp_path
    lab.runtime = NoopRuntime()
    lab.sessions = SessionStore(lab.runtime.create_session)
    return lab.create_agent(
        workspace_root=tmp_path,
        workspace_mode=workspace_mode,
    )


def test_run_check_executes_configured_argv_and_requires_approval(tmp_path) -> None:
    agent = _agent(
        tmp_path,
        {
            "test": {
                "argv": [sys.executable, "-c", "print('passed')"],
                "timeout_sec": 2,
            }
        },
    )
    task = agent.create_task(title="Check", description="Run tests")
    plan = ActionPlan.from_dict(
        {"title": "Check", "actions": [{"type": "run_check", "check": "test"}]}
    )

    requirements = plan.required_approvals()
    assert [item.action_type for item in requirements] == ["run_check"]
    refused = TaskExecutor(agent).execute_plan(task, plan)
    assert refused.status == "approval_required"

    completed = TaskExecutor(agent).execute_plan(task, plan, approved=True)
    assert completed.status == "completed"
    check = completed.actions[0].data["check"]
    assert check["status"] == "exited"
    assert check["returncode"] == 0
    assert check["stdout"] == "passed\n"
    assert check["timed_out"] is False


def test_run_check_rejects_unknown_symbolic_name(tmp_path) -> None:
    agent = _agent(tmp_path, {"test": {"argv": ["true"]}})
    task = agent.create_task(title="Check", description="Run bad check")

    result = TaskExecutor(agent).execute(
        task,
        [RunCheckAction("model-supplied-command")],
    )

    assert result.status == "failed"
    assert result.actions[0].error == "unknown configured check: model-supplied-command"


def test_run_check_rejects_read_only_workspace(tmp_path) -> None:
    agent = _agent(
        tmp_path,
        {"test": {"argv": [sys.executable, "-c", "print('not run')"]}},
        workspace_mode=Workspace.READ_ONLY,
    )
    task = agent.create_task(title="Check", description="Run in read-only workspace")

    result = TaskExecutor(agent).execute(
        task,
        [RunCheckAction("test")],
    )

    assert result.status == "failed"
    assert result.actions[0].error == "run_check is not allowed in a read-only workspace"


def test_run_check_never_uses_shell(tmp_path, monkeypatch) -> None:
    agent = _agent(tmp_path, {"test": {"argv": ["true"]}})
    observed = {}
    original = subprocess.Popen

    def spy(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    result = agent.workspace.checks.run("test")

    assert result.ok
    assert observed["shell"] is False
    assert observed["cwd"] == tmp_path
    assert observed["stdin"] == subprocess.DEVNULL


def test_run_check_timeout_and_output_limits_are_structured(tmp_path) -> None:
    agent = _agent(
        tmp_path,
        {
            "timeout": {
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; print('x' * 100, flush=True); time.sleep(5)",
                ],
                "timeout_sec": 0.1,
                "max_output_bytes": 10,
            }
        },
    )
    task = agent.create_task(title="Timeout", description="Timeout")

    result = TaskExecutor(agent).execute(task, [RunCheckAction("timeout")])

    assert result.status == "failed"
    check = result.actions[0].data["check"]
    assert check["status"] == "timeout"
    assert check["timed_out"] is True
    assert check["stdout_truncated"] is True
    assert len(check["stdout"].encode()) <= 10


def test_run_check_launch_and_nonzero_fail_task_with_diagnostics(tmp_path) -> None:
    agent = _agent(
        tmp_path,
        {
            "missing": {"argv": ["definitely-not-agentcore-command"]},
            "nonzero": {
                "argv": [sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"]
            },
        },
    )
    missing_task = agent.create_task(title="Missing", description="Missing")
    missing = TaskExecutor(agent).execute(missing_task, [RunCheckAction("missing")])
    assert missing.status == "failed"
    assert missing.actions[0].data["check"]["status"] == "launch_failed"

    nonzero_task = agent.create_task(title="Nonzero", description="Nonzero")
    nonzero = TaskExecutor(agent).execute(nonzero_task, [RunCheckAction("nonzero")])
    assert nonzero.status == "failed"
    assert nonzero.actions[0].data["check"]["status"] == "exited"
    assert nonzero.actions[0].data["check"]["returncode"] == 3


def test_run_check_environment_is_controlled_and_config_env_is_trusted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTCORE_SECRET_SHOULD_NOT_LEAK", "secret")
    agent = _agent(
        tmp_path,
        {
            "env": {
                "argv": [
                    sys.executable,
                    "-c",
                    (
                        "import os; "
                        "print(os.getenv('AGENTCORE_TRUSTED', '')); "
                        "print(os.getenv('AGENTCORE_SECRET_SHOULD_NOT_LEAK', 'absent'))"
                    ),
                ],
                "env": {"AGENTCORE_TRUSTED": "yes"},
            }
        },
    )

    result = agent.workspace.checks.run("env")

    assert result.ok
    assert result.stdout.splitlines() == ["yes", "absent"]
    assert "secret" not in str(result.as_dict())


def test_workspace_check_configuration_is_validated(tmp_path) -> None:
    with pytest.raises(ValueError, match="argv"):
        Workspace(tmp_path / "bad", checks={"test": {"argv": "make"}})
