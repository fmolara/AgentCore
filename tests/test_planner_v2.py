from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

import pytest

from agentcore_server import (
    ActionPlan,
    AgentLab,
    IterativeLLMPlanner,
    ListEventSink,
    ReplaceTextAction,
    TaskExecutor,
)
from agentcore_server.generation.result import GenerationMetrics, GenerationResult
from agentcore_server.generation.stream import StreamChunk
from agentcore_server.local import LocalAgentCoreApp
from agentcore_server.local.cli import LocalExitCode, run_cli
from agentcore_server.planning import ExplorationLimits, build_planner
from agentcore_server.planning.exploration import ExplorationPlan
from agentcore_server.planning.explorer import (
    ExplorationBudgetError,
    WorkspaceExplorer,
)
from agentcore_server.planning.iterative import ITERATIVE_SYSTEM_PROMPT
from agentcore_server.planning.llm import PLANNER_SYSTEM_PROMPT
from agentcore_server.runtime.base import Runtime
from agentcore_server.server.state import AgentCoreServerState
from agentcore_server.sessions import Session, SessionStore
from agentcore_server.tasks import TaskStatus
from agentcore_server.workspace import Workspace


def _metrics() -> GenerationMetrics:
    return GenerationMetrics(
        prompt_tokens=10,
        generated_tokens=10,
        ttft_sec=0.01,
        tokens_per_sec=100.0,
        wall_sec=0.1,
    )


class ScriptedRuntime(Runtime):
    def __init__(self, responses: list[str], *, auto_accept_reviews: bool = True) -> None:
        self.responses = list(responses)
        self.auto_accept_reviews = auto_accept_reviews
        self.prompts: list[str] = []
        self.generation_options: list[dict[str, Any]] = []
        self.loaded = True

    def load(self) -> None:
        self.loaded = True

    def shutdown(self) -> None:
        self.loaded = False

    def ready(self) -> bool:
        return self.loaded

    def health(self) -> dict[str, Any]:
        return {"runtime_name": "scripted", "ready": self.ready()}

    def statistics(self) -> dict[str, Any]:
        return self.health()

    def warmup(self, prompt: str | None = None, max_tokens: int = 16) -> GenerationResult:
        return GenerationResult(text="", metrics=_metrics())

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        return Session(system_prompt=system_prompt)

    def generate(self, session: Session, prompt: str, **kwargs: Any) -> GenerationResult:
        self.generation_options.append(dict(kwargs))
        response = self._next(prompt)
        session.add_user_message(prompt)
        session.add_assistant_message(response)
        return GenerationResult(text=response, metrics=_metrics())

    def stream(
        self,
        session: Session,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        self.generation_options.append(dict(kwargs))
        response = self._next(prompt)
        session.add_user_message(prompt)
        yield StreamChunk.started()
        midpoint = max(1, len(response) // 2)
        yield StreamChunk.delta(response[:midpoint])
        yield StreamChunk.delta(response[midpoint:])
        session.add_assistant_message(response)
        yield StreamChunk.completed(text=response, metrics=_metrics())

    def tokenize(self, text_or_messages: Any) -> int:
        return 1

    def _next(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.auto_accept_reviews and prompt.startswith(
            "You are an independent AgentCore candidate-plan reviewer."
        ):
            return json.dumps(
                {
                    "verdict": "accept",
                    "summary": "No material defect identified in scripted test candidate.",
                    "findings": [],
                }
            )
        if not self.responses:
            raise AssertionError("scripted runtime has no response")
        return self.responses.pop(0)


def _lab(
    workspace: Path,
    responses: list[str],
    *,
    config: dict[str, Any] | None = None,
    auto_accept_reviews: bool = True,
) -> AgentLab:
    runtime = ScriptedRuntime(responses, auto_accept_reviews=auto_accept_reviews)
    lab = AgentLab.__new__(AgentLab)
    lab.config = config or {
        "runtime": "fake",
        "workspace": {"root": str(workspace)},
        "planner": {"mode": "iterative"},
    }
    lab.project_root = workspace.parent
    lab.runtime = runtime
    lab.sessions = SessionStore(runtime.create_session)
    return lab


def _explore_response(*actions: dict[str, Any]) -> str:
    return json.dumps(
        {
            "phase": "explore",
            "summary": "Locate parser implementation and tests.",
            "actions": list(actions),
        }
    )


def _final_response(*actions: dict[str, Any]) -> str:
    return json.dumps(
        {
            "phase": "final",
            "plan": {
                "title": "Fix uppercase hexadecimal parsing",
                "description": "Patch the concrete parser and validate it.",
                "actions": list(actions),
                "metadata": {"planner": "iterative_llm"},
            },
        }
    )


def _review_accept(summary: str = "No material defect identified.") -> str:
    return json.dumps({"verdict": "accept", "summary": summary, "findings": []})


def _review_revise(problem: str = "Candidate is unsafe.") -> str:
    return json.dumps(
        {
            "verdict": "revise",
            "summary": "Material correction required.",
            "findings": [
                {
                    "severity": "major",
                    "requirement": "portable arithmetic",
                    "problem": problem,
                    "required_change": "Use defined unsigned arithmetic for LONG_MIN magnitude.",
                }
            ],
        }
    )


def test_task_report_is_not_advertised_to_planners_but_old_plans_remain_valid() -> None:
    assert '- task_report: {"type":"task_report"}' not in PLANNER_SYSTEM_PROMPT
    assert '- task_report: {"type":"task_report"}' not in ITERATIVE_SYSTEM_PROMPT

    plan = ActionPlan.from_dict(
        {
            "title": "Legacy report plan",
            "actions": [{"type": "task_report"}],
        }
    )

    assert [action.action_type for action in plan.actions] == ["task_report"]


def _consume(iterator):
    events = []
    while True:
        try:
            events.append(next(iterator))
        except StopIteration as stop:
            return events, stop.value


def _prepare_parser_workspace(path: Path) -> str:
    (path / "src").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "include").mkdir()
    (path / "src" / "parser.c").write_text(
        "int parse_hex(char c) { return c >= 'a' && c <= 'f'; }\n",
        encoding="utf-8",
    )
    (path / "tests" / "test_parser.c").write_text(
        'assert(parse("0x1a"));\n',
        encoding="utf-8",
    )
    (path / "include" / "parser.h").write_text("int parse_hex(char c);\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AgentCore Test",
            "-c",
            "user.email=agentcore@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_iterative_planner_explores_then_returns_complete_proposal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    baseline = _prepare_parser_workspace(workspace)
    instruction = "Accept uppercase hexadecimal literals without changing decimal parsing."
    responses = [
        _explore_response(
            {
                "type": "search_files",
                "root": ".",
                "name_pattern": "*.c",
                "content_query": "parse_hex",
                "max_results": 20,
            },
            {"type": "read_file", "path": "tests/test_parser.c", "max_lines": 100},
        ),
        _final_response(
            {
                "type": "replace_text",
                "path": "src/parser.c",
                "old": "c >= 'a' && c <= 'f'",
                "new": "(c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')",
            },
            {
                "type": "replace_text",
                "path": "tests/test_parser.c",
                "old": 'assert(parse("0x1a"));',
                "new": 'assert(parse("0x1a"));\\nassert(parse("0x1A"));',
            },
            {"type": "run_check", "check": "build"},
            {"type": "run_check", "check": "test"},
            {"type": "git_diff"},
            {"type": "task_report"},
        ),
    ]
    config = {
        "runtime": "fake",
        "workspace": {
            "root": str(workspace),
            "checks": {
                "build": {"argv": ["make"]},
                "test": {"argv": ["make", "test"]},
            },
        },
        "planner": {"mode": "iterative"},
    }
    lab = _lab(workspace, responses, config=config)
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Fix parser", description=instruction)
    planner = build_planner(config)

    events, result = _consume(
        agent.propose_plan_stream(
            task,
            instruction=instruction,
            planner=planner,
        )
    )

    assert result.ok
    assert result.proposal is not None
    assert isinstance(planner, IterativeLLMPlanner)
    assert task.status == TaskStatus.CREATED
    assert task.checkpoints() == []
    assert subprocess.check_output(["git", "status", "--short"], cwd=workspace, text=True) == ""
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip() == baseline
    action_types = [action.action_type for action in result.proposal.action_plan.actions]
    assert action_types == [
        "replace_text",
        "replace_text",
        "run_check",
        "run_check",
        "git_diff",
        "task_report",
    ]
    assert [item.action_type for item in result.proposal.approval_requirements] == [
        "replace_text",
        "replace_text",
        "run_check",
        "run_check",
    ]

    runtime = lab.runtime
    assert isinstance(runtime, ScriptedRuntime)
    assert len(runtime.prompts) == 3
    assert all(prompt.count(instruction) == 1 for prompt in runtime.prompts)
    assert all("Allowed exploration actions:" in prompt for prompt in runtime.prompts[:2])
    assert "independent AgentCore candidate-plan reviewer" in runtime.prompts[2]
    assert str(workspace) in runtime.prompts[0]
    assert str(workspace) in runtime.prompts[1]
    assert "src/parser.c" in runtime.prompts[1]
    assert "remaining_budget" in runtime.prompts[1]

    event_types = [event.event_type for event in events]
    assert event_types.count("planning.started") == 1
    assert event_types.count("planning.final_plan.generated") == 1
    assert event_types.count("plan.proposed") == 1
    exploration_action_events = [
        event.event_type
        for event in events
        if event.event_type.startswith("exploration.action.")
    ]
    assert exploration_action_events == [
        "exploration.action.started",
        "exploration.action.completed",
        "exploration.action.started",
        "exploration.action.completed",
    ]
    assistant_events = [event for event in events if event.event_type.startswith("assistant.")]
    assert all(
        "exploration." not in str(event.payload.get("delta", ""))
        for event in assistant_events
    )
    assert [event.event_type for event in sink.events][1:] == event_types


def test_exploration_ordinary_failure_keeps_other_observations(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.write_text("ok.txt", "content\n")
    limits = ExplorationLimits()
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Read candidates",
            "actions": [
                {"type": "read_file", "path": "missing.txt"},
                {"type": "read_file", "path": "ok.txt"},
            ],
        },
        limits=limits,
    )

    observations, _ = WorkspaceExplorer(workspace, limits=limits).execute(plan)

    assert [item.status for item in observations] == ["failed", "ok"]
    assert observations[1].data["text"] == "content\n"


def test_exploration_validates_whole_round_before_execution(tmp_path, monkeypatch) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.write_text("ok.txt", "content\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.txt").write_text("outside\n", encoding="utf-8")
    (workspace.root / "escape").symlink_to(outside, target_is_directory=True)
    limits = ExplorationLimits()
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Unsafe round",
            "actions": [
                {"type": "read_file", "path": "ok.txt"},
                {"type": "read_file", "path": "escape/outside.txt"},
            ],
        },
        limits=limits,
    )
    called = False
    explorer = WorkspaceExplorer(workspace, limits=limits)
    original = explorer._read_file

    def spy(action):
        nonlocal called
        called = True
        return original(action)

    monkeypatch.setattr(explorer, "_read_file", spy)
    with pytest.raises(ValueError, match="escapes workspace"):
        explorer.execute(plan)
    assert called is False


def test_exploration_schema_rejects_mutation_and_unknown_action() -> None:
    limits = ExplorationLimits()
    for action_type in ("write_file", "shell"):
        with pytest.raises(ValueError, match="unknown exploration action"):
            ExplorationPlan.from_dict(
                {
                    "phase": "explore",
                    "summary": "Bad",
                    "actions": [{"type": action_type, "path": "x"}],
                },
                limits=limits,
            )


def test_read_file_directory_is_bounded_failed_observation(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.mkdir("src")
    limits = ExplorationLimits()
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Read src",
            "actions": [{"type": "read_file", "path": "src"}],
        },
        limits=limits,
    )

    observations, _ = WorkspaceExplorer(workspace, limits=limits).execute(plan)

    assert observations[0].status == "failed"
    assert observations[0].error == "read_file target is a directory"


def test_list_search_and_read_are_deterministic_and_bounded(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.mkdir("src")
    workspace.write_text("src/z.c", "needle\nsecond\nthird\n")
    workspace.write_text("src/a.c", "needle\n")
    workspace.write_text("src/b.bin", "x\0y")
    workspace.write_text(".hidden", "needle")
    limits = ExplorationLimits(max_files_returned=2, max_single_file_bytes=16)
    explorer = WorkspaceExplorer(workspace, limits=limits)

    listing = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "List",
            "actions": [{"type": "list_directory", "path": ".", "max_depth": 2}],
        },
        limits=limits,
    )
    listed, _ = explorer.execute(listing)
    paths = [entry["path"] for entry in listed[0].data["entries"]]
    assert paths == sorted(paths)
    assert ".hidden" not in paths
    assert listed[0].truncated is True

    search = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Search",
            "actions": [
                {
                    "type": "search_files",
                    "root": "src",
                    "name_pattern": "*.c",
                    "content_query": "needle",
                    "max_results": 2,
                }
            ],
        },
        limits=limits,
    )
    found, _ = explorer.execute(search)
    assert [item["path"] for item in found[0].data["matches"]] == [
        "src/a.c",
        "src/z.c",
    ]

    read = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Read",
            "actions": [
                {
                    "type": "read_file",
                    "path": "src/z.c",
                    "start_line": 2,
                    "max_lines": 1,
                    "max_bytes": 16,
                }
            ],
        },
        limits=limits,
    )
    read_result, _ = explorer.execute(read)
    assert read_result[0].data["text"] == "second\n"
    assert read_result[0].truncated is True


def test_symlink_escape_and_total_budget_are_hard_failures(tmp_path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    workspace = Workspace(root)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    limits = ExplorationLimits(max_total_actions=1)
    explorer = WorkspaceExplorer(workspace, limits=limits)

    unsafe = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Unsafe",
            "actions": [{"type": "read_file", "path": "escape/secret.txt"}],
        },
        limits=limits,
    )
    with pytest.raises(ValueError, match="escapes workspace"):
        explorer.execute(unsafe)

    valid = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Missing",
            "actions": [{"type": "read_file", "path": "missing"}],
        },
        limits=limits,
    )
    explorer.execute(valid)
    with pytest.raises(ExplorationBudgetError, match="max_total_actions"):
        explorer.execute(valid)


def test_max_rounds_fails_without_starting_task(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "list_directory", "path": ".", "max_depth": 1}),
            _explore_response({"type": "list_directory", "path": ".", "max_depth": 1}),
            _explore_response({"type": "list_directory", "path": ".", "max_depth": 1}),
        ],
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Explore forever", description="Explore")
    planner = IterativeLLMPlanner(limits=ExplorationLimits(max_rounds=2))

    result = planner.propose(agent, task, instruction="Explore")

    assert result.status == "failed"
    assert "more exploration after max_rounds=2" in (result.error or "")
    assert task.status == TaskStatus.CREATED
    assert [event.event_type for event in sink.events].count("planning.failed") == 1
    runtime = lab.runtime
    assert isinstance(runtime, ScriptedRuntime)
    assert "phase MUST be \"final\" or \"cannot_plan\"" in runtime.prompts[-1]


def test_multiple_exploration_rounds_feed_only_new_observations(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "parser.c").write_text("parser token\n", encoding="utf-8")
    instruction = "Inspect parser and produce a complete report plan."
    lab = _lab(
        workspace,
        [
            _explore_response(
                {"type": "list_directory", "path": ".", "max_depth": 2}
            ),
            _explore_response(
                {"type": "read_file", "path": "src/parser.c", "max_lines": 20}
            ),
            _final_response({"type": "git_diff"}, {"type": "task_report"}),
        ],
    )
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Inspect", description=instruction)

    result = build_planner(lab.config).propose(
        agent,
        task,
        instruction=instruction,
    )

    assert result.ok
    runtime = lab.runtime
    assert isinstance(runtime, ScriptedRuntime)
    assert len(runtime.prompts) == 4
    assert "src/parser.c" in runtime.prompts[1]
    assert "parser token" in runtime.prompts[2]
    assert all(prompt.count(instruction) == 1 for prompt in runtime.prompts)


def test_observation_byte_budget_fails_visibly(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.txt").write_text("x" * 1000, encoding="utf-8")
    limits = ExplorationLimits(
        max_single_file_bytes=1000,
        max_observation_text_per_action=1000,
        max_total_observation_bytes=100,
    )
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Read large file",
            "actions": [{"type": "read_file", "path": "large.txt"}],
        },
        limits=limits,
    )

    with pytest.raises(ExplorationBudgetError, match="max_total_observation_bytes"):
        WorkspaceExplorer(Workspace(workspace), limits=limits).execute(plan)


def test_search_reports_binary_encoding_and_symlink_skips(tmp_path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("needle", encoding="utf-8")
    (workspace_path / "binary.dat").write_bytes(b"needle\0binary")
    (workspace_path / "invalid.dat").write_bytes(b"needle\xff")
    (workspace_path / "link.dat").symlink_to(outside)
    workspace = Workspace(workspace_path)
    limits = ExplorationLimits()
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Search data",
            "actions": [
                {
                    "type": "search_files",
                    "root": ".",
                    "name_pattern": "*.dat",
                    "content_query": "needle",
                    "max_results": 10,
                }
            ],
        },
        limits=limits,
    )

    observations, _ = WorkspaceExplorer(workspace, limits=limits).execute(plan)
    data = observations[0].data

    assert data["matches"] == []
    assert data["skipped_binary"] == ["binary.dat"]
    assert data["skipped_encoding"] == ["invalid.dat"]
    assert data["skipped_symlink"] == ["link.dat"]


def test_search_caps_total_files_and_content_bytes(tmp_path) -> None:
    workspace = Workspace(tmp_path / "workspace")
    workspace.mkdir("src")
    for index in range(5):
        workspace.write_text(f"src/{index}.c", "x" * 32)
    limits = ExplorationLimits(
        max_search_files_scanned=2,
        max_search_bytes=40,
        max_single_file_bytes=32,
    )
    plan = ExplorationPlan.from_dict(
        {
            "phase": "explore",
            "summary": "Bounded search",
            "actions": [
                {
                    "type": "search_files",
                    "root": "src",
                    "name_pattern": "*.c",
                    "content_query": "not-present",
                    "max_results": 10,
                }
            ],
        },
        limits=limits,
    )

    observations, _ = WorkspaceExplorer(workspace, limits=limits).execute(plan)
    observation = observations[0]

    assert observation.truncated is True
    assert observation.data["files_scanned"] == 2
    assert observation.data["content_bytes_scanned"] == 40
    assert observation.data["truncation_reasons"] == ["max_search_bytes"]


def test_planning_does_not_invoke_task_executor(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "list_directory", "path": "."}),
            _final_response({"type": "git_diff"}, {"type": "task_report"}),
        ],
    )
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Inspect", description="Inspect")

    monkeypatch.setattr(
        TaskExecutor,
        "execute",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TaskExecutor invoked during planning")
        ),
    )
    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.ok
    assert task.status == TaskStatus.CREATED


def test_final_read_only_exploration_disguised_as_final_is_rejected(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("x", encoding="utf-8")
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "read_file", "path": "parser.c"}),
            _final_response({"type": "read_file", "path": "parser.c"}),
        ],
    )
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Fix parser", description="Fix parser")

    result = build_planner(lab.config).propose(agent, task, instruction="Fix parser")

    assert result.status == "failed"
    assert "exploration-only" in (result.error or "")


def test_composition_roots_use_same_iterative_class_and_values(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "runtime": "fake",
        "workspace": {"root": str(workspace)},
        "planner": {
            "mode": "iterative",
            "max_tokens": 777,
            "exploration": {"max_rounds": 2, "max_total_actions": 9},
        },
    }
    lab = _lab(
        workspace,
        [_final_response({"type": "git_diff"}, {"type": "task_report"})],
        config=config,
    )

    local = LocalAgentCoreApp(lab, workspace=str(workspace))
    server = AgentCoreServerState(
        lab,
        workspace_root=workspace,
        start_runtime=False,
        warmup=False,
    )

    assert type(local.planner) is type(server.planner) is IterativeLLMPlanner
    assert local.planner.max_tokens == server.planner.max_tokens == 777
    assert local.planner.limits == server.planner.limits
    assert local.planner.limits.max_rounds == 2
    assert local.planner.limits.max_total_actions == 9
    assert isinstance(TaskExecutor(local.agent), TaskExecutor)

    agent_record = server.create_agent(
        system_prompt=None,
        workspace_root=str(workspace),
        workspace_mode=Workspace.READ_WRITE,
        workspace_metadata={},
        generation_options={},
    )
    task_record = server.create_task(
        agent_record.id,
        title="Inspect",
        description="Inspect workspace",
        metadata={},
    )
    result = server.create_proposal(
        task_record.id,
        instruction="Inspect workspace",
        max_tokens=768,
        temperature=0,
    )

    assert result.ok
    assert result.proposal is not None
    assert server.get_proposal(result.proposal.id).proposal is result.proposal


def test_local_proposal_only_can_override_simple_config_with_iterative(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("parser\n", encoding="utf-8")
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "read_file", "path": "parser.c"}),
            _final_response({"type": "git_diff"}, {"type": "task_report"}),
        ],
        config={
            "runtime": "fake",
            "workspace": {"root": str(workspace)},
            "planner": {"mode": "simple"},
        },
    )
    trace = tmp_path / "trace.jsonl"

    code = run_cli(
        [
            "--config",
            "unused",
            "--workspace",
            str(workspace),
            "--prompt",
            "Inspect parser",
            "--planner",
            "iterative",
            "--proposal-only",
            "--trace-file",
            str(trace),
            "--no-color",
            "--no-warmup",
        ],
        lab_factory=lambda _: lab,
    )

    assert code == LocalExitCode.SUCCESS
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    event_types = [record["event_type"] for record in records]
    assert event_types.count("planning.started") == 1
    assert event_types.count("plan.proposed") == 1
    assert not any(record["event_type"] == "task.started" for record in records)


def test_each_final_proposal_gets_new_identity(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("x", encoding="utf-8")
    response = _final_response({"type": "git_diff"}, {"type": "task_report"})
    lab = _lab(workspace, [response, response])
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Inspect", description="Inspect")
    planner = build_planner(lab.config)

    first = planner.propose(agent, task, instruction="Inspect")
    second = planner.propose(agent, task, instruction="Inspect")

    assert first.proposal is not None and second.proposal is not None
    assert first.proposal.id != second.proposal.id


def test_generation_budget_is_capped_to_remaining_context(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        [_final_response({"type": "git_diff"}, {"type": "task_report"})],
        config={
            "runtime": "fake",
            "workspace": {"root": str(workspace)},
            "context": {"max_context_tokens": 1200},
            "planner": {"mode": "iterative", "max_tokens": 500},
        },
    )
    monkeypatch.setattr(lab.runtime, "tokenize", lambda messages: 900)
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Inspect", description="Inspect")

    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.ok
    assert lab.runtime.generation_options[0]["max_tokens"] == 268
    budget = next(
        event for event in sink.events if event.event_type == "planning.generation_budget"
    )
    assert budget.payload["requested_max_tokens"] == 500
    assert budget.payload["effective_max_tokens"] == 268


def test_generation_fails_before_model_when_context_is_exhausted(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        [_final_response({"type": "git_diff"})],
        config={
            "runtime": "fake",
            "workspace": {"root": str(workspace)},
            "context": {"max_context_tokens": 1000},
            "planner": {"mode": "iterative"},
        },
    )
    monkeypatch.setattr(lab.runtime, "tokenize", lambda messages: 950)
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Inspect", description="Inspect")

    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.status == "failed"
    assert "insufficient model context" in (result.error or "")
    assert lab.runtime.prompts == []


def test_compact_candidate_is_reviewed_before_proposal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("return 0;\n", encoding="utf-8")
    lab = _lab(
        workspace,
        [
            _final_response(
                {
                    "type": "replace_text",
                    "path": "parser.c",
                    "old": "return 0;",
                    "new": "return 1;",
                },
                {"type": "git_diff"},
            ),
            _review_accept(),
        ],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Edit parser", description="Edit parser")

    result = build_planner(lab.config).propose(agent, task, instruction="Edit parser")

    assert result.ok and result.proposal is not None
    event_types = [event.event_type for event in sink.events]
    assert event_types.index("planning.review.completed") < event_types.index(
        "planning.final_candidate.accepted"
    )
    assert event_types.count("planning.final_candidate.accepted") == 1
    assert task.status == TaskStatus.CREATED
    assert (workspace / "parser.c").read_text(encoding="utf-8") == "return 0;\n"


def test_candidate_diagnostics_cover_existing_write_redundant_read_and_checks(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("return 0;\n", encoding="utf-8")
    instruction = "Run the configured build check and configured test check."
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "read_file", "path": "parser.c"}),
            _final_response(
                {"type": "read_file", "path": "parser.c"},
                {"type": "write_file", "path": "parser.c", "content": "return 1;\n"},
            ),
            _review_accept(),
        ],
        config={
            "runtime": "fake",
            "workspace": {
                "root": str(workspace),
                "checks": {"build": {"argv": ["make"]}, "test": {"argv": ["make", "test"]}},
            },
            "planner": {
                "mode": "iterative",
                "finalization": {"max_action_payload_bytes": 4},
            },
        },
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Edit", description=instruction)

    result = build_planner(lab.config).propose(agent, task, instruction=instruction)

    assert result.ok
    generated = next(
        event for event in sink.events if event.event_type == "planning.final_candidate.generated"
    )
    codes = {item["code"] for item in generated.payload["candidate"]["diagnostics"]}
    assert codes == {
        "existing_file_write",
        "large_action_payload",
        "missing_explicit_check",
        "redundant_exploration_read",
    }


def test_malformed_candidate_gets_exactly_one_recovery_and_executes_nothing(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("return 0;\n", encoding="utf-8")
    lab = _lab(
        workspace,
        [
            '{"phase":"final","plan":{"title":"truncated',
            _final_response({"type": "git_diff"}),
            _review_accept(),
        ],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Inspect", description="Inspect")
    monkeypatch.setattr(
        TaskExecutor,
        "execute",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("executor invoked")),
    )

    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.ok
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("planning.format_recovery.started") == 1
    assert event_types.count("planning.format_recovery.completed") == 1
    assert task.status == TaskStatus.CREATED
    assert (workspace / "parser.c").read_text(encoding="utf-8") == "return 0;\n"


def test_second_malformed_candidate_fails_without_proposal(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        ["{bad", "{still bad"],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Inspect", description="Inspect")

    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.status == "failed"
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("planning.format_recovery.started") == 1
    assert event_types.count("planning.failed") == 1
    assert "plan.proposed" not in event_types


def test_malformed_review_has_one_recovery_and_never_implies_acceptance(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lab = _lab(
        workspace,
        [
            _final_response({"type": "git_diff"}),
            '{"verdict":"accept"',
            _review_accept("Recovered review accepted the candidate."),
        ],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Inspect", description="Inspect")

    result = build_planner(lab.config).propose(agent, task, instruction="Inspect")

    assert result.ok and result.proposal is not None
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("planning.format_recovery.started") == 1
    assert event_types.count("planning.final_candidate.accepted") == 1


def test_review_revises_once_and_fresh_review_accepts(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("limit\n", encoding="utf-8")
    unsafe = _final_response(
        {
            "type": "replace_text",
            "path": "parser.c",
            "old": "limit",
            "new": "(unsigned long)(-LONG_MIN)",
        }
    )
    safe = _final_response(
        {
            "type": "replace_text",
            "path": "parser.c",
            "old": "limit",
            "new": "0UL - (unsigned long)LONG_MIN",
        },
        {"type": "git_diff"},
    )
    lab = _lab(
        workspace,
        [unsafe, _review_revise(), safe, _review_accept("Revised arithmetic is defined.")],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Portable parser", description="Avoid -LONG_MIN")

    result = build_planner(lab.config).propose(
        agent,
        task,
        instruction="Avoid -LONG_MIN",
    )

    assert result.ok and result.proposal is not None
    replacement = result.proposal.action_plan.actions[0]
    assert isinstance(replacement, ReplaceTextAction)
    assert replacement.new == "0UL - (unsigned long)LONG_MIN"
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("planning.revision.started") == 1
    assert event_types.count("planning.revision.completed") == 1
    assert event_types.count("planning.review.started") == 2
    assert event_types.count("planning.review.completed") == 2
    assert event_types.count("planning.final_candidate.accepted") == 1
    candidate_ids = [
        event.payload["candidate"]["id"]
        for event in sink.events
        if event.event_type in {
            "planning.final_candidate.generated",
            "planning.revision.completed",
        }
    ]
    assert len(candidate_ids) == 2
    assert candidate_ids[0] != candidate_ids[1]


def test_malformed_revision_gets_one_recovery_before_final_review(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("old\n", encoding="utf-8")
    candidate = _final_response(
        {"type": "replace_text", "path": "parser.c", "old": "old", "new": "unsafe"}
    )
    revised = _final_response(
        {"type": "replace_text", "path": "parser.c", "old": "old", "new": "safe"}
    )
    lab = _lab(
        workspace,
        [candidate, _review_revise(), '{"phase":"final"', revised, _review_accept()],
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Edit", description="Edit safely")

    result = build_planner(lab.config).propose(agent, task, instruction="Edit safely")

    assert result.ok and result.proposal is not None
    replacement = result.proposal.action_plan.actions[0]
    assert isinstance(replacement, ReplaceTextAction)
    assert replacement.new == "safe"
    event_types = [event.event_type for event in sink.events]
    assert event_types.count("planning.format_recovery.started") == 1
    assert event_types.count("planning.revision.started") == 1
    assert event_types.count("planning.review.started") == 2


@pytest.mark.parametrize(
    "second_review",
    [
        _review_revise("Still unsafe."),
        json.dumps({"verdict": "cannot_verify", "reason": "Insufficient evidence."}),
    ],
)
def test_revised_candidate_must_pass_final_review(tmp_path, second_review) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("old\n", encoding="utf-8")
    candidate = _final_response(
        {"type": "replace_text", "path": "parser.c", "old": "old", "new": "new"}
    )
    lab = _lab(
        workspace,
        [candidate, _review_revise(), candidate, second_review],
        auto_accept_reviews=False,
    )
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Edit", description="Edit")

    result = build_planner(lab.config).propose(agent, task, instruction="Edit")

    assert result.status == "failed"
    assert result.proposal is None
    assert task.status == TaskStatus.CREATED
    assert (workspace / "parser.c").read_text(encoding="utf-8") == "old\n"


def test_phase_specific_generation_budgets_are_distinct(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "parser.c").write_text("x\n", encoding="utf-8")
    config = {
        "runtime": "fake",
        "workspace": {"root": str(workspace)},
        "planner": {
            "mode": "iterative",
            "finalization": {
                "budgets": {
                    "exploration": 300,
                    "final_candidate": 700,
                    "review": 220,
                }
            },
        },
    }
    lab = _lab(
        workspace,
        [
            _explore_response({"type": "read_file", "path": "parser.c"}),
            _final_response({"type": "git_diff"}),
            _review_accept(),
        ],
        config=config,
        auto_accept_reviews=False,
    )
    sink = ListEventSink()
    agent = lab.create_agent(workspace_root=workspace, event_sink=sink)
    task = agent.create_task(title="Inspect", description="Inspect")

    result = build_planner(config).propose(agent, task, instruction="Inspect")

    assert result.ok
    budgets = {
        event.payload["budget_kind"]: event.payload["effective_max_tokens"]
        for event in sink.events
        if event.event_type == "planning.generation_budget"
    }
    assert budgets == {"exploration": 300, "final_candidate": 700, "review": 220}


def test_signed_parser_scripted_regression_rejects_unsafe_then_accepts_compact_revision(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "include").mkdir()
    (workspace / "src" / "parser.c").write_text("unsigned long limit = LONG_MAX;\n", encoding="utf-8")
    (workspace / "tests" / "test_parser.c").write_text("test_unsigned();\n", encoding="utf-8")
    (workspace / "include" / "parser.h").write_text("int parse(const char *, long *);\n", encoding="utf-8")
    instruction = (
        "Run the configured build check and configured test check. "
        "Support LONG_MIN without undefined behavior."
    )
    unsafe = _final_response(
        {
            "type": "replace_text",
            "path": "src/parser.c",
            "old": "LONG_MAX",
            "new": "(unsigned long)(-LONG_MIN)",
        },
        {"type": "run_check", "check": "build"},
        {"type": "run_check", "check": "test"},
        {"type": "git_diff"},
    )
    revised = _final_response(
        {
            "type": "replace_text",
            "path": "src/parser.c",
            "old": "unsigned long limit = LONG_MAX;",
            "new": "unsigned long limit = 0UL - (unsigned long)LONG_MIN;",
        },
        {
            "type": "replace_text",
            "path": "tests/test_parser.c",
            "old": "test_unsigned();",
            "new": "test_unsigned();\ntest_signed_boundaries();",
        },
        {"type": "run_check", "check": "build"},
        {"type": "run_check", "check": "test"},
        {"type": "git_diff"},
    )
    lab = _lab(
        workspace,
        [
            _explore_response(
                {"type": "read_file", "path": "src/parser.c"},
                {"type": "read_file", "path": "include/parser.h"},
                {"type": "read_file", "path": "tests/test_parser.c"},
            ),
            unsafe,
            _review_revise("The candidate evaluates -LONG_MIN."),
            revised,
            _review_accept("Defined unsigned arithmetic and requested checks are present."),
        ],
        config={
            "runtime": "fake",
            "workspace": {
                "root": str(workspace),
                "checks": {"build": {"argv": ["make"]}, "test": {"argv": ["make", "test"]}},
            },
            "planner": {"mode": "iterative"},
        },
        auto_accept_reviews=False,
    )
    agent = lab.create_agent(workspace_root=workspace)
    task = agent.create_task(title="Signed parser", description=instruction)

    result = build_planner(lab.config).propose(agent, task, instruction=instruction)

    assert result.ok and result.proposal is not None
    actions = result.proposal.action_plan.actions
    assert [action.action_type for action in actions] == [
        "replace_text",
        "replace_text",
        "run_check",
        "run_check",
        "git_diff",
    ]
    assert result.proposal.status.value == "proposed"
    assert task.status == TaskStatus.CREATED
    assert (workspace / "src" / "parser.c").read_text() == "unsigned long limit = LONG_MAX;\n"
