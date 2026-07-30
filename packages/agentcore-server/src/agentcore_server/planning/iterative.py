from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from agentcore_server.events import AgentEvent
from agentcore_server.executor import (
    ActionPlan,
    ApprovalPolicy,
    PlanProposal,
    action_to_dict,
)
from agentcore_server.executor.actions import (
    ReadFileAction,
    ReplaceTextAction,
    RunCheckAction,
    WriteFileAction,
)
from agentcore_server.generation.result import GenerationMetrics
from agentcore_server.planning.exploration import (
    ExplorationLimits,
    ExplorationRound,
    PlanningDecision,
    PlanningPhase,
)
from agentcore_server.planning.explorer import (
    ExplorationBudgetError,
    ExplorationError,
    WorkspaceExplorer,
)
from agentcore_server.planning.finalization import (
    CandidateReview,
    FinalCandidate,
    PlanDiagnostic,
    build_recovery_prompt,
    build_review_prompt,
    build_revision_prompt,
)
from agentcore_server.planning.json_output import parse_json_object
from agentcore_server.planning.planner import PlannerResult
from agentcore_server.tasks import Task


ITERATIVE_SYSTEM_PROMPT = """You are AgentCore's bounded workspace planner.

Return exactly one JSON object. Do not return Markdown, prose, comments, or
code fences. Do not reveal hidden chain-of-thought.

Choose exactly one phase:

EXPLORE:
{"phase":"explore","summary":"visible summary","actions":[...]}

FINAL:
{"phase":"final","plan":{"title":"...","description":"...","actions":[...],"metadata":{"planner":"iterative_llm"}}}

CANNOT_PLAN:
{"phase":"cannot_plan","reason":"visible reason"}

Exploration is temporary, read-only, bounded, and internal to planning. An
EXPLORE response is not the final task plan. After observations arrive, either
request another bounded discovery round or emit FINAL. The FINAL plan must
cover the complete user task and will be shown to a human for explicit
approval before execution.

Allowed exploration actions:
- list_directory: {"type":"list_directory","path":"relative/path","max_depth":1,"include_hidden":false}
- search_files: {"type":"search_files","root":"relative/path","name_pattern":"*.c","content_query":"optional literal","max_results":20}
- read_file: {"type":"read_file","path":"relative/file","start_line":1,"max_lines":200,"max_bytes":65536}

Exploration forbids writes, commands, Git mutation, network access, absolute
paths, and paths outside the workspace.

Allowed FINAL action types:
- read_file: {"type":"read_file","path":"relative/file"}
- write_file: {"type":"write_file","path":"relative/file","content":"complete text"}
- replace_text: {"type":"replace_text","path":"relative/file","old":"exact text","new":"replacement","count":-1}
- create_checkpoint: {"type":"create_checkpoint","label":"short label","description":"optional text"}
- git_status: {"type":"git_status"}
- git_diff: {"type":"git_diff"}
{run_check_schema}

Forbidden in FINAL:
- shell commands or model-supplied command arguments
- arbitrary Git commands
- network operations
- absolute paths or files outside the workspace
- unknown action types
- automatic commits

Checks and mutations belong only in FINAL. Use only concrete paths learned
from observations. Do not return a FINAL plan containing only discovery reads.
Do not repeat read_file actions already completed during exploration unless a
new execution-time read has a concrete purpose. For localized edits to existing
files, strongly prefer compact replace_text actions with exact unique old/new
text. Use write_file for a genuinely new file or when complete replacement is
objectively necessary. Keep embedded action payloads compact.
"""

FINAL_RESPONSE_SCHEMA = (
    '{"phase":"final","plan":{"title":"...","description":"...",'
    '"actions":[{"type":"..."}],"metadata":{"planner":"iterative_llm"}}}'
)
REVIEW_RESPONSE_SCHEMA = (
    '{"verdict":"accept","summary":"...","findings":[]} OR '
    '{"verdict":"revise","summary":"...","findings":[{"severity":"major",'
    '"requirement":"...","problem":"...","required_change":"..."}]} OR '
    '{"verdict":"cannot_verify","reason":"..."}'
)


class IterativeLLMPlanner:
    diagnostics_managed = True

    def __init__(
        self,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        limits: ExplorationLimits | None = None,
        check_names: tuple[str, ...] = (),
        phase_budgets: dict[str, int] | None = None,
        minimum_phase_tokens: dict[str, int] | None = None,
        max_action_payload_bytes: int = 16384,
        forbid_existing_file_write: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.limits = limits or ExplorationLimits()
        self.check_names = tuple(sorted(check_names))
        configured_budgets = dict(phase_budgets or {})
        self.phase_budgets = {
            "exploration": configured_budgets.get("exploration", max_tokens),
            "final_candidate": configured_budgets.get("final_candidate", max(max_tokens, 2048)),
            "format_recovery": configured_budgets.get("format_recovery", max(max_tokens, 2048)),
            "review": configured_budgets.get("review", min(max_tokens, 768)),
            "revision": configured_budgets.get("revision", max(max_tokens, 2048)),
            "final_review": configured_budgets.get("final_review", min(max_tokens, 768)),
        }
        configured_minimums = dict(minimum_phase_tokens or {})
        self.minimum_phase_tokens = {
            "exploration": configured_minimums.get("exploration", 128),
            "final_candidate": configured_minimums.get("final_candidate", 512),
            "format_recovery": configured_minimums.get("format_recovery", 512),
            "review": configured_minimums.get("review", 192),
            "revision": configured_minimums.get("revision", 512),
            "final_review": configured_minimums.get("final_review", 192),
        }
        if (
            not isinstance(max_action_payload_bytes, int)
            or isinstance(max_action_payload_bytes, bool)
            or max_action_payload_bytes <= 0
        ):
            raise ValueError("max_action_payload_bytes must be positive")
        if not isinstance(forbid_existing_file_write, bool):
            raise ValueError("forbid_existing_file_write must be boolean")
        self.max_action_payload_bytes = max_action_payload_bytes
        self.forbid_existing_file_write = forbid_existing_file_write
        for phase, value in self.phase_budgets.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"planner budget for {phase} must be a positive integer")

    def propose(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None = None,
        **generation_options: Any,
    ) -> PlannerResult:
        iterator = self._run(
            agent,
            task,
            instruction=instruction,
            approval_policy=approval_policy,
            stream_model=False,
            emit_plan_proposed=False,
            generation_options=generation_options,
        )
        return _consume_result(iterator)

    def propose_stream(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None = None,
        **generation_options: Any,
    ) -> Iterator[AgentEvent]:
        return self._run(
            agent,
            task,
            instruction=instruction,
            approval_policy=approval_policy,
            stream_model=True,
            emit_plan_proposed=True,
            generation_options=generation_options,
        )

    def _run(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        approval_policy: ApprovalPolicy | None,
        stream_model: bool,
        emit_plan_proposed: bool,
        generation_options: dict[str, Any],
    ) -> Iterator[AgentEvent]:
        explorer = WorkspaceExplorer(agent.workspace, limits=self.limits)
        rounds: list[ExplorationRound] = []
        options = {"temperature": self.temperature}
        options.update(generation_options)
        instruction_digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        yield self._event(
            agent,
            task,
            "planning.started",
            "Bounded workspace planning started",
            {
                "planner": "iterative",
                "limits": self.limits.as_dict(),
                "task_context_sha256": instruction_digest,
            },
        )

        last_raw = ""
        last_metrics: GenerationMetrics | None = None
        for round_number in range(1, self.limits.max_rounds + 2):
            prompt = self.build_prompt(
                agent,
                task,
                instruction=instruction,
                rounds=tuple(rounds),
                round_number=round_number,
                actions_used=explorer.total_actions,
                observation_bytes_used=explorer.total_observation_bytes,
            )
            yield self._event(
                agent,
                task,
                "exploration.round.started",
                f"Planning round {round_number} started",
                {
                    "round": round_number,
                    "context_mode": "stateless_immutable_context",
                    "session_id": agent.session.id,
                    "workspace_root": str(agent.workspace.root),
                    "task_context_sha256": instruction_digest,
                    "instruction_in_current_prompt": True,
                    "previous_observation_rounds": len(rounds),
                    "remaining_budget": self._remaining_budget(explorer, round_number),
                },
            )
            yield self._event(
                agent,
                task,
                "planner.prompt",
                f"Effective iterative planner prompt prepared for round {round_number}",
                {
                    "round": round_number,
                    "prompt": prompt,
                    "sanitized": True,
                    "visible_model_input": True,
                },
            )

            try:
                phase = "exploration" if not rounds else "final_candidate"
                last_raw, last_metrics = yield from self._generate_structured(
                    agent,
                    task,
                    prompt,
                    phase=phase,
                    round_number=round_number,
                    options=options,
                    stream_model=stream_model,
                )
            except ValueError as exc:
                result = PlannerResult.failed(
                    error=str(exc),
                    raw_text=last_raw,
                    metrics=last_metrics,
                )
                yield self._planning_failed(
                    agent,
                    task,
                    result.error,
                    round_number,
                    rounds,
                )
                return result
            try:
                parsed = parse_json_object(last_raw)
                decision = PlanningDecision.from_dict(parsed, limits=self.limits)
            except Exception as exc:
                yield self._event(
                    agent,
                    task,
                    "planning.final_candidate.parse_failed",
                    "Structured planning response could not be parsed",
                    {
                        "round": round_number,
                        "error": str(exc),
                        "malformed_output": _bounded_text(last_raw),
                    },
                )
                recovered = yield from self._recover_planning_decision(
                    agent,
                    task,
                    instruction=instruction,
                    rounds=rounds,
                    malformed_output=last_raw,
                    round_number=round_number,
                    options=options,
                    stream_model=stream_model,
                )
                if recovered is None:
                    result = PlannerResult.failed(
                        error=str(exc),
                        raw_text=last_raw,
                        metrics=last_metrics,
                    )
                    yield self._planning_failed(agent, task, result.error, round_number, rounds)
                    return result
                parsed, decision, last_raw, last_metrics = recovered

            yield self._event(
                agent,
                task,
                "planning.response.parsed",
                f"Structured planning response parsed as {decision.phase.value}",
                {
                    "round": round_number,
                    "phase": decision.phase.value,
                    "structured_response": parsed,
                },
            )
            if decision.phase == PlanningPhase.CANNOT_PLAN:
                result = PlannerResult.failed(
                    error=decision.reason or "model cannot plan task",
                    raw_text=last_raw,
                    metrics=last_metrics,
                )
                yield self._planning_failed(agent, task, result.error, round_number, rounds)
                return result

            if decision.phase == PlanningPhase.FINAL:
                try:
                    _validate_action_plan_mapping(decision.final_plan or {})
                    plan = ActionPlan.from_dict(decision.final_plan or {})
                    diagnostics = self._validate_final_plan(
                        agent,
                        plan,
                        explored=bool(rounds),
                        rounds=rounds,
                        instruction=instruction,
                    )
                except Exception as exc:
                    yield self._event(
                        agent,
                        task,
                        "planning.final_candidate.validation_failed",
                        "Final candidate failed deterministic validation",
                        {"round": round_number, "error": str(exc)},
                    )
                    result = PlannerResult.failed(
                        error=str(exc),
                        raw_text=last_raw,
                        metrics=last_metrics,
                    )
                    yield self._planning_failed(agent, task, result.error, round_number, rounds)
                    return result

                candidate = FinalCandidate(plan, tuple(diagnostics))
                yield self._event(
                    agent,
                    task,
                    "planning.final_candidate.generated",
                    f"Final candidate generated: {plan.title}",
                    {
                        "round": round_number,
                        "candidate": candidate.as_dict(),
                    },
                )
                finalized = yield from self._review_and_maybe_revise(
                    agent,
                    task,
                    instruction=instruction,
                    rounds=rounds,
                    candidate=candidate,
                    candidate_raw=last_raw,
                    candidate_metrics=last_metrics,
                    round_number=round_number,
                    options=options,
                    stream_model=stream_model,
                )
                if finalized is None:
                    result = PlannerResult.failed(
                        error="candidate review did not accept a final plan",
                        raw_text=last_raw,
                        metrics=last_metrics,
                    )
                    yield self._planning_failed(agent, task, result.error, round_number, rounds)
                    return result
                candidate, review, last_raw, last_metrics = finalized
                plan = candidate.action_plan
                diagnostic_dicts = [item.as_dict() for item in candidate.diagnostics]
                proposal = PlanProposal.from_action_plan(
                    task_id=task.id,
                    action_plan=plan,
                    summary=f"Workspace-aware proposal for: {instruction}",
                    approval_policy=approval_policy,
                    metadata={
                        "planner": "iterative_llm",
                        "instruction": instruction,
                        "exploration_rounds": len(rounds),
                        "exploration_actions": explorer.total_actions,
                        "observation_bytes": explorer.total_observation_bytes,
                        "validation_warnings": diagnostic_dicts,
                        "candidate_id": candidate.id,
                        "review": review.as_dict(),
                    },
                )
                result = PlannerResult.proposed(
                    proposal=proposal,
                    raw_text=last_raw,
                    metrics=last_metrics,
                )
                yield self._event(
                    agent,
                    task,
                    "planning.final_plan.generated",
                    f"Final ActionPlan generated: {plan.title}",
                    {
                        "round": round_number,
                        "action_plan": plan.as_dict(),
                        "candidate_id": candidate.id,
                        "validation_warnings": diagnostic_dicts,
                    },
                )
                yield self._event(
                    agent,
                    task,
                    "approval.policy",
                    "Approval policy evaluated for final plan",
                    {
                        "requirements": [
                            requirement.as_dict()
                            for requirement in proposal.approval_requirements
                        ],
                    },
                )
                if emit_plan_proposed:
                    yield self._event(
                        agent,
                        task,
                        "plan.proposed",
                        f"Plan proposed: {proposal.title}",
                        {
                            "proposal": proposal.as_dict(),
                            "approval_requirements": [
                                requirement.as_dict()
                                for requirement in proposal.approval_requirements
                            ],
                        },
                    )
                return result

            assert decision.exploration is not None
            if len(rounds) >= self.limits.max_rounds:
                result = PlannerResult.failed(
                    error=(
                        "planner requested more exploration after "
                        f"max_rounds={self.limits.max_rounds}"
                    ),
                    raw_text=last_raw,
                    metrics=last_metrics,
                )
                yield self._planning_failed(
                    agent,
                    task,
                    result.error,
                    round_number,
                    rounds,
                )
                return result
            plan = decision.exploration
            yield self._event(
                agent,
                task,
                "exploration.plan.generated",
                f"Read-only exploration plan generated for round {round_number}",
                {"round": round_number, "plan": plan.as_dict()},
            )
            try:
                explorer.validate(plan)
                observations = []
                observation_bytes = 0
                for action in plan.actions:
                    yield self._event(
                        agent,
                        task,
                        "exploration.action.started",
                        f"Exploration action started: {action.action_type}",
                        {
                            "round": round_number,
                            "action_id": action.id,
                            "action_type": action.action_type,
                        },
                    )
                    observation = explorer.execute_action(action)
                    size = explorer.observation_size(observation)
                    explorer.validate_observation_budget(
                        round_bytes=observation_bytes,
                        observation_bytes=size,
                        observations=observations,
                    )
                    observations.append(observation)
                    observation_bytes += size
                    event_type = (
                        "exploration.action.completed"
                        if observation.status == "ok"
                        else "exploration.action.failed"
                    )
                    yield self._event(
                        agent,
                        task,
                        event_type,
                        f"Exploration action {observation.status}: {observation.action_type}",
                        {
                            "round": round_number,
                            "action_id": observation.action_id,
                            "action_type": observation.action_type,
                            "status": observation.status,
                            "truncated": observation.truncated,
                            "error": observation.error,
                        },
                    )
                explorer.commit_round(plan, observation_bytes=observation_bytes)
            except (ExplorationError, ValueError) as exc:
                result = PlannerResult.failed(
                    error=str(exc),
                    raw_text=last_raw,
                    metrics=last_metrics,
                )
                partial = (
                    [item.as_dict() for item in exc.observations]
                    if isinstance(exc, ExplorationBudgetError)
                    else []
                )
                yield self._planning_failed(
                    agent,
                    task,
                    result.error,
                    round_number,
                    rounds,
                    partial_observations=partial,
                )
                return result

            completed_round = ExplorationRound(
                number=round_number,
                plan=plan,
                observations=tuple(observations),
                observation_bytes=observation_bytes,
            )
            rounds.append(completed_round)
            yield self._event(
                agent,
                task,
                "exploration.observations.ready",
                f"Exploration observations ready for round {round_number}",
                {
                    "round": round_number,
                    "observations": [
                        observation.as_dict() for observation in observations
                    ],
                    "observation_bytes": observation_bytes,
                },
            )
            yield self._event(
                agent,
                task,
                "exploration.round.completed",
                f"Planning round {round_number} completed",
                {
                    "round": round_number,
                    "actions": len(plan.actions),
                    "observation_bytes": observation_bytes,
                },
            )
            yield self._event(
                agent,
                task,
                "replan.started",
                "Replanning with bounded workspace observations",
                {
                    "next_round": round_number + 1,
                    "exploration_rounds_remaining": max(
                        0,
                        self.limits.max_rounds - len(rounds),
                    ),
                },
            )

        raise AssertionError("unreachable planning state")

    def _generate_structured(
        self,
        agent,
        task: Task,
        prompt: str,
        *,
        phase: str,
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
    ) -> Iterator[AgentEvent]:
        planning_session = agent.runtime.create_session(
            system_prompt=agent.session.system_prompt,
        )
        requested = self.phase_budgets[phase]
        caller_cap = options.get("max_tokens")
        if isinstance(caller_cap, int) and not isinstance(caller_cap, bool):
            requested = min(requested, caller_cap)
        selected_options = dict(options)
        selected_options["max_tokens"] = requested
        round_options, prompt_tokens = self._round_generation_options(
            agent,
            planning_session,
            prompt,
            selected_options,
            minimum_tokens=self.minimum_phase_tokens[phase],
        )
        yield self._event(
            agent,
            task,
            "planning.generation_budget",
            f"Generation budget selected for {phase}",
            {
                "round": round_number,
                "budget_kind": phase,
                "prompt_tokens": prompt_tokens,
                "requested_max_tokens": requested,
                "effective_max_tokens": round_options["max_tokens"],
                "minimum_required_tokens": self.minimum_phase_tokens[phase],
                "context_tokens": self._context_limit(agent),
            },
        )
        metrics: GenerationMetrics | None = None
        if stream_model:
            raw_chunks: list[str] = []
            completed_text = ""
            model_failed: str | None = None
            for event in agent._stream_with_session(
                planning_session,
                prompt,
                task=task,
                **round_options,
            ):
                if event.event_type == "assistant.delta":
                    delta = event.payload.get("delta")
                    if isinstance(delta, str):
                        raw_chunks.append(delta)
                elif event.event_type == "assistant.completed":
                    text = event.payload.get("text")
                    if isinstance(text, str):
                        completed_text = text
                    metrics = _metrics_from_payload(event.payload.get("metrics"))
                elif event.event_type == "assistant.failed":
                    model_failed = str(event.payload.get("error") or "model stream failed")
                yield event
            raw_text = (completed_text or "".join(raw_chunks)).strip()
            if model_failed:
                raise ValueError(model_failed)
        else:
            generated = agent._ask_with_session(
                planning_session,
                prompt,
                **round_options,
            )
            raw_text = generated.text.strip()
            metrics = generated.metrics
        yield self._event(
            agent,
            task,
            "planner.raw_output",
            f"Visible model output captured for {phase}",
            {
                "round": round_number,
                "phase": phase,
                "text": raw_text,
                "content_kind": "visible_model_text",
            },
        )
        return raw_text, metrics

    def _recover_planning_decision(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        rounds: list[ExplorationRound],
        malformed_output: str,
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
    ) -> Iterator[AgentEvent]:
        yield self._event(
            agent,
            task,
            "planning.format_recovery.started",
            "One bounded planning format recovery started",
            {"round": round_number},
        )
        prompt = build_recovery_prompt(
            expected_schema=(
                '{"phase":"explore","summary":"...","actions":[...]} OR '
                + FINAL_RESPONSE_SCHEMA
                + ' OR {"phase":"cannot_plan","reason":"..."}'
            ),
            task_context=self._task_context(task, instruction),
            observations=self._observation_context(rounds),
            malformed_output=_bounded_text(malformed_output),
        )
        try:
            raw, metrics = yield from self._generate_structured(
                agent,
                task,
                prompt,
                phase="format_recovery",
                round_number=round_number,
                options=options,
                stream_model=stream_model,
            )
            parsed = parse_json_object(raw)
            decision = PlanningDecision.from_dict(parsed, limits=self.limits)
        except Exception as exc:
            yield self._event(
                agent,
                task,
                "planning.format_recovery.completed",
                "Planning format recovery failed",
                {"round": round_number, "status": "failed", "error": str(exc)},
            )
            return None
        yield self._event(
            agent,
            task,
            "planning.format_recovery.completed",
            "Planning format recovery completed",
            {"round": round_number, "status": "recovered"},
        )
        return parsed, decision, raw, metrics

    def _review_and_maybe_revise(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        rounds: list[ExplorationRound],
        candidate: FinalCandidate,
        candidate_raw: str,
        candidate_metrics: GenerationMetrics | None,
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
    ) -> Iterator[AgentEvent]:
        review, review_raw, review_metrics = yield from self._review_candidate(
            agent,
            task,
            instruction=instruction,
            rounds=rounds,
            candidate=candidate,
            round_number=round_number,
            options=options,
            stream_model=stream_model,
            phase="review",
        )
        if review is None or review.verdict == "cannot_verify":
            return None
        if review.verdict == "accept":
            yield self._candidate_accepted(agent, task, candidate, review, revised=False)
            return candidate, review, candidate_raw, candidate_metrics

        yield self._event(
            agent,
            task,
            "planning.revision.started",
            "Single candidate revision started",
            {
                "round": round_number,
                "candidate_id": candidate.id,
                "review": review.as_dict(),
            },
        )
        revision_prompt = build_revision_prompt(
            task_context=self._task_context(task, instruction),
            observations=self._observation_context(rounds),
            candidate=candidate,
            review=review,
            final_schema=FINAL_RESPONSE_SCHEMA,
            check_names=self.check_names,
        )
        try:
            raw, metrics = yield from self._generate_structured(
                agent,
                task,
                revision_prompt,
                phase="revision",
                round_number=round_number,
                options=options,
                stream_model=stream_model,
            )
            parsed, decision, raw, metrics = yield from self._parse_with_recovery(
                agent,
                task,
                raw=raw,
                metrics=metrics,
                expected_schema=FINAL_RESPONSE_SCHEMA,
                task_context=self._task_context(task, instruction),
                observations=self._observation_context(rounds),
                round_number=round_number,
                options=options,
                stream_model=stream_model,
            )
            if decision.phase != PlanningPhase.FINAL:
                raise ValueError("candidate revision must return phase=final")
            _validate_action_plan_mapping(decision.final_plan or {})
            revised_plan = ActionPlan.from_dict(decision.final_plan or {})
            diagnostics = self._validate_final_plan(
                agent,
                revised_plan,
                explored=bool(rounds),
                rounds=rounds,
                instruction=instruction,
            )
            revised = FinalCandidate(revised_plan, tuple(diagnostics))
        except Exception as exc:
            yield self._event(
                agent,
                task,
                "planning.revision.completed",
                "Candidate revision failed",
                {"round": round_number, "status": "failed", "error": str(exc)},
            )
            return None
        yield self._event(
            agent,
            task,
            "planning.revision.completed",
            "Candidate revision completed",
            {
                "round": round_number,
                "status": "completed",
                "previous_candidate_id": candidate.id,
                "candidate": revised.as_dict(),
            },
        )
        final_review, review_raw, review_metrics = yield from self._review_candidate(
            agent,
            task,
            instruction=instruction,
            rounds=rounds,
            candidate=revised,
            round_number=round_number,
            options=options,
            stream_model=stream_model,
            phase="final_review",
        )
        if final_review is None or final_review.verdict != "accept":
            return None
        yield self._candidate_accepted(agent, task, revised, final_review, revised=True)
        return revised, final_review, raw, metrics

    def _review_candidate(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        rounds: list[ExplorationRound],
        candidate: FinalCandidate,
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
        phase: str,
    ) -> Iterator[AgentEvent]:
        yield self._event(
            agent,
            task,
            "planning.review.started",
            "Independent candidate review started",
            {"round": round_number, "candidate_id": candidate.id, "review_phase": phase},
        )
        prompt = build_review_prompt(
            task_context=self._task_context(task, instruction),
            observations=self._observation_context(rounds),
            candidate=candidate,
            check_names=self.check_names,
        )
        try:
            raw, metrics = yield from self._generate_structured(
                agent,
                task,
                prompt,
                phase=phase,
                round_number=round_number,
                options=options,
                stream_model=stream_model,
            )
            parsed, review, raw, metrics = yield from self._parse_review_with_recovery(
                agent,
                task,
                raw=raw,
                metrics=metrics,
                instruction=instruction,
                rounds=rounds,
                round_number=round_number,
                options=options,
                stream_model=stream_model,
            )
        except Exception as exc:
            yield self._event(
                agent,
                task,
                "planning.review.completed",
                "Independent candidate review failed",
                {
                    "round": round_number,
                    "candidate_id": candidate.id,
                    "review_phase": phase,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            return None, "", None
        yield self._event(
            agent,
            task,
            "planning.review.completed",
            "Independent candidate review completed",
            {
                "round": round_number,
                "candidate_id": candidate.id,
                "review_phase": phase,
                "status": "completed",
                "review": review.as_dict(),
            },
        )
        return review, raw, metrics

    def _parse_with_recovery(
        self,
        agent,
        task: Task,
        *,
        raw: str,
        metrics: GenerationMetrics | None,
        expected_schema: str,
        task_context: dict[str, Any],
        observations: list[dict[str, Any]],
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
    ) -> Iterator[AgentEvent]:
        try:
            parsed = parse_json_object(raw)
            decision = PlanningDecision.from_dict(parsed, limits=self.limits)
            return parsed, decision, raw, metrics
        except Exception as first_error:
            yield self._event(
                agent,
                task,
                "planning.final_candidate.parse_failed",
                "Structured finalization response could not be parsed",
                {
                    "round": round_number,
                    "error": str(first_error),
                    "malformed_output": _bounded_text(raw),
                },
            )
        yield self._event(
            agent,
            task,
            "planning.format_recovery.started",
            "One bounded finalization format recovery started",
            {"round": round_number},
        )
        prompt = build_recovery_prompt(
            expected_schema=expected_schema,
            task_context=task_context,
            observations=observations,
            malformed_output=_bounded_text(raw),
        )
        recovered_raw, recovered_metrics = yield from self._generate_structured(
            agent,
            task,
            prompt,
            phase="format_recovery",
            round_number=round_number,
            options=options,
            stream_model=stream_model,
        )
        try:
            parsed = parse_json_object(recovered_raw)
            decision = PlanningDecision.from_dict(parsed, limits=self.limits)
        except Exception as exc:
            yield self._event(
                agent,
                task,
                "planning.format_recovery.completed",
                "Finalization format recovery failed",
                {"round": round_number, "status": "failed", "error": str(exc)},
            )
            raise ValueError(str(exc)) from exc
        yield self._event(
            agent,
            task,
            "planning.format_recovery.completed",
            "Finalization format recovery completed",
            {"round": round_number, "status": "recovered"},
        )
        return parsed, decision, recovered_raw, recovered_metrics

    def _parse_review_with_recovery(
        self,
        agent,
        task: Task,
        *,
        raw: str,
        metrics: GenerationMetrics | None,
        instruction: str,
        rounds: list[ExplorationRound],
        round_number: int,
        options: dict[str, Any],
        stream_model: bool,
    ) -> Iterator[AgentEvent]:
        try:
            parsed = parse_json_object(raw)
            return parsed, CandidateReview.from_dict(parsed), raw, metrics
        except Exception as first_error:
            yield self._event(
                agent,
                task,
                "planning.final_candidate.parse_failed",
                "Structured review response could not be parsed",
                {
                    "round": round_number,
                    "error": str(first_error),
                    "malformed_output": _bounded_text(raw),
                },
            )
        yield self._event(
            agent,
            task,
            "planning.format_recovery.started",
            "One bounded review format recovery started",
            {"round": round_number},
        )
        prompt = build_recovery_prompt(
            expected_schema=REVIEW_RESPONSE_SCHEMA,
            task_context=self._task_context(task, instruction),
            observations=self._observation_context(rounds),
            malformed_output=_bounded_text(raw),
        )
        recovered_raw, recovered_metrics = yield from self._generate_structured(
            agent,
            task,
            prompt,
            phase="format_recovery",
            round_number=round_number,
            options=options,
            stream_model=stream_model,
        )
        try:
            parsed = parse_json_object(recovered_raw)
            review = CandidateReview.from_dict(parsed)
        except Exception as exc:
            yield self._event(
                agent,
                task,
                "planning.format_recovery.completed",
                "Review format recovery failed",
                {"round": round_number, "status": "failed", "error": str(exc)},
            )
            raise ValueError(str(exc)) from exc
        yield self._event(
            agent,
            task,
            "planning.format_recovery.completed",
            "Review format recovery completed",
            {"round": round_number, "status": "recovered"},
        )
        return parsed, review, recovered_raw, recovered_metrics

    def _candidate_accepted(
        self,
        agent,
        task: Task,
        candidate: FinalCandidate,
        review: CandidateReview,
        *,
        revised: bool,
    ) -> AgentEvent:
        return self._event(
            agent,
            task,
            "planning.final_candidate.accepted",
            "Final candidate accepted by independent review",
            {
                "candidate_id": candidate.id,
                "revised": revised,
                "review": review.as_dict(),
            },
        )

    @staticmethod
    def _task_context(task: Task, instruction: str) -> dict[str, Any]:
        return {"id": task.id, "title": task.title, "instruction": instruction}

    @staticmethod
    def _observation_context(rounds: list[ExplorationRound]) -> list[dict[str, Any]]:
        return [round_.as_dict() for round_ in rounds]

    def build_prompt(
        self,
        agent,
        task: Task,
        *,
        instruction: str,
        rounds: tuple[ExplorationRound, ...],
        round_number: int,
        actions_used: int,
        observation_bytes_used: int,
    ) -> str:
        check_names = ", ".join(self.check_names) or "(none configured)"
        run_check_schema = (
            "- run_check: "
            '{"type":"run_check","check":"configured-name"} '
            f"(configured names: {check_names})"
        )
        remaining = {
            "rounds": max(0, self.limits.max_rounds - round_number + 1),
            "actions": max(0, self.limits.max_total_actions - actions_used),
            "observation_bytes": max(
                0,
                self.limits.max_total_observation_bytes - observation_bytes_used,
            ),
        }
        workspace = {
            "root": str(agent.workspace.root),
            "top_level": _top_level_listing(agent),
            "git_status": agent.git.status().stdout if agent.git.is_repo() else "",
        }
        system = ITERATIVE_SYSTEM_PROMPT.replace(
            "{run_check_schema}",
            run_check_schema,
        )
        context = {
            "task": {
                "id": task.id,
                "title": task.title,
                "instruction": instruction,
            },
            "workspace": workspace,
            "previous_observations": [
                {
                    "round": round_.number,
                    "observations": [
                        {
                            "action_type": observation.action_type,
                            "status": observation.status,
                            "data": observation.data,
                            "error": observation.error,
                            "truncated": observation.truncated,
                        }
                        for observation in round_.observations
                    ],
                }
                for round_ in rounds
            ],
            "remaining_budget": remaining,
            "terminal_decision_required": round_number > self.limits.max_rounds,
        }
        label = "Immutable task context for this stateless planning round:"
        prompt = (
            system
            + "\n\n"
            + label
            + "\n"
            + json.dumps(context, sort_keys=True, separators=(",", ":"))
        )
        if round_number > self.limits.max_rounds:
            prompt += (
                "\n\nNo exploration rounds remain. The phase MUST be \"final\" "
                "or \"cannot_plan\". Do not return \"explore\"."
            )
        return prompt

    def _validate_final_plan(
        self,
        agent,
        plan: ActionPlan,
        *,
        explored: bool,
        rounds: list[ExplorationRound],
        instruction: str,
    ) -> list[PlanDiagnostic]:
        if not plan.actions:
            raise ValueError("final action plan must contain at least one action")
        if explored and all(isinstance(action, ReadFileAction) for action in plan.actions):
            raise ValueError("FINAL response is still an exploration-only read plan")

        diagnostics: list[PlanDiagnostic] = []
        explored_paths = {
            str(observation.data.get("path"))
            for round_ in rounds
            for observation in round_.observations
            if isinstance(observation.data.get("path"), str)
        }
        has_effect = False
        configured_actions: set[str] = set()
        for index, action in enumerate(plan.actions):
            if isinstance(action, ReadFileAction):
                path = agent.workspace._resolve(action.path)
                if path.exists() and path.is_dir():
                    raise ValueError(f"final read_file target is a directory: {action.path}")
                if str(action.path) in explored_paths:
                    diagnostics.append(
                        PlanDiagnostic(
                            code="redundant_exploration_read",
                            message=f"read_file repeats explored path: {action.path}",
                            action_index=index,
                        )
                    )
            elif isinstance(action, ReplaceTextAction):
                path = agent.workspace._resolve(action.path)
                if path.exists() and path.is_dir():
                    raise ValueError(f"final replace_text target is a directory: {action.path}")
                self._diagnose_payload(
                    diagnostics,
                    index,
                    len(action.old.encode("utf-8")) + len(action.new.encode("utf-8")),
                )
                has_effect = True
            elif isinstance(action, WriteFileAction):
                path = agent.workspace._resolve(action.path)
                if path.exists() and path.is_dir():
                    raise ValueError(f"final write_file target is a directory: {action.path}")
                if path.exists():
                    if self.forbid_existing_file_write:
                        raise ValueError(
                            f"whole-file replacement is forbidden for existing file: {action.path}"
                        )
                    diagnostics.append(
                        PlanDiagnostic(
                            code="existing_file_write",
                            message=(
                                "write_file replaces existing file; prefer compact replace_text "
                                f"when practical: {action.path}"
                            ),
                            action_index=index,
                        )
                    )
                self._diagnose_payload(
                    diagnostics,
                    index,
                    len(action.content.encode("utf-8")),
                )
                has_effect = True
            elif isinstance(action, RunCheckAction):
                if action.check not in self.check_names:
                    raise ValueError(f"unknown configured check: {action.check}")
                configured_actions.add(action.check)
                has_effect = True
            else:
                serialized = action_to_dict(action)
                path = serialized.get("path")
                if isinstance(path, str):
                    agent.workspace._resolve(path)
                if action.action_type in {
                    "write_file",
                    "create_checkpoint",
                }:
                    has_effect = True
        if not has_effect:
            diagnostics.append(
                PlanDiagnostic(
                    code="no_effect",
                    message="final plan has no mutating action or configured check",
                )
            )
        lowered_instruction = instruction.lower()
        for check_name in self.check_names:
            explicit = f"configured {check_name.lower()} check"
            if explicit in lowered_instruction and check_name not in configured_actions:
                diagnostics.append(
                    PlanDiagnostic(
                        code="missing_explicit_check",
                        message=f"explicitly requested configured check is missing: {check_name}",
                        severity="major",
                    )
                )
        return diagnostics

    def _diagnose_payload(
        self,
        diagnostics: list[PlanDiagnostic],
        action_index: int,
        payload_bytes: int,
    ) -> None:
        if payload_bytes > self.max_action_payload_bytes:
            diagnostics.append(
                PlanDiagnostic(
                    code="large_action_payload",
                    message=(
                        f"embedded action payload is {payload_bytes} bytes; "
                        f"configured warning threshold is {self.max_action_payload_bytes}"
                    ),
                    action_index=action_index,
                )
            )

    def _remaining_budget(
        self,
        explorer: WorkspaceExplorer,
        round_number: int,
    ) -> dict[str, int]:
        return {
            "rounds": max(0, self.limits.max_rounds - round_number + 1),
            "actions": max(0, self.limits.max_total_actions - explorer.total_actions),
            "observation_bytes": max(
                0,
                self.limits.max_total_observation_bytes
                - explorer.total_observation_bytes,
            ),
        }

    def _round_generation_options(
        self,
        agent,
        planning_session,
        prompt: str,
        options: dict[str, Any],
        *,
        minimum_tokens: int,
    ) -> tuple[dict[str, Any], int | None]:
        selected = dict(options)
        requested = selected.get("max_tokens", self.max_tokens)
        if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
            raise ValueError("planner max_tokens must be a positive integer")
        if requested < minimum_tokens:
            raise ValueError(
                f"configured output budget {requested} is below safe minimum {minimum_tokens}"
            )
        context_limit = self._context_limit(agent)
        if context_limit is None:
            return selected, None
        messages = planning_session.transcript()
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = agent.runtime.tokenize(messages)
        available = context_limit - prompt_tokens - 32
        if available < minimum_tokens:
            raise ValueError(
                "insufficient model context for safe structured response: "
                f"prompt_tokens={prompt_tokens}, context_tokens={context_limit}, "
                f"minimum_output_tokens={minimum_tokens}"
            )
        selected["max_tokens"] = min(requested, available)
        return selected, prompt_tokens

    @staticmethod
    def _context_limit(agent) -> int | None:
        context = agent.lab.config.get("context", {})
        if not isinstance(context, dict):
            return None
        value = context.get("max_context_tokens")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return None
        return value

    def _planning_failed(
        self,
        agent,
        task: Task,
        error: str | None,
        round_number: int,
        rounds: list[ExplorationRound],
        partial_observations: list[dict[str, Any]] | None = None,
    ) -> AgentEvent:
        return self._event(
            agent,
            task,
            "planning.failed",
            "Workspace planning failed",
            {
                "error": error or "unknown planning failure",
                "round": round_number,
                "completed_rounds": [item.as_dict() for item in rounds],
                "partial_observations": partial_observations or [],
            },
        )

    @staticmethod
    def _event(
        agent,
        task: Task,
        event_type: str,
        summary: str,
        payload: dict[str, Any],
    ) -> AgentEvent:
        event = AgentEvent(
            event_type=event_type,
            summary=summary,
            task_id=task.id,
            session_id=agent.session.id,
            payload=payload,
        )
        agent._emit_existing_event(event)
        return event


def _consume_result(iterator: Iterator[AgentEvent]) -> PlannerResult:
    while True:
        try:
            next(iterator)
        except StopIteration as stop:
            result = stop.value
            if not isinstance(result, PlannerResult):
                raise RuntimeError("iterative planner ended without PlannerResult")
            return result


def _top_level_listing(agent) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for name in agent.workspace.list("."):
        path = agent.workspace.root / name
        kind = "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file"
        entries.append({"path": name, "kind": kind})
    return entries


def _validate_action_plan_mapping(data: dict[str, Any]) -> None:
    unknown = sorted(set(data) - {"id", "title", "description", "actions", "metadata"})
    if unknown:
        raise ValueError("unknown final ActionPlan field(s): " + ", ".join(unknown))


def _metrics_from_payload(data: Any) -> GenerationMetrics | None:
    if not isinstance(data, dict):
        return None
    try:
        return GenerationMetrics(
            prompt_tokens=int(data["prompt_tokens"]),
            generated_tokens=int(data["generated_tokens"]),
            ttft_sec=data.get("ttft_sec"),
            tokens_per_sec=float(data["tokens_per_sec"]),
            wall_sec=float(data["wall_sec"]),
        )
    except Exception:
        return None


def _bounded_text(text: str, limit: int = 2048) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated by AgentCore diagnostics]"
