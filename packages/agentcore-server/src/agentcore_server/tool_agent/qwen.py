from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Any

from agentcore_server.agents import Agent
from agentcore_server.generation import AssistantTurn, ToolCall, ToolResult
from agentcore_server.tasks import Task, TaskStatus
from agentcore_server.tool_agent.models import (
    ToolAgentLimits,
    ToolRunResult,
    ToolApprovalGateway,
    ToolApprovalRequest,
    ToolSteeringInbox,
)
from agentcore_server.tool_agent.tools import (
    ToolRegistry,
    ToolSafetyViolation,
    ValidatedToolCall,
    encode_tool_result,
)
from agentcore_server.tool_agent.protocols import TOOL_AGENT_SYSTEM_PROMPT


QWEN_TOOL_AGENT_SYSTEM_PROMPT = TOOL_AGENT_SYSTEM_PROMPT

_REPLAYABLE_READ_ONLY_TOOLS = frozenset({
    "list_directory",
    "search_files",
    "read_file",
    "git_status",
    "git_diff",
})
_DISCOVERY_TOOLS = frozenset({"list_directory", "search_files", "read_file"})


@dataclass
class _TurnProgress:
    turn: int
    tool_calls: int = 0
    discovery_calls: int = 0
    repeated_discovery_calls: int = 0
    repeated_malformed_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class _ProgressTracker:
    """Track objective recent tool activity for one bounded runway decision."""

    def __init__(self, window: int, initial_workspace_state: str | None) -> None:
        self.window = window
        self.turns: deque[_TurnProgress] = deque(maxlen=window)
        self._seen_discovery: set[str] = set()
        self._seen_malformed: set[str] = set()
        self._rejected: set[str] = set()
        self._failed_checks: dict[str, int] = {}
        self._latest_mutation_turn = -1
        self._workspace_state = initial_workspace_state

    def begin_turn(self, turn: int, calls: tuple[ToolCall, ...]) -> None:
        activity = _TurnProgress(turn=turn, tool_calls=len(calls))
        for call in calls:
            if call.parsing_error:
                signature = self._call_signature(call)
                if signature in self._seen_malformed:
                    activity.repeated_malformed_calls += 1
                self._seen_malformed.add(signature)
            if call.function_name not in _DISCOVERY_TOOLS:
                continue
            activity.discovery_calls += 1
            signature = self._call_signature(call)
            if signature in self._seen_discovery:
                activity.repeated_discovery_calls += 1
            self._seen_discovery.add(signature)
        self.turns.append(activity)

    def record_tool(
        self,
        *,
        turn: int,
        tool: str,
        success: bool,
        data: dict[str, Any],
        workspace_state_digest: str | None = None,
    ) -> None:
        activity = self._activity(turn)
        if success and tool in {"edit", "write_file"}:
            if (
                workspace_state_digest is not None
                and workspace_state_digest == self._workspace_state
            ):
                return
            self._workspace_state = workspace_state_digest
            self._latest_mutation_turn = turn
            activity.events.append({
                "turn": turn,
                "kind": "workspace_changed",
                "workspace_state_sha256": workspace_state_digest,
            })
            return
        if tool == "run_check":
            check = data.get("check") if isinstance(data.get("check"), dict) else {}
            name = str(check.get("name") or check.get("check") or "configured")
            if success:
                kind = "check_passed"
                failed_turn = self._failed_checks.get(name)
                if (
                    failed_turn is not None
                    and failed_turn < self._latest_mutation_turn <= turn
                ):
                    kind = "check_recovered"
                activity.events.append({"turn": turn, "kind": kind, "check": name})
            else:
                self._failed_checks[name] = turn
                activity.events.append({"turn": turn, "kind": "check_failed", "check": name})
            return
        if success and tool == "git_diff":
            activity.events.append({"turn": turn, "kind": "git_diff_inspected"})

    def record_rejection(self, turn: int, call: ToolCall) -> None:
        signature = self._call_signature(call)
        if signature in self._rejected:
            self._activity(turn).events.append({
                "turn": turn,
                "kind": "repeated_rejection",
            })
        self._rejected.add(signature)

    def runway_evidence(self) -> list[dict[str, Any]] | None:
        activities = list(self.turns)
        events = [event for activity in activities for event in activity.events]
        mutations = [event for event in events if event["kind"] == "workspace_changed"]
        validations = [
            event
            for event in events
            if event["kind"] in {"check_passed", "check_recovered", "git_diff_inspected"}
        ]
        if not mutations or not any(
            mutation["turn"] <= validation["turn"]
            for mutation in mutations
            for validation in validations
        ):
            return None
        repeated = sum(item.repeated_discovery_calls for item in activities)
        non_discovery = sum(item.tool_calls - item.discovery_calls for item in activities)
        if repeated > non_discovery:
            return None
        if any(item.repeated_malformed_calls for item in activities):
            return None
        if any(event["kind"] == "repeated_rejection" for event in events):
            return None
        return events

    def _activity(self, turn: int) -> _TurnProgress:
        if not self.turns or self.turns[-1].turn != turn:
            self.turns.append(_TurnProgress(turn=turn))
        return self.turns[-1]

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        return sha256(
            (call.function_name + "\0" + call.argument_text).encode("utf-8")
        ).hexdigest()


class ToolLoopAgent:
    """Persistent native tool loop with AgentCore workspace and approval safety."""

    def __init__(
        self,
        agent: Agent,
        *,
        approval_gateway: ToolApprovalGateway,
        limits: ToolAgentLimits | None = None,
        steering: ToolSteeringInbox | None = None,
    ) -> None:
        self.agent = agent
        self.approval_gateway = approval_gateway
        self.limits = limits or ToolAgentLimits()
        self.steering = steering or ToolSteeringInbox()
        self.registry = ToolRegistry(
            agent.workspace,
            default_read_lines=self.limits.default_read_lines,
            max_read_lines=self.limits.max_read_lines,
            max_directory_depth=self.limits.max_directory_depth,
            max_search_results=self.limits.max_search_results,
            max_edit_old_bytes=self.limits.max_edit_old_bytes,
            max_edit_new_bytes=self.limits.max_edit_new_bytes,
            max_write_file_bytes=self.limits.max_write_file_bytes,
            max_preview_bytes=self.limits.max_preview_bytes,
            max_changed_lines=self.limits.max_changed_lines,
        )

    def run(self, task: Task, instruction: str) -> ToolRunResult:
        if task.status != TaskStatus.CREATED:
            raise ValueError("native tool loop requires a created task")
        if task not in self.agent.tasks():
            raise ValueError("task is not owned by this agent")
        task.start()
        self.agent.session.add_user_message(instruction)
        protocol = getattr(self.agent.runtime, "tool_protocol", None)
        self._emit("agent.loop.started", "Native tool loop started", task, {
            "limits": self.limits.__dict__,
            "protocol": getattr(protocol, "name", "qwen"),
            "tools": list(self.registry.definitions),
        })
        results: list[ToolResult] = []
        turns = 0
        current_turn_limit = self.limits.max_model_turns
        absolute_turn_limit = (
            self.limits.max_model_turns + self.limits.completion_runway_turns
        )
        runway_granted = False
        progress = _ProgressTracker(
            self.limits.progress_window_turns,
            self._workspace_state_digest(),
        )
        self._last_context_preflight_healthy = True
        total_calls = 0
        total_result_bytes = 0
        consecutive_failures = 0
        total_rejections = 0
        consecutive_rejections = 0
        mutation_revision = 0
        successful_checks: dict[str, int] = {}
        git_diff_revision = -1
        required_checks, require_git_diff = self._completion_requirements(instruction)
        final_text = ""
        error: str | None = None

        try:
            while True:
                if turns >= current_turn_limit:
                    self._cancel_if_requested(task)
                    evidence = progress.runway_evidence()
                    runway_eligible = (
                        not runway_granted
                        and current_turn_limit == self.limits.max_model_turns
                        and self._last_context_preflight_healthy
                        and total_rejections < self.limits.max_rejected_side_effecting_calls
                        and consecutive_rejections
                        < self.limits.max_consecutive_rejected_side_effecting_calls
                        and evidence is not None
                    )
                    if runway_eligible:
                        runway_granted = True
                        current_turn_limit = absolute_turn_limit
                        self._emit(
                            "agent.turn_runway.granted",
                            "One bounded completion runway granted",
                            task,
                            {
                                "base_limit": self.limits.max_model_turns,
                                "runway_turns": self.limits.completion_runway_turns,
                                "absolute_limit": absolute_turn_limit,
                                "current_turn": turns,
                                "recent_progress_events": evidence,
                            },
                        )
                    else:
                        if runway_granted:
                            raise LoopLimitError(
                                f"absolute_model_turns={absolute_turn_limit} reached after one completion runway"
                            )
                        raise LoopLimitError(
                            f"max_model_turns={self.limits.max_model_turns} reached; "
                            "completion runway not eligible"
                        )
                self._cancel_if_requested(task)
                turns += 1
                self._emit("agent.turn.started", "Assistant tool turn started", task, {
                    "turn": turns,
                })
                turn = self._generate_turn(task, turns)
                self._emit("agent.turn.completed", "Assistant tool turn completed", task, {
                    "turn": turns,
                    "finish_reason": turn.finish_reason,
                    "tool_call_count": len(turn.tool_calls),
                    "metrics": turn.metrics.as_dict(),
                })
                progress.begin_turn(turns, turn.tool_calls)
                if not turn.tool_calls:
                    if not turn.text.strip():
                        raise RuntimeError("model returned neither tool calls nor a final response")
                    missing = self._missing_completion_requirements(
                        required_checks=required_checks,
                        require_git_diff=require_git_diff,
                        mutation_revision=mutation_revision,
                        successful_checks=successful_checks,
                        git_diff_revision=git_diff_revision,
                    )
                    if missing:
                        message = (
                            "Required task work remains before the final response: "
                            + "; ".join(missing)
                            + ". Use the native tools, then provide the final response."
                        )
                        self.agent.session.add_user_message(message)
                        self._emit(
                            "agent.completion.incomplete",
                            "Model attempted to finish before required work completed",
                            task,
                            {"turn": turns, "missing": missing},
                        )
                        continue
                    final_text = turn.text
                    task.complete()
                    self._emit("agent.final", "Native tool agent completed", task, {
                        "text": final_text,
                    })
                    break
                if len(turn.tool_calls) > self.limits.max_tool_calls_per_turn:
                    raise LoopLimitError(
                        f"max_tool_calls_per_turn={self.limits.max_tool_calls_per_turn} exceeded"
                    )
                if total_calls + len(turn.tool_calls) > self.limits.max_total_tool_calls:
                    raise LoopLimitError(
                        f"max_total_tool_calls={self.limits.max_total_tool_calls} exceeded"
                    )
                total_calls += len(turn.tool_calls)
                validated = self._validate_turn_calls(task, turn.tool_calls)
                for call, item, validation_error in validated:
                    self._cancel_if_requested(task)
                    remaining_result_bytes = (
                        self.limits.max_total_tool_result_bytes - total_result_bytes
                    )
                    if remaining_result_bytes < 256:
                        raise LoopLimitError(
                            "max_total_tool_result_bytes="
                            f"{self.limits.max_total_tool_result_bytes} reached"
                        )
                    if validation_error is not None:
                        success = False
                        data = {"success": False, "error": validation_error, "kind": "validation"}
                        self._emit("tool.validation.failed", "Tool call validation failed", task, {
                            "tool_call_id": call.id,
                            "tool": call.function_name,
                            "error": validation_error,
                        })
                    else:
                        assert item is not None
                        success, data = self._execute_one(task, item)
                        if data.get("rejected"):
                            progress.record_rejection(turns, call)
                            total_rejections += 1
                            consecutive_rejections += 1
                            consecutive_failures = 0
                            if (
                                total_rejections
                                >= self.limits.max_rejected_side_effecting_calls
                            ):
                                raise LoopLimitError(
                                    "max_rejected_side_effecting_calls="
                                    f"{self.limits.max_rejected_side_effecting_calls} reached"
                                )
                            if (
                                consecutive_rejections
                                >= self.limits.max_consecutive_rejected_side_effecting_calls
                            ):
                                raise LoopLimitError(
                                    "max_consecutive_rejected_side_effecting_calls="
                                    f"{self.limits.max_consecutive_rejected_side_effecting_calls} reached"
                                )
                        else:
                            consecutive_rejections = 0
                        if success and item.definition.name in {"edit", "write_file"}:
                            mutation_revision += 1
                        elif success and item.definition.name == "run_check":
                            successful_checks[item.arguments["check"]] = mutation_revision
                        elif success and item.definition.name == "git_diff":
                            git_diff_revision = mutation_revision
                        workspace_state_digest = None
                        if success and item.definition.name in {"edit", "write_file"}:
                            workspace_state_digest = self._workspace_state_digest()
                        progress.record_tool(
                            turn=turns,
                            tool=item.definition.name,
                            success=success,
                            data=data,
                            workspace_state_digest=workspace_state_digest,
                        )
                    result, result_bytes = self._result(
                        call,
                        success,
                        data,
                        max_bytes=min(
                            self.limits.max_single_tool_result_bytes,
                            remaining_result_bytes,
                        ),
                    )
                    total_result_bytes += result_bytes
                    self.agent.session.add_tool_result(result)
                    results.append(result)
                    if success:
                        consecutive_failures = 0
                    elif not data.get("rejected"):
                        consecutive_failures += 1
                        if consecutive_failures >= self.limits.max_consecutive_tool_failures:
                            raise LoopLimitError(
                                "max_consecutive_tool_failures="
                                f"{self.limits.max_consecutive_tool_failures} reached"
                            )
                steering = self.steering.take()
                if steering is not None:
                    self.agent.session.add_user_message(steering)
                    self._emit("agent.steering.queued", "Steering message appended", task, {
                        "message": steering,
                    })
        except ToolAgentCancelled as exc:
            error = str(exc)
            if task.status == TaskStatus.RUNNING:
                task.cancel(error)
        except LoopLimitError as exc:
            error = str(exc)
            self._emit("agent.loop.limit_reached", "Native tool loop limit reached", task, {
                "error": error,
            })
            if task.status == TaskStatus.RUNNING:
                task.fail(error)
        except (ToolSafetyViolation, RuntimeError) as exc:
            error = str(exc) or exc.__class__.__name__
            if task.status == TaskStatus.RUNNING:
                task.fail(error)
        except BaseException as exc:
            error = str(exc) or exc.__class__.__name__
            if task.status == TaskStatus.RUNNING:
                task.fail(error)
            raise
        finally:
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                report = task.report()
                self._emit("task.report", "Authoritative final task report captured", task, {
                    "report": report.as_dict(),
                })
                diff = self.agent.git.diff().stdout if self.agent.git.is_repo() else ""
                self._emit("git.diff", "Final Git diff captured", task, {"diff": diff})

        return ToolRunResult(
            status=task.status.value,
            final_text=final_text,
            turns=turns,
            tool_calls=total_calls,
            tool_results=tuple(results),
            report=task.report(),
            error=error,
            metadata={
                "runway_granted": runway_granted,
                "runway_turns_used": max(0, turns - self.limits.max_model_turns),
                "normal_turn_limit": self.limits.max_model_turns,
                "absolute_turn_limit": absolute_turn_limit,
            },
        )

    def _generate_turn(self, task: Task, turn_number: int) -> AssistantTurn:
        completed: AssistantTurn | None = None
        self._emit("assistant.started", "Assistant response started", task, {"turn": turn_number})
        try:
            while True:
                restart_after_compaction = False
                stream = self.agent.runtime.stream_tool_turn(
                    self.agent.session,
                    self.registry.schemas(),
                    context_safety_margin_tokens=self.limits.context_safety_margin_tokens,
                    minimum_output_tokens=self.limits.minimum_output_tokens,
                    **self.agent.generation_options,
                )
                for chunk in stream:
                    if chunk.chunk_type == "failed":
                        raise RuntimeError(chunk.error or "native tool turn failed")
                    if chunk.chunk_type == "started":
                        capacity = chunk.metadata
                        self._last_context_preflight_healthy = bool(
                            capacity.get("sufficient", True)
                        )
                        compacted = self._compact_context_if_needed(capacity)
                        event_type = (
                            "agent.context.preflight"
                            if capacity.get("sufficient") or compacted is not None
                            else "agent.context.insufficient"
                        )
                        if "effective_max_tokens" in capacity:
                            self._emit(
                                event_type,
                                "Native tool-turn context capacity checked",
                                task,
                                {**capacity, "turn": turn_number},
                            )
                        if compacted is not None:
                            self._emit(
                                "agent.context.compacted",
                                "Old replayable tool result compacted",
                                task,
                                {
                                    **compacted,
                                    "turn": turn_number,
                                    "prompt_tokens_before": capacity["exact_prompt_tokens"],
                                    "effective_max_tokens_before": capacity[
                                        "effective_max_tokens"
                                    ],
                                    "target_output_tokens": self._context_recovery_target(
                                        capacity
                                    ),
                                },
                            )
                            close = getattr(stream, "close", None)
                            if close is not None:
                                close()
                            restart_after_compaction = True
                            break
                    elif chunk.chunk_type == "text_delta" and chunk.text_delta:
                        self._emit("assistant.delta", "Assistant response delta", task, {
                            "delta": chunk.text_delta,
                            "turn": turn_number,
                        })
                    elif (
                        chunk.chunk_type == "tool_call_delta"
                        and chunk.tool_call_delta is not None
                    ):
                        # Raw fragments remain runtime data; lifecycle events are emitted after assembly.
                        continue
                    elif chunk.chunk_type == "completed":
                        completed = chunk.turn
                if not restart_after_compaction:
                    break
        except RuntimeError as exc:
            self._emit("agent.turn.failed", "Assistant tool turn failed", task, {
                "turn": turn_number,
                "error": str(exc) or exc.__class__.__name__,
                "error_type": exc.__class__.__name__,
            })
            raise
        if completed is None:
            raise RuntimeError("native tool turn did not complete")
        self._emit("assistant.completed", "Assistant response completed", task, {
            "text": completed.text,
            "turn": turn_number,
            "metrics": completed.metrics.as_dict(),
        })
        for call in completed.tool_calls:
            self._emit("tool.call.received", "Native tool call received", task, {
                "tool_call_id": call.id,
                "index": call.index,
                "tool": call.function_name,
                "argument_bytes": len(call.argument_text.encode("utf-8")),
                "argument_sha256": sha256(call.argument_text.encode("utf-8")).hexdigest(),
                "parsing_error": call.parsing_error,
            })
        return completed

    def _compact_context_if_needed(self, capacity: dict[str, Any]) -> dict[str, Any] | None:
        if "effective_max_tokens" not in capacity:
            return None
        configured = int(capacity.get("configured_max_tokens", 0))
        if configured < self.limits.minimum_output_tokens:
            return None
        if int(capacity["effective_max_tokens"]) >= self._context_recovery_target(capacity):
            return None
        return self.agent.session.compact_oldest_tool_result(
            replayable_tools=_REPLAYABLE_READ_ONLY_TOOLS,
            preserve_recent=self.limits.preserve_recent_tool_results,
        )

    def _context_recovery_target(self, capacity: dict[str, Any]) -> int:
        configured = int(capacity.get("configured_max_tokens", 0))
        return min(
            configured,
            max(
                self.limits.minimum_output_tokens,
                self.limits.context_recovery_target_tokens,
            ),
        )

    def _validate_turn_calls(
        self,
        task: Task,
        calls: tuple[ToolCall, ...],
    ) -> list[tuple[ToolCall, ValidatedToolCall | None, str | None]]:
        validated: list[tuple[ToolCall, ValidatedToolCall | None, str | None]] = []
        for call in sorted(calls, key=lambda item: item.index):
            try:
                item = self.registry.validate(call)
            except ToolSafetyViolation as exc:
                self._emit("tool.validation.failed", "Tool call violated workspace safety", task, {
                    "tool_call_id": call.id,
                    "tool": call.function_name,
                    "error": str(exc),
                    "fatal": True,
                })
                raise
            except ValueError as exc:
                validated.append((call, None, str(exc)))
            else:
                validated.append((call, item, None))
                self._emit("tool.call.validated", "Tool call validated", task, {
                    "tool_call_id": call.id,
                    "tool": call.function_name,
                    "target": item.target,
                })
        return validated

    def _execute_one(
        self,
        task: Task,
        validated: ValidatedToolCall,
    ) -> tuple[bool, dict[str, Any]]:
        call = validated.call
        preview = None
        if validated.definition.side_effecting:
            try:
                preview = self.registry.preview(validated)
            except ToolSafetyViolation:
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                data = {
                    "success": False,
                    "error": str(exc),
                    "kind": "tool_preview",
                    "tool_call_id": call.id,
                }
                self._emit("tool.preview.failed", "Tool effect preview failed", task, {
                    "tool_call_id": call.id,
                    "tool": call.function_name,
                    "target": validated.target,
                    "error": str(exc),
                })
                return False, data
            request = ToolApprovalRequest(
                call=call,
                target=validated.target,
                arguments=dict(validated.arguments),
                expected_side_effect=self.registry.expected_side_effect(validated),
                preview=preview,
            )
            self._emit("tool.preview.created", "Complete tool effect preview created", task, {
                "tool_call_id": call.id,
                "tool": call.function_name,
                "target": validated.target,
                "preview": preview.summary_dict(),
            })
            event_request = request.as_dict()
            event_request["arguments"] = self.registry.bounded_arguments(validated)
            self._emit(
                "tool.approval.required",
                "Explicit tool approval required",
                task,
                event_request,
            )
            decision = self.approval_gateway.request(request)
            if decision.tool_call_id != call.id or decision.preview_digest != preview.digest:
                raise ToolSafetyViolation(
                    "approval decision does not match the tool call and preview digest"
                )
            if not decision.approved:
                self._emit("tool.rejected", "Tool call rejected", task, {
                    "tool_call_id": call.id,
                    "tool": call.function_name,
                    "target": validated.target,
                    "preview_id": preview.preview_id,
                    "preview_digest": preview.digest,
                    "reason": decision.reason or "rejected by operator",
                })
                return False, {
                    "success": False,
                    "rejected": True,
                    "error": decision.reason or "rejected by operator",
                    "tool_call_id": call.id,
                }
            self._emit("tool.approved", "Tool call approved", task, {
                "tool_call_id": call.id,
                "tool": call.function_name,
                "target": validated.target,
                "preview_id": preview.preview_id,
                "preview_digest": preview.digest,
            })
            self._cancel_if_requested(task)
        self._emit("tool.started", "Tool execution started", task, {
            "tool_call_id": call.id,
            "tool": call.function_name,
            "target": validated.target,
            "preview_id": None if preview is None else preview.preview_id,
            "preview_digest": None if preview is None else preview.digest,
        })
        try:
            success, data = self.registry.execute(validated, preview=preview)
        except ToolSafetyViolation:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            success = False
            data = {"success": False, "error": str(exc), "kind": "tool_execution"}
        event_type = "tool.completed" if success else "tool.failed"
        self._emit(event_type, "Tool execution completed" if success else "Tool execution failed", task, {
            "tool_call_id": call.id,
            "tool": call.function_name,
            "target": validated.target,
            "success": success,
            "preview_id": None if preview is None else preview.preview_id,
            "preview_digest": None if preview is None else preview.digest,
            "result": self._bounded_event_data(data),
        })
        return success, data

    def _completion_requirements(self, instruction: str) -> tuple[tuple[str, ...], bool]:
        required: list[str] = []
        for name in self.agent.workspace.checks.names():
            escaped = re.escape(name)
            pattern = re.compile(
                rf"\b(?:run|execute)\b[^\n]{{0,80}}\b(?:the\s+)?(?:configured\s+)?{escaped}\s+check\b",
                re.IGNORECASE,
            )
            if pattern.search(instruction):
                required.append(name)
        require_git_diff = bool(
            re.search(
                r"\b(?:inspect|show|produce)\b[^\n]{0,80}\bgit\s+diff\b",
                instruction,
                re.IGNORECASE,
            )
        )
        return tuple(required), require_git_diff

    @staticmethod
    def _missing_completion_requirements(
        *,
        required_checks: tuple[str, ...],
        require_git_diff: bool,
        mutation_revision: int,
        successful_checks: dict[str, int],
        git_diff_revision: int,
    ) -> list[str]:
        missing = [
            f"configured check '{name}' has not passed after the latest mutation"
            for name in required_checks
            if successful_checks.get(name) != mutation_revision
        ]
        if require_git_diff and git_diff_revision != mutation_revision:
            missing.append("git_diff has not been inspected after the latest mutation")
        return missing

    def _result(
        self,
        call: ToolCall,
        success: bool,
        data: dict[str, Any],
        *,
        max_bytes: int,
    ) -> tuple[ToolResult, int]:
        content, truncated = encode_tool_result(
            data,
            max_bytes=max_bytes,
        )
        encoded_bytes = len(content.encode("utf-8"))
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.function_name,
            success=success,
            content=content,
            metadata={"bytes": encoded_bytes, "truncated": truncated},
        ), encoded_bytes

    @staticmethod
    def _bounded_event_data(data: dict[str, Any], limit: int = 4096) -> dict[str, Any]:
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) <= limit:
            return data
        return {
            "truncated": True,
            "bytes": len(encoded),
            "sha256": sha256(encoded).hexdigest(),
            "excerpt": encoded[:limit].decode("utf-8", errors="ignore"),
        }

    def _workspace_state_digest(self) -> str | None:
        if not self.agent.git.is_repo():
            return None
        status = self.agent.git.status()
        diff = self.agent.git.diff()
        if not status.ok or not diff.ok:
            return None
        return sha256(
            (status.stdout + "\0" + diff.stdout).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _cancel_if_requested(task: Task) -> None:
        if task.cancellation_requested:
            raise ToolAgentCancelled(task.cancellation_reason or "tool agent cancelled")

    def _emit(
        self,
        event_type: str,
        summary: str,
        task: Task,
        payload: dict[str, Any],
    ) -> None:
        self.agent.emit_event(event_type, summary, task=task, payload=payload)


class ToolAgentCancelled(RuntimeError):
    pass


QwenToolAgent = ToolLoopAgent


class LoopLimitError(RuntimeError):
    pass
