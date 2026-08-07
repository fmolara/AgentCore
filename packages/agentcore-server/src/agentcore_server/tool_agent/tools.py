from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import unified_diff
from hashlib import sha256
import json
from typing import Any, Callable

from agentcore_server.generation import ToolCall
from agentcore_server.tool_agent.models import ToolEffectPreview
from agentcore_server.workspace import DiscoveryLimits, Workspace, WorkspaceDiscovery


class ToolSafetyViolation(RuntimeError):
    pass


class ToolPayloadLimitError(ValueError):
    pass


class StaleToolPreviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    side_effecting: bool
    target_field: str | None = None

    def native_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                # Match SGLang's validated OpenAI Tool representation exactly.
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


@dataclass(frozen=True)
class ValidatedToolCall:
    call: ToolCall
    definition: ToolDefinition
    arguments: dict[str, Any]

    @property
    def target(self) -> str | None:
        if self.definition.target_field is None:
            return None
        value = self.arguments.get(self.definition.target_field)
        return value if isinstance(value, str) else None


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        *,
        default_read_lines: int = 120,
        max_read_lines: int = 500,
        max_directory_depth: int = 4,
        max_search_results: int = 100,
        max_edit_old_bytes: int = 64 * 1024,
        max_edit_new_bytes: int = 64 * 1024,
        max_write_file_bytes: int = 256 * 1024,
        max_preview_bytes: int = 128 * 1024,
        max_changed_lines: int = 1200,
    ) -> None:
        self.workspace = workspace
        self.default_read_lines = default_read_lines
        self.max_read_lines = max_read_lines
        self.max_directory_depth = max_directory_depth
        self.max_search_results = max_search_results
        self.max_edit_old_bytes = max_edit_old_bytes
        self.max_edit_new_bytes = max_edit_new_bytes
        self.max_write_file_bytes = max_write_file_bytes
        self.max_preview_bytes = max_preview_bytes
        self.max_changed_lines = max_changed_lines
        self.discovery = WorkspaceDiscovery(
            workspace,
            limits=DiscoveryLimits(
                max_directory_depth=max_directory_depth,
                max_files_returned=max_search_results,
                max_read_lines=max_read_lines,
            ),
        )
        string = {"type": "string"}
        integer = {"type": "integer"}
        boolean = {"type": "boolean"}
        self.definitions = {
            definition.name: definition
            for definition in (
                ToolDefinition(
                    "list_directory",
                    "List bounded workspace entries in deterministic order.",
                    {"path": string, "max_depth": integer, "include_hidden": boolean},
                    ("path",),
                    False,
                    "path",
                ),
                ToolDefinition(
                    "search_files",
                    "Search workspace file names and optionally bounded UTF-8 contents.",
                    {
                        "root": string,
                        "name_pattern": string,
                        "content_query": string,
                        "max_results": integer,
                    },
                    ("root",),
                    False,
                    "root",
                ),
                ToolDefinition(
                    "read_file",
                    "Read a bounded line range from one workspace file.",
                    {"path": string, "start_line": integer, "max_lines": integer},
                    ("path",),
                    False,
                    "path",
                ),
                ToolDefinition("git_status", "Show workspace Git status.", {}, (), False),
                ToolDefinition(
                    "git_diff",
                    "Show the current workspace Git diff, optionally for one path.",
                    {"path": string},
                    (),
                    False,
                    "path",
                ),
                ToolDefinition(
                    "edit",
                    "Replace exactly one unique occurrence in an existing file.",
                    {"path": string, "old": string, "new": string},
                    ("path", "old", "new"),
                    True,
                    "path",
                ),
                ToolDefinition(
                    "write_file",
                    "Write a complete workspace file; prefer this for new files.",
                    {"path": string, "content": string},
                    ("path", "content"),
                    True,
                    "path",
                ),
                ToolDefinition(
                    "run_check",
                    "Run one trusted configured symbolic workspace check.",
                    {"check": string},
                    ("check",),
                    True,
                    "check",
                ),
            )
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [self.definitions[name].native_schema() for name in self.definitions]

    def validate(self, call: ToolCall) -> ValidatedToolCall:
        if call.parsing_error is not None or call.arguments is None:
            raise ValueError(call.parsing_error or "tool arguments are unavailable")
        try:
            definition = self.definitions[call.function_name]
        except KeyError as exc:
            raise ValueError(f"unknown tool: {call.function_name}") from exc
        arguments = dict(call.arguments)
        allowed = set(definition.properties)
        unknown = sorted(set(arguments) - allowed)
        missing = sorted(set(definition.required) - set(arguments))
        if unknown:
            raise ValueError("unknown argument field(s): " + ", ".join(unknown))
        if missing:
            raise ValueError("missing required argument field(s): " + ", ".join(missing))
        for name, value in arguments.items():
            expected = definition.properties[name]["type"]
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"argument '{name}' must be a string")
            if expected == "integer" and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"argument '{name}' must be an integer")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"argument '{name}' must be a boolean")
        self._validate_values(definition, arguments)
        return ValidatedToolCall(call=call, definition=definition, arguments=arguments)

    def execute(
        self,
        validated: ValidatedToolCall,
        *,
        preview: ToolEffectPreview | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if validated.definition.side_effecting:
            if preview is None:
                raise ValueError("side-effecting tool execution requires an approval preview")
            try:
                current = self.preview(validated)
            except (OSError, RuntimeError, ValueError) as exc:
                raise StaleToolPreviewError(
                    f"approval preview is stale: workspace effect can no longer be reproduced: {exc}"
                ) from exc
            if current.digest != preview.digest:
                raise StaleToolPreviewError(
                    "approval preview is stale: workspace changed after approval preview"
                )
        handlers: dict[str, Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]]] = {
            "list_directory": self._list_directory,
            "search_files": self._search_files,
            "read_file": self._read_file,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "edit": self._edit,
            "write_file": self._write_file,
            "run_check": self._run_check,
        }
        return handlers[validated.definition.name](validated.arguments)

    def preview(self, validated: ValidatedToolCall) -> ToolEffectPreview:
        name = validated.definition.name
        if not validated.definition.side_effecting:
            raise ValueError(f"tool does not require an approval preview: {name}")
        if name in {"edit", "write_file", "run_check"}:
            self.workspace._require_writable()
        if name == "edit":
            return self._preview_edit(validated)
        if name == "write_file":
            return self._preview_write_file(validated)
        if name == "run_check":
            return self._preview_run_check(validated)
        raise ValueError(f"unsupported preview tool: {name}")

    def expected_side_effect(self, validated: ValidatedToolCall) -> str:
        return {
            "edit": "modify one existing workspace file by an exact unique replacement",
            "write_file": "write the complete contents of one workspace file",
            "run_check": "execute trusted workspace code using the configured symbolic check",
        }.get(validated.definition.name, "read workspace state")

    def bounded_arguments(self, validated: ValidatedToolCall, *, limit: int = 2000) -> dict[str, Any]:
        bounded: dict[str, Any] = {}
        for name, value in validated.arguments.items():
            if isinstance(value, str) and len(value) > limit:
                bounded[name] = {
                    "sha256": sha256(value.encode("utf-8")).hexdigest(),
                    "bytes": len(value.encode("utf-8")),
                    "excerpt": value[:limit],
                    "truncated": True,
                }
            else:
                bounded[name] = value
        return bounded

    def _validate_values(self, definition: ToolDefinition, arguments: dict[str, Any]) -> None:
        path_name = definition.target_field if definition.name != "run_check" else None
        if path_name is not None and path_name in arguments:
            try:
                self.workspace._resolve(arguments[path_name])
            except ValueError as exc:
                raise ToolSafetyViolation(str(exc)) from exc
        if definition.name == "read_file":
            start = arguments.get("start_line", 1)
            lines = arguments.get("max_lines", self.default_read_lines)
            if start <= 0:
                raise ValueError("start_line must be positive")
            if lines <= 0 or lines > self.max_read_lines:
                raise ValueError(f"max_lines must be between 1 and {self.max_read_lines}")
        if definition.name == "list_directory":
            depth = arguments.get("max_depth", 1)
            if depth < 0 or depth > self.max_directory_depth:
                raise ValueError(
                    f"max_depth must be between 0 and {self.max_directory_depth}"
                )
        if definition.name == "search_files":
            count = arguments.get("max_results", min(50, self.max_search_results))
            if count <= 0 or count > self.max_search_results:
                raise ValueError(f"max_results must be between 1 and {self.max_search_results}")
        if definition.name == "edit" and arguments["old"] == "":
            raise ValueError("edit old text must not be empty")
        if definition.name == "edit":
            old_bytes = len(arguments["old"].encode("utf-8"))
            new_bytes = len(arguments["new"].encode("utf-8"))
            if old_bytes > self.max_edit_old_bytes:
                raise ToolPayloadLimitError(
                    f"edit old text exceeds {self.max_edit_old_bytes} bytes; split the change into smaller exact edits"
                )
            if new_bytes > self.max_edit_new_bytes:
                raise ToolPayloadLimitError(
                    f"edit new text exceeds {self.max_edit_new_bytes} bytes; split the change into smaller exact edits"
                )
        if definition.name == "write_file":
            content_bytes = len(arguments["content"].encode("utf-8"))
            if content_bytes > self.max_write_file_bytes:
                raise ToolPayloadLimitError(
                    f"write_file content exceeds {self.max_write_file_bytes} bytes"
                )
        if definition.name == "run_check" and arguments["check"] not in self.workspace.checks.names():
            raise ValueError(f"unknown configured check: {arguments['check']}")

    def _preview_edit(self, validated: ValidatedToolCall) -> ToolEffectPreview:
        args = validated.arguments
        resolved = self.workspace._resolve(args["path"])
        if not resolved.exists():
            raise FileNotFoundError(f"file does not exist: {args['path']}")
        if not resolved.is_file():
            raise IsADirectoryError(f"path is not a file: {args['path']}")
        original = resolved.read_text(encoding="utf-8")
        matches = original.count(args["old"])
        display = resolved.relative_to(self.workspace.root).as_posix()
        if matches == 0:
            raise ValueError(f"exact edit text not found in {display}")
        if matches > 1:
            raise ValueError(f"exact edit is ambiguous in {display}: {matches} matches")
        updated = original.replace(args["old"], args["new"], 1)
        changed_lines = max(
            len(args["old"].splitlines()), len(args["new"].splitlines()), 1
        )
        changed_bytes = len(args["old"].encode("utf-8")) + len(
            args["new"].encode("utf-8")
        )
        return self._file_preview(
            validated,
            display=display,
            original=original,
            updated=updated,
            source_exists=True,
            match_count=1,
            changed_lines=changed_lines,
            changed_bytes=changed_bytes,
            effect_type="exact_unique_edit",
        )

    def _preview_write_file(self, validated: ValidatedToolCall) -> ToolEffectPreview:
        args = validated.arguments
        resolved = self.workspace._resolve(args["path"])
        if resolved.exists() and not resolved.is_file():
            raise IsADirectoryError(f"path is not a file: {args['path']}")
        source_exists = resolved.exists()
        original = resolved.read_text(encoding="utf-8") if source_exists else ""
        updated = args["content"]
        display = resolved.relative_to(self.workspace.root).as_posix()
        changed_lines = max(len(original.splitlines()), len(updated.splitlines()), 1)
        changed_bytes = len(original.encode("utf-8")) + len(updated.encode("utf-8"))
        return self._file_preview(
            validated,
            display=display,
            original=original,
            updated=updated,
            source_exists=source_exists,
            match_count=None,
            changed_lines=changed_lines,
            changed_bytes=changed_bytes,
            effect_type="replace_existing_file" if source_exists else "create_new_file",
        )

    def _file_preview(
        self,
        validated: ValidatedToolCall,
        *,
        display: str,
        original: str,
        updated: str,
        source_exists: bool,
        match_count: int | None,
        changed_lines: int,
        changed_bytes: int,
        effect_type: str,
    ) -> ToolEffectPreview:
        if changed_lines > self.max_changed_lines:
            raise ToolPayloadLimitError(
                f"tool change affects {changed_lines} lines, exceeding {self.max_changed_lines}; split the change into smaller exact edits"
            )
        from_name = f"a/{display}" if source_exists else "/dev/null"
        content = "".join(
            unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=from_name,
                tofile=f"b/{display}",
            )
        )
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > self.max_preview_bytes:
            raise ToolPayloadLimitError(
                f"prospective diff exceeds {self.max_preview_bytes} bytes; split the change into smaller exact edits"
            )
        source_sha = sha256(original.encode("utf-8")).hexdigest() if source_exists else None
        result_sha = sha256(updated.encode("utf-8")).hexdigest()
        return self._make_preview(
            validated,
            target=display,
            effect_type=effect_type,
            content=content,
            source_sha256=source_sha,
            result_sha256=result_sha,
            source_exists=source_exists,
            match_count=match_count,
            changed_bytes=changed_bytes,
            changed_lines=changed_lines,
            metadata={"preview_format": "unified_diff"},
        )

    def _preview_run_check(self, validated: ValidatedToolCall) -> ToolEffectPreview:
        name = validated.arguments["check"]
        definition = self.workspace.checks.definitions[name]
        content = json.dumps(
            {
                "check": name,
                "argv": list(definition.argv),
                "cwd": str(self.workspace.root),
                "timeout_sec": definition.timeout_sec,
                "max_output_bytes": definition.max_output_bytes,
                "shell": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return self._make_preview(
            validated,
            target=name,
            effect_type="trusted_symbolic_check",
            content=content,
            source_sha256=None,
            result_sha256=None,
            source_exists=None,
            match_count=None,
            changed_bytes=0,
            changed_lines=0,
            metadata={
                "argv": list(definition.argv),
                "cwd": str(self.workspace.root),
                "timeout_sec": definition.timeout_sec,
                "shell": False,
            },
        )

    @staticmethod
    def _make_preview(
        validated: ValidatedToolCall,
        *,
        target: str | None,
        effect_type: str,
        content: str,
        source_sha256: str | None,
        result_sha256: str | None,
        source_exists: bool | None,
        match_count: int | None,
        changed_bytes: int,
        changed_lines: int,
        metadata: dict[str, Any],
    ) -> ToolEffectPreview:
        content_sha = sha256(content.encode("utf-8")).hexdigest()
        digest_data = {
            "tool_call_id": validated.call.id,
            "tool": validated.definition.name,
            "target": target,
            "effect_type": effect_type,
            "content_sha256": content_sha,
            "source_sha256": source_sha256,
            "result_sha256": result_sha256,
            "source_exists": source_exists,
            "match_count": match_count,
            "changed_bytes": changed_bytes,
            "changed_lines": changed_lines,
            "metadata": metadata,
        }
        digest = sha256(
            json.dumps(digest_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ToolEffectPreview(
            preview_id=f"preview-{digest[:20]}",
            tool_call_id=validated.call.id,
            tool=validated.definition.name,
            target=target,
            effect_type=effect_type,
            content=content,
            content_sha256=content_sha,
            digest=digest,
            source_sha256=source_sha256,
            result_sha256=result_sha256,
            source_exists=source_exists,
            match_count=match_count,
            changed_bytes=changed_bytes,
            changed_lines=changed_lines,
            within_limits=True,
            metadata=metadata,
        )

    def _list_directory(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.discovery.list_directory(
            args["path"],
            max_depth=args.get("max_depth", 1),
            include_hidden=args.get("include_hidden", False),
        )
        return result.success, result.as_dict()

    def _search_files(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.discovery.search_files(
            args["root"],
            name_pattern=args.get("name_pattern", "*"),
            content_query=args.get("content_query"),
            max_results=args.get("max_results", min(50, self.max_search_results)),
        )
        return result.success, result.as_dict()

    def _read_file(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.discovery.read_file(
            args["path"],
            start_line=args.get("start_line", 1),
            max_lines=args.get("max_lines", self.default_read_lines),
        )
        return result.success, result.as_dict()

    def _git_status(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        del args
        result = self.workspace.git.status()
        return result.ok, {
            "success": result.ok,
            "returncode": result.returncode,
            "status": result.stdout,
            "stderr": result.stderr,
        }

    def _git_diff(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.workspace.git.diff(args.get("path"))
        return result.ok, {
            "success": result.ok,
            "path": args.get("path"),
            "returncode": result.returncode,
            "diff": result.stdout,
            "stderr": result.stderr,
        }

    def _edit(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.workspace.files.replace_text_unique(
            args["path"], args["old"], args["new"]
        )
        return True, {"success": True, "edit": asdict(result)}

    def _write_file(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.workspace.files.write_text(args["path"], args["content"])
        return True, {"success": True, "edit": asdict(result)}

    def _run_check(self, args: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        result = self.workspace.checks.run(args["check"])
        return result.ok, {"success": result.ok, "check": result.as_dict()}


QwenToolRegistry = ToolRegistry


def encode_tool_result(
    data: dict[str, Any],
    *,
    max_bytes: int,
) -> tuple[str, bool]:
    if max_bytes < 256:
        raise ValueError("max_bytes must be at least 256")
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content, False
    digest = sha256(encoded).hexdigest()
    excerpt_limit = max_bytes
    while excerpt_limit >= 0:
        excerpt = encoded[:excerpt_limit].decode("utf-8", errors="ignore")
        bounded = json.dumps(
            {
                "success": data.get("success", False),
                "result_truncated": True,
                "original_bytes": len(encoded),
                "sha256": digest,
                "excerpt": excerpt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        overflow = len(bounded.encode("utf-8")) - max_bytes
        if overflow <= 0:
            return bounded, True
        excerpt_limit -= max(overflow, 1)
    raise ValueError("tool result metadata does not fit configured byte limit")
