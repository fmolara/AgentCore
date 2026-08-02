from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable

from agentcore_server.generation import ToolCall
from agentcore_server.workspace import DiscoveryLimits, Workspace, WorkspaceDiscovery


class ToolSafetyViolation(RuntimeError):
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


class QwenToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        *,
        default_read_lines: int = 120,
        max_read_lines: int = 500,
        max_directory_depth: int = 4,
        max_search_results: int = 100,
    ) -> None:
        self.workspace = workspace
        self.default_read_lines = default_read_lines
        self.max_read_lines = max_read_lines
        self.max_directory_depth = max_directory_depth
        self.max_search_results = max_search_results
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

    def execute(self, validated: ValidatedToolCall) -> tuple[bool, dict[str, Any]]:
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
        if definition.name == "run_check" and arguments["check"] not in self.workspace.checks.names():
            raise ValueError(f"unknown configured check: {arguments['check']}")

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
