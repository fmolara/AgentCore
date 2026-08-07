from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "agentclient" / "config.toml"
DEFAULT_SYSTEM_PROMPT = "You are a concise coding assistant. Propose safe AgentCore ActionPlans only."


@dataclass(frozen=True)
class ClientConfig:
    server_url: str = "http://127.0.0.1:8080"
    default_workspace: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    color: bool = True
    request_timeout_sec: float = 300.0

    def with_overrides(
        self,
        *,
        server_url: str | None = None,
        default_workspace: str | None = None,
        system_prompt: str | None = None,
        color: bool | None = None,
        request_timeout_sec: float | None = None,
    ) -> "ClientConfig":
        return replace(
            self,
            server_url=server_url or self.server_url,
            default_workspace=default_workspace if default_workspace is not None else self.default_workspace,
            system_prompt=system_prompt or self.system_prompt,
            color=self.color if color is None else color,
            request_timeout_sec=self.request_timeout_sec if request_timeout_sec is None else request_timeout_sec,
        )


def load_client_config(path: str | Path | None = None) -> ClientConfig:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not config_path.exists():
        return ClientConfig()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ClientConfig()
    return ClientConfig(
        server_url=_str(data.get("server_url"), ClientConfig.server_url),
        default_workspace=_optional_str(data.get("default_workspace")),
        system_prompt=_str(data.get("system_prompt"), DEFAULT_SYSTEM_PROMPT),
        color=_bool(data.get("color"), True),
        request_timeout_sec=_float(data.get("request_timeout_sec"), 300.0),
    )


def _str(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _float(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
