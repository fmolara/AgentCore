from typing import TYPE_CHECKING, Any

from agentclient.config import ClientConfig, load_client_config
from agentclient.exit_codes import ExitCode

if TYPE_CHECKING:
    from agentclient.cli import RemoteAgentCLI

__all__ = [
    "ClientConfig",
    "ExitCode",
    "RemoteAgentCLI",
    "load_client_config",
]


def __getattr__(name: str) -> Any:
    if name == "RemoteAgentCLI":
        from agentclient.cli import RemoteAgentCLI

        return RemoteAgentCLI
    raise AttributeError(name)
