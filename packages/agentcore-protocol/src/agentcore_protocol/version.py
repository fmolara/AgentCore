from __future__ import annotations

API_VERSION = "v1"
PROTOCOL_VERSION = "1.0"
SCHEMA_VERSION = "1.0"


def major_version(version: str) -> str:
    return version.split(".", 1)[0]


def compatible_protocol(server_version: str, client_version: str = PROTOCOL_VERSION) -> bool:
    return major_version(server_version) == major_version(client_version)


def compatible_schema(server_version: str, client_version: str = SCHEMA_VERSION) -> bool:
    return major_version(server_version) == major_version(client_version)
