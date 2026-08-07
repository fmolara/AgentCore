from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("planner output must be a JSON object")
    return data


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("planner output is not valid JSON")
    return text[start : end + 1]
