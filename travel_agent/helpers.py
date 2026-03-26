"""Shared helpers for Travel-Agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def safe_json_loads(data: str) -> Dict[str, Any]:
    """Parse JSON safely; return error info on failure."""

    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:  # noqa: BLE001
        return {"__parse_error": str(exc)}


def normalize_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Convert tool_calls from OpenAI objects to plain dicts for uniform handling."""

    if not raw:
        return []
    result: List[Dict[str, Any]] = []
    for item in raw:
        if hasattr(item, "model_dump"):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = item
        else:
            item_dict = {
                "id": getattr(item, "id", ""),
                "type": getattr(item, "type", ""),
                "function": {
                    "name": getattr(getattr(item, "function", None), "name", ""),
                    "arguments": getattr(getattr(item, "function", None), "arguments", "{}"),
                },
            }
        result.append(item_dict)
    return result
