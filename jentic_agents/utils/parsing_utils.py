"""Utility helpers for stripping markdown code fences ( ``` or ```json … ``` )
from JSON strings and for recursively cleaning nested data structures.
"""

from __future__ import annotations

from typing import Any, List, Callable, Dict

__all__: List[str] = ["strip_backtick_fences", "cleanse", "unwrap_singleton_json", "extract_fenced_code", "safe_json_loads", "resolve_placeholders"]

import re
import json


def unwrap_singleton_json(text: str) -> str | Any:
    """If *text* looks like a JSON object that contains just **one** field
    (e.g. '{"id": "abc"}' or '{"list_id": "123"}'), return the inner
    value (e.g. 'abc'). Otherwise return *text* unchanged.

    This is useful when placeholder resolution expands to a JSON string but the
    target API expects only the raw primitive value.
    """
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return text
    try:
        import json

        data = json.loads(s)
        if isinstance(data, dict) and len(data) == 1:
            return next(iter(data.values()))
    except Exception:
        # Not valid JSON, fall through
        pass
    return text


def strip_backtick_fences(text: str) -> str:
    """Remove leading/trailing triple-backtick fences from *text*.

    Handles opening fences with an optional language hint (e.g. ```json).
    If *text* is not fenced, it is returned unchanged.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s

    lines: List[str] = s.splitlines()
    # Drop the opening fence (with optional language tag)
    lines = lines[1:]
    # Remove trailing fence(s)
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def cleanse(obj: Any) -> Any:
    """Recursively strip markdown fences from strings inside *obj*.

    Works for plain strings, lists and dictionaries. Other data types are
    returned unchanged.
    """
    if isinstance(obj, str):
        return unwrap_singleton_json(strip_backtick_fences(obj))
    if isinstance(obj, list):
        return [cleanse(v) for v in obj]
    if isinstance(obj, dict):
        return {k: cleanse(v) for k, v in obj.items()}
    return obj


def extract_fenced_code(text: str) -> str:
    """Return the first triple-backtick-fenced block, else raise."""
    m = re.search(r"```[\s\S]+?```", text)
    if not m:
        raise RuntimeError("No fenced code block found in text")
    fenced = m.group(0)
    # Remove opening and closing fences (```)
    inner = fenced.strip("`")  # remove all backticks at ends
    # After stripping, drop any leading language hint (e.g. ```markdown)
    if "\n" in inner:
        inner = inner.split("\n", 1)[1]  # drop first line (language) if present
    # Remove trailing fence that may remain after stripping leading backticks
    if inner.endswith("```"):
        inner = inner[:-3]
    return inner.strip()

def safe_json_loads(text: str) -> Dict:
    """Parse JSON even if wrapped in a Markdown fence."""
    s = strip_backtick_fences(text.strip())
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\n{s}")

def resolve_placeholders(obj: Any, resolver: Callable[[Any], Any]) -> Any:
    """Recursively resolve placeholders in obj using the provided resolver function."""
    try:
        result = resolver(obj)
        return cleanse(result)
    except KeyError:
        # Return the original object with unresolved placeholders
        return obj 