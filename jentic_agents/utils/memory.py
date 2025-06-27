"""Simple in‑memory store + placeholder resolution helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryItem:
    key: str
    value: Any
    description: str
    type: str | None = None
    schema: Dict[str, Any] | None = None

    def as_prompt_line(self) -> str:
        part_type = f" ({self.type})" if self.type else ""
        return f"• {self.key}{part_type} – {self.description}"


class ScratchPadMemory:
    """Dict‑backed memory suitable for a single agent run."""

    _store: Dict[str, MemoryItem]

    def __init__(self) -> None:
        self._store = {}

    # ---------- public API ----------
    def set(self, key: str, value: Any, description: str, type_: str | None = None, schema: Dict[str, Any] | None = None) -> None:
        self._store[key] = MemoryItem(key, value, description, type_, schema)

    def get(self, key: str) -> Any:
        return self._store[key].value

    def has(self, key: str) -> bool:
        return key in self._store

    def enumerate_for_prompt(self) -> str:
        if not self._store:
            return "(memory empty)"
        lines = ["Available memory:"]
        for item in self._store.values():
            lines.append(item.as_prompt_line())
        return "\n".join(lines)

    # ---------- placeholder resolution ----------

    _MEMORY_RE = re.compile(r"\$\{memory\.([\w_\.]+)\}")

    def resolve_placeholders(self, obj: Any) -> Any:
        """Recursively substitute ${memory.foo.bar} placeholders inside *obj*."""
        if isinstance(obj, str):
            return self._MEMORY_RE.sub(lambda m: self._lookup(m.group(1)), obj)
        if isinstance(obj, list):
            return [self.resolve_placeholders(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.resolve_placeholders(v) for k, v in obj.items()}
        return obj

    # ---------- private helpers ----------

    def _lookup(self, dotted_path: str) -> str:
        key, *path = dotted_path.split(".")
        item = self._store[key].value
        for part in path:
            item = item[part]
        # stringify JSON‑serialisable objects;
        return json.dumps(item) if not isinstance(item, str) else item 