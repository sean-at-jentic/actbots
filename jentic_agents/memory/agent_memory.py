"""
# Create memory instance
memory = create_agent_memory()

# Add memories
ids = memory.add("User loves pizza", user_id="alice", agent_id="bot1", session_id="sess1")

# Search memories
results = memory.search("food preferences", user_id="alice", agent_id="bot1", limit=3)

# Get all memories
all_memories = memory.get_all(user_id="alice", agent_id="bot1")

# Get context for chat
context = memory.get_context_for_chat("What food does user like?", user_id="alice")

# Update memory
memory.update(memory_id="mem_123", data="Updated memory text")

# Delete memories
memory.delete_memory(memory_id="mem_123")
memory.delete_all(user_id="alice", agent_id="bot1")

# Health check
status = memory.health_check()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

from mem0 import Memory
from ..utils.logger import get_logger
from .base_memory import BaseMemory
from ..utils.config import get_config_value


logger = get_logger(__name__)


class AgentMemory(BaseMemory):
    """High-level memory wrapper with a minimal footprint."""

    # Construction helpers
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_telemetry: bool = False,
    ):
        """Create an AgentMemory instance.

        Args:
            config: Mem0 configuration dict. Must not be None; use
                `create_agent_memory()` to obtain a ready-made LiteLLM config.
            enable_telemetry: Keep Mem0 telemetry on if True.
        """

        if config is None:
            raise ValueError("`config` is required. Ue create_agent_memory()")

        # Silence Mem0 telemetry unless explicitly re-enabled
        if not enable_telemetry:
            os.environ.setdefault("MEM0_TELEMETRY", "false")

        self.memory = Memory.from_config(config)
        self.config = config

        self._kv: Dict[str, Any] = {}  # alias for fast look-ups – mirrors _store.value

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Persist *messages* and return their Mem0 ids."""

        meta = dict(metadata or {})
        if agent_id:
            meta["agent_id"] = agent_id
        if session_id:
            meta["session_id"] = session_id

        result = self.memory.add(messages, user_id=user_id, metadata=meta)
        return [item["id"] for item in result.get("results", [])]

    # search
    def search(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Semantic search with optional post-filtering by metadata."""

        # Fetch more than we need if post-filtering is requested so we still
        # return *limit* items whenever possible.
        initial_limit = limit * 3 if (agent_id or session_id or filters) else limit

        raw = self.memory.search(query=query, user_id=user_id, limit=initial_limit)

        if not (agent_id or session_id or filters):
            return raw

        def _included(mem: Dict[str, Any]) -> bool:
            # Handle case where metadata is None
            metadata = mem.get("metadata") or {}

            # Check agent_id (stored as top-level field)
            if agent_id and mem.get("agent_id") != agent_id:
                return False

            # Check session_id (stored in metadata)
            if session_id and metadata.get("session_id") != session_id:
                return False

            # Check additional filters (in metadata)
            if filters and any(metadata.get(k) != v for k, v in filters.items()):
                return False
            return True

        filtered: List[Dict[str, Any]] = []
        for mem in raw.get("results", []):
            if _included(mem):
                filtered.append(mem)
                if len(filtered) >= limit:
                    break

        return {"results": filtered}

    # retrieval
    def get_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return *limit* memories optionally filtered by agent / session."""

        raw = self.memory.get_all(user_id=user_id, limit=limit)
        data = raw["results"] if isinstance(raw, dict) else raw

        if agent_id or session_id:
            data = [
                m
                for m in data
                if (
                    # Check agent_id (stored as top-level field)
                    (not agent_id or m.get("agent_id") == agent_id)
                    and (
                        # Check session_id (stored in metadata)
                        not session_id
                        or (m.get("metadata") or {}).get("session_id") == session_id
                    )
                )
            ]

        return data

    # update
    def update(
        self, memory_id: str, data: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.memory.update(memory_id=memory_id, data=data)

    # deletion
    def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        try:
            self.memory.delete(memory_id=memory_id)
            return True
        except Exception as exc:
            logger.debug("delete_memory failed: %s", exc)
            return False

    def delete_all(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        memories = self.get_all(user_id, agent_id, session_id)
        deleted = 0
        for mem in memories:
            if self.delete_memory(mem["id"]):
                deleted += 1
        return deleted

    # analytics
    def get_memory_usage(self, user_id: str) -> Dict[str, Any]:
        memories = self.get_all(user_id)

        sessions = {
            m.get("metadata", {}).get("session_id")
            for m in memories
            if m.get("metadata")
        }
        agents = {
            m.get("metadata", {}).get("agent_id") for m in memories if m.get("metadata")
        }

        types: Dict[str, int] = {}
        for m in memories:
            t = m.get("metadata", {}).get("type", "general")
            types[t] = types.get(t, 0) + 1

        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "unique_sessions": len(sessions),
            "unique_agents": len(agents),
            "memory_types": types,
            "sessions": list(sessions),
            "agents": list(agents),
        }

    # chat helper
    def get_context_for_chat(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_memories: int = 5,
    ) -> str:
        res = self.search(query, user_id, agent_id, session_id, max_memories)
        hits = res.get("results", [])
        if not hits:
            return "No relevant memories found."

        lines = ["Relevant memories:"]
        for m in hits:
            lines.append(
                f"- {m.get('memory', '')} (relevance: {m.get('score', 0.0):.2f})"
            )
        return "\n".join(lines)

    # health-check
    def health_check(self) -> Dict[str, Any]:
        status = {
            "status": "healthy",
            "mem0_config": self.config,
            "timestamp": time.time(),
            "kv_store_size": len(self._kv),
        }

        try:
            mem_id_list = self.add("health-check", user_id="_health")
            if mem_id_list:
                self.delete_memory(mem_id_list[0])
        except Exception as exc:  # pragma: no cover – defensive
            status["status"] = "unhealthy"
            status["errors"].append(str(exc))

        return status

    def resolve_placeholders(self, obj: Any) -> Any:
        """Placeholder resolution is not supported by AgentMemory."""
        logger.warning("resolve_placeholders is not implemented in AgentMemory")
        return obj

    def validate_placeholders(self, args: Dict[str, Any], required_fields: list) -> tuple[Optional[str], Optional[str]]:
        """Placeholder validation is not supported by AgentMemory."""
        logger.warning("validate_placeholders is not implemented in AgentMemory")
        return None, None

    # BaseMemory interface implementations
    def store(self, key: str, value: Any) -> None:
        """Store a value in the in-memory key-value dictionary."""
        self._kv[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        return self._kv.get(key)

    def delete(self, key: str) -> bool:
        """Delete a stored value by key."""
        if key in self._kv:
            del self._kv[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored values."""
        self._kv.clear()

    def keys(self) -> list[str]:
        """Get all stored keys."""
        return list(self._kv.keys())


# factory
def create_agent_memory() -> AgentMemory:
    """Create an `AgentMemory` instance configured to use LiteLLM + Chroma."""

    memory_config = get_config_value("memory", default={})
    try:
        chroma_path = memory_config["chroma_path"]
        llm_model = memory_config["llm_model"]
        llm_provider = memory_config["llm_provider"]
        embed_model = memory_config["embed_model"]
    except Exception as e:
        raise Exception(f"Error loading memory config: {str(e)}")

    cfg = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_memory",
                "path": chroma_path,
            },
        },
        "llm": {
            "provider": llm_provider,
            "config": {"model": llm_model, "temperature": 0.1},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": embed_model},
        },
    }

    return AgentMemory(config=cfg)
