"""LiteLLM wrapper for the Jentic Agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict
import os, json
import asyncio
import concurrent.futures
from .config import get_config_value


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async version of chat that runs sync method in thread pool."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.chat, messages, **kwargs)


class LiteLLMChatLLM(BaseLLM):
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        import litellm

        if model is None:
            model = get_config_value("llm", "model", default="gpt-4o")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = litellm

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self._client.completion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        content = resp.choices[0].message.content
        return content or ""
