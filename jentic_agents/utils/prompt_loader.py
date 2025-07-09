"""
Utility for loading prompts from the filesystem.
"""
import json
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


def load_prompt(prompt_name: str) -> Any:
    """Load a prompt from the prompts directory. Return JSON if file is JSON, else string."""
    # Prompts are stored relative to the 'jentic_agents' package root.
    # This makes the loader independent of where it's called from.
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("{"):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from prompt file: {prompt_path}")
                    logger.error(f"--- FAULTY PROMPT CONTENT ---\n{content}\n")
                    raise e  # Re-raise the original error after logging
            return content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise RuntimeError(f"Prompt file not found: {prompt_path}") 
    