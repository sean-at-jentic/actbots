import tomllib
from typing import Any, Dict
from pathlib import Path
import os
import sys

_CONFIG_CACHE: Dict[str, Any] = {}

CONFIG_FILE = Path(__file__).parents[2] / "pyproject.toml"


def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        config_path = Path(CONFIG_FILE).resolve()
        try:
            with open(config_path, "rb") as f:
                full_config = tomllib.load(f)
            config = full_config.get("tool", {}).get("actbots", {})

            if "llm" in config and "provider" in config["llm"] and "model" in config["llm"]:
                provider = config["llm"]["provider"]
                model = config["llm"]["model"]
                if not model.startswith(provider + "/"):
                    config["llm"]["model"] = f"{provider}/{model}"

            if (
                "memory" in config
                and "llm_provider" in config["memory"]
                and "llm_model" in config["memory"]
            ):
                provider = config["memory"]["llm_provider"]
                model = config["memory"]["llm_model"]
                if not model.startswith(provider + "/"):
                    config["memory"]["llm_model"] = f"{provider}/{model}"

            _CONFIG_CACHE = config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from pyproject.toml: {e}")
    return _CONFIG_CACHE


def get_config() -> Dict[str, Any]:
    """Return the full config as a dict."""
    return _load_config()


def get_config_value(*keys, default=None) -> Any:
    """Get a nested config value by keys, e.g. get_config_value('llm', 'model')."""
    config = _load_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config


def get_discord_config() -> Dict[str, Any]:
    """Return the discord config as a dict."""
    config = _load_config()
    return config.get("discord", {})


def validate_api_keys():
    """Validate required API keys are present."""
    if not os.getenv("JENTIC_API_KEY"):
        print("ERROR: Missing JENTIC_API_KEY in your .env file.")
        sys.exit(1)

    from .config import get_config_value
    provider = get_config_value("llm", "provider", default="openai")
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    if provider in api_key_map and not os.getenv(api_key_map[provider]):
        print(f"ERROR: LLM provider is {provider} but {api_key_map[provider]} is not set in .env.")
        sys.exit(1)
