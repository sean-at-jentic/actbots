import os
import json
from typing import Any, Dict
from pathlib import Path

_CONFIG_CACHE: Dict[str, Any] = {}

CONFIG_FILE = os.environ.get(
    "JENTIC_CONFIG", os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
)


def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        config_path = Path(CONFIG_FILE).resolve()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _CONFIG_CACHE = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    return _CONFIG_CACHE


def get_config() -> Dict[str, Any]:
    """Return the full config as a dict."""
    return _load_config()


def get_config_value(*keys, default=None) -> Any:
    """Get a nested config value by keys, e.g. get_config_value('llm', 'model')."""
    cfg = _load_config()
    for key in keys:
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg
