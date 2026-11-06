"""Required-config-key access: symbols come from config, never from code defaults."""
from typing import Any


def require(cfg: dict, dotted_key: str) -> Any:
    """Return the value at ``dotted_key`` or raise a KeyError naming the key."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config is missing required key '{dotted_key}'")
        node = node[part]
    return node
