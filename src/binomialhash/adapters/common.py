"""Shared handler dispatch used by every provider adapter.

The core job: given a tool name and an arguments dict, find the matching
ToolSpec and invoke its handler.  Provider-specific adapters normalise
their incoming payloads (JSON-string arguments, protobuf args, etc.)
into a plain ``dict`` before calling :func:`handle_tool_call`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from ..tools.base import ToolSpec


def _build_index(specs: Sequence[ToolSpec]) -> Dict[str, ToolSpec]:
    """Map tool names to their specs for O(1) lookup."""
    return {s.name: s for s in specs}


def handle_tool_call(
    specs: Sequence[ToolSpec],
    name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Dispatch *name* to the correct handler in *specs*.

    Parameters
    ----------
    specs:
        The full set of ``ToolSpec`` objects (from ``get_all_tools``).
    name:
        Tool name the model chose to call.
    arguments:
        Already-parsed keyword arguments.  Provider adapters are
        responsible for deserialising JSON strings / protobuf maps
        before calling this function.

    Returns
    -------
    Whatever the handler returns (usually a ``dict`` or ``str``).

    Raises
    ------
    KeyError
        If *name* does not match any spec.
    """
    idx = _build_index(specs)
    if name not in idx:
        raise KeyError(
            f"Unknown tool '{name}'. "
            f"Available: {sorted(idx.keys())}"
        )
    return idx[name].handler(**arguments)


def safe_handle_tool_call(
    specs: Sequence[ToolSpec],
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Like :func:`handle_tool_call` but catches exceptions.

    Returns ``{"result": ...}`` on success or
    ``{"error": "<message>"}`` on failure.  Useful when you want to
    feed errors back to the model rather than crash the host process.
    """
    try:
        result = handle_tool_call(specs, name, arguments)
        return {"result": result}
    except Exception as exc:
        logger.exception("Tool '%s' raised %s", name, type(exc).__name__)
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """Normalise *raw* into a plain dict.

    Handles three shapes callers might pass in:

    - Already a dict → returned as-is.
    - A JSON string (OpenAI / xAI ``function_call.arguments``) → decoded.
    - ``None`` or empty → returns ``{}``.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}
