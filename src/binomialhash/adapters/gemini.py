"""Google Gemini adapter — ``google.genai`` / Vertex AI.

Translates :class:`ToolSpec` objects into function-declaration dicts
that Gemini expects inside ``types.Tool(function_declarations=[...])``.

Declaration format (per Google docs)::

    {
        "name": "bh_retrieve",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": { ... },
            "required": [ ... ]
        }
    }

The adapter returns **plain dicts**, not SDK objects.  Wrap them
yourself::

    from google.genai import types
    from binomialhash.adapters.gemini import get_gemini_tools

    decls = get_gemini_tools(specs)
    gemini_tools = types.Tool(function_declarations=decls)

This keeps ``google-genai`` out of our dependency tree.

Usage::

    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.gemini import (
        get_gemini_tools,
        handle_gemini_tool_call,
    )

    specs = get_all_tools(bh)
    decls = get_gemini_tools(specs)

    # When model returns a function_call part:
    result = handle_gemini_tool_call(specs, fc.name, fc.args)
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Sequence

from ..tools.base import ToolSpec
from .common import handle_tool_call

# Gemini names: must start with letter/underscore, up to 64 chars total
_GEMINI_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.:-]{0,63}$")


def _validate_name(name: str) -> None:
    if not _GEMINI_NAME_RE.match(name):
        raise ValueError(
            f"Tool name '{name}' does not match Gemini's naming rules: "
            f"must start with letter or underscore, max 64 chars, "
            f"allowed chars: a-z A-Z 0-9 _ . : -"
        )


def _spec_to_gemini(spec: ToolSpec) -> Dict[str, Any]:
    """Convert one ToolSpec into a Gemini function declaration dict."""
    _validate_name(spec.name)
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": copy.deepcopy(spec.input_schema),
    }


def _normalize_args(args: Any) -> Dict[str, Any]:
    """Convert Gemini's proto-map *args* into a plain dict.

    ``function_call.args`` from the Gemini SDK is a protobuf MapComposite
    that behaves like a dict but isn't one.  Calling ``dict()`` on it
    (or iterating its items) produces a plain dict safely.
    """
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    # Gemini SDK returns a protobuf MapComposite; dict() materialises it
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


# -- public API -----------------------------------------------------------

def get_gemini_tools(specs: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    """Return function declaration dicts for the Gemini API.

    Wrap the result with the SDK yourself::

        from google.genai import types
        gemini_tools = types.Tool(function_declarations=get_gemini_tools(specs))

    Parameters
    ----------
    specs:
        ToolSpec objects (from ``get_all_tools(bh)``).
    """
    return [_spec_to_gemini(s) for s in specs]


def handle_gemini_tool_call(
    specs: Sequence[ToolSpec],
    name: str,
    args: Any,
) -> Any:
    """Execute a tool call from a Gemini ``function_call`` part.

    Parameters
    ----------
    specs:
        The same specs used to generate the declarations.
    name:
        ``function_call.name`` from the model response.
    args:
        ``function_call.args`` — a protobuf MapComposite or plain dict.
    """
    return handle_tool_call(specs, name, _normalize_args(args))
