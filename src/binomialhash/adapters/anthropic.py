"""Anthropic adapter — Claude Messages API.

Translates :class:`ToolSpec` objects into the dict shapes expected by
the ``anthropic`` Python SDK's ``tools`` parameter.

Anthropic format::

    {
        "name": "bh_retrieve",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": { ... },
            "required": [ ... ]
        }
    }

This is nearly identical to our internal ``ToolSpec`` layout.  The
adapter validates names against Anthropic's regex, and optionally
appends example inputs to the description for few-shot tool-use guidance.

No ``anthropic`` SDK import is required — returns plain dicts.

Usage::

    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.anthropic import (
        get_anthropic_tools,
        handle_anthropic_tool_use,
    )

    specs = get_all_tools(bh)
    tools = get_anthropic_tools(specs)
    # pass to client.messages.create(tools=tools, ...)

    # When model returns a tool_use content block:
    result = handle_anthropic_tool_use(specs, block.name, block.input)
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..tools.base import ToolSpec
from .common import handle_tool_call

_ANTHROPIC_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_name(name: str) -> None:
    if not _ANTHROPIC_NAME_RE.match(name):
        raise ValueError(
            f"Tool name '{name}' does not match Anthropic's required "
            f"pattern: ^[a-zA-Z0-9_-]{{1,64}}$"
        )


def _spec_to_anthropic(
    spec: ToolSpec,
    *,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert one ToolSpec into an Anthropic tool definition dict."""
    _validate_name(spec.name)
    description = spec.description
    if examples:
        lines = []
        for ex in examples:
            pairs = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in ex.items())
            lines.append(f"  {{{pairs}}}")
        description += "\n\nExample inputs:\n" + "\n".join(lines)
    tool: Dict[str, Any] = {
        "name": spec.name,
        "description": description,
        "input_schema": copy.deepcopy(spec.input_schema),
    }
    return tool


# -- public API -----------------------------------------------------------

def get_anthropic_tools(
    specs: Sequence[ToolSpec],
    *,
    examples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Return tool definitions for the Anthropic Messages API.

    Parameters
    ----------
    specs:
        ToolSpec objects (from ``get_all_tools(bh)``).
    examples:
        Optional mapping of ``{tool_name: [example_input, ...]}``.
        If provided, matching tool names get example inputs appended
        to the description for few-shot tool-use guidance.
    """
    examples = examples or {}
    return [
        _spec_to_anthropic(s, examples=examples.get(s.name))
        for s in specs
    ]


def handle_anthropic_tool_use(
    specs: Sequence[ToolSpec],
    name: str,
    tool_input: Dict[str, Any],
) -> Any:
    """Execute a tool call from an Anthropic ``tool_use`` content block.

    Parameters
    ----------
    specs:
        The same specs used to generate the tool list.
    name:
        ``content_block.name`` from the model's ``tool_use`` output.
    tool_input:
        ``content_block.input`` — Anthropic delivers this as a
        **parsed dict**, not a JSON string.
    """
    if not isinstance(tool_input, dict):
        tool_input = {}
    return handle_tool_call(specs, name, tool_input)
