"""xAI / Grok adapter.

xAI's API is OpenAI-compatible — their tool schema is identical to
the OpenAI Responses API format (``type / name / description /
parameters``).  The official xAI docs even show using the OpenAI
Python SDK with ``base_url="https://api.x.ai/v1"``.

This module re-exports the OpenAI adapter under xAI-branded names so
that:

1. ``from binomialhash.adapters.xai import get_xai_tools`` reads
   naturally in xAI-targeting code.
2. If xAI ever diverges from OpenAI's schema, we have a dedicated
   module to add xAI-specific logic without touching the OpenAI
   adapter.

Usage::

    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.xai import (
        get_xai_tools,
        handle_xai_tool_call,
    )

    specs = get_all_tools(bh)
    tools = get_xai_tools(specs)
    # pass to client.responses.create(tools=...) via xAI endpoint

    # When model returns a function_call item:
    result = handle_xai_tool_call(specs, item.name, item.arguments)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..tools.base import ToolSpec
from .openai import get_openai_tools, handle_openai_tool_call


def get_xai_tools(
    specs: Sequence[ToolSpec],
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Return tool definitions for the xAI / Grok API.

    Format is identical to OpenAI Responses API.

    Parameters
    ----------
    specs:
        ToolSpec objects (from ``get_all_tools(bh)``).
    strict:
        Passed through to the OpenAI adapter's strict-mode logic.
    """
    # xAI is wire-compatible with OpenAI Responses API format
    return get_openai_tools(specs, strict=strict, format="responses")


def handle_xai_tool_call(
    specs: Sequence[ToolSpec],
    name: str,
    arguments: Any,
) -> Any:
    """Execute a tool call from an xAI / Grok response.

    Parameters
    ----------
    specs:
        The same specs used to generate the tool list.
    name:
        ``function_call.name`` from the model output.
    arguments:
        ``function_call.arguments`` — JSON string or dict.
    """
    return handle_openai_tool_call(specs, name, arguments)
