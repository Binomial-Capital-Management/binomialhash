"""OpenAI adapter — Responses API and Chat Completions API.

Translates :class:`ToolSpec` objects into the dict shapes that the
OpenAI Python SDK (``openai>=1.0``) expects in its ``tools`` parameter.

Two formats are supported:

* **Responses API** (recommended, ``client.responses.create``):
  top-level ``type / name / description / parameters``.
* **Chat Completions API** (legacy, ``client.chat.completions.create``):
  externally-tagged ``type / function: {name, description, parameters}``.

Neither format requires importing ``openai`` — the adapter returns
plain dicts that you pass straight to the SDK.

Usage::

    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.openai import (
        get_openai_tools,
        handle_openai_tool_call,
    )

    specs = get_all_tools(bh)
    tools = get_openai_tools(specs)  # pass to client.responses.create(tools=...)

    # When model returns a function_call item:
    result = handle_openai_tool_call(specs, item.name, item.arguments)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

from ..tools.base import ToolSpec
from .common import handle_tool_call, parse_arguments


def _spec_to_responses(spec: ToolSpec, *, strict: bool = False) -> Dict[str, Any]:
    """Convert one ToolSpec into an OpenAI **Responses API** tool dict."""
    params = copy.deepcopy(spec.input_schema)

    if strict:
        params.setdefault("additionalProperties", False)

    tool: Dict[str, Any] = {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": params,
    }
    if strict:
        tool["strict"] = True
    return tool


def _spec_to_chat_completions(spec: ToolSpec, *, strict: bool = False) -> Dict[str, Any]:
    """Convert one ToolSpec into an OpenAI **Chat Completions API** tool dict."""
    params = copy.deepcopy(spec.input_schema)

    if strict:
        params.setdefault("additionalProperties", False)

    fn: Dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "parameters": params,
    }
    if strict:
        fn["strict"] = True

    return {"type": "function", "function": fn}


# -- public API -----------------------------------------------------------

def get_openai_tools(
    specs: Sequence[ToolSpec],
    *,
    strict: bool = False,
    format: str = "responses",
) -> List[Dict[str, Any]]:
    """Return tool definitions for the OpenAI API.

    Parameters
    ----------
    specs:
        ToolSpec objects (from ``get_all_tools(bh)``).
    strict:
        If ``True``, enable Structured Outputs constraints
        (``additionalProperties: false``, all props required).
    format:
        ``"responses"`` (default) for the Responses API, or
        ``"chat_completions"`` for the legacy Chat Completions API.
    """
    if format == "chat_completions":
        return [_spec_to_chat_completions(s, strict=strict) for s in specs]
    return [_spec_to_responses(s, strict=strict) for s in specs]


def handle_openai_tool_call(
    specs: Sequence[ToolSpec],
    name: str,
    arguments: Any,
) -> Any:
    """Execute a tool call from an OpenAI response.

    Parameters
    ----------
    specs:
        The same specs used to generate the tool list.
    name:
        ``function_call.name`` from the model output.
    arguments:
        ``function_call.arguments`` — a JSON **string** from the
        Responses API, or already a dict if pre-parsed.
    """
    return handle_tool_call(specs, name, parse_arguments(arguments))
