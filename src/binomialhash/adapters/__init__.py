"""Provider adapters for BinomialHash tool specs.

Each sub-module translates :class:`~binomialhash.tools.ToolSpec` objects
into the dict shape a specific LLM provider SDK expects, and provides a
handler function to dispatch incoming tool calls.

Quick usage by provider::

    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.openai import get_openai_tools
    from binomialhash.adapters.anthropic import get_anthropic_tools
    from binomialhash.adapters.gemini import get_gemini_tools
    from binomialhash.adapters.xai import get_xai_tools

Or use the convenience router::

    from binomialhash.adapters import get_tools_for_provider
    tools = get_tools_for_provider(specs, provider="openai")
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..tools.base import ToolSpec
from .anthropic import get_anthropic_tools, handle_anthropic_tool_use
from .common import handle_tool_call, parse_arguments, safe_handle_tool_call
from .gemini import get_gemini_tools, handle_gemini_tool_call
from .openai import get_openai_tools, handle_openai_tool_call
from .xai import get_xai_tools, handle_xai_tool_call

# provider name → formatter function; looked up at runtime by get_tools_for_provider
_PROVIDERS = {
    "openai": get_openai_tools,
    "anthropic": get_anthropic_tools,
    "gemini": get_gemini_tools,
    "xai": get_xai_tools,
}


def get_tools_for_provider(
    specs: Sequence[ToolSpec],
    provider: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Return tool definitions formatted for *provider*.

    Parameters
    ----------
    specs:
        ToolSpec objects (from ``get_all_tools(bh)``).
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"gemini"``, ``"xai"``.
    **kwargs:
        Forwarded to the provider's ``get_*_tools`` function
        (e.g. ``strict=True`` for OpenAI).

    Raises
    ------
    ValueError
        If *provider* is not recognised.
    """
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {sorted(_PROVIDERS.keys())}"
        )
    return fn(specs, **kwargs)


__all__ = [
    "get_tools_for_provider",
    # common
    "handle_tool_call",
    "safe_handle_tool_call",
    "parse_arguments",
    # openai
    "get_openai_tools",
    "handle_openai_tool_call",
    # anthropic
    "get_anthropic_tools",
    "handle_anthropic_tool_use",
    # gemini
    "get_gemini_tools",
    "handle_gemini_tool_call",
    # xai
    "get_xai_tools",
    "handle_xai_tool_call",
]
