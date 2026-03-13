"""Provider-neutral tool metadata structure.

A ToolSpec holds everything needed to register a BinomialHash tool with
any LLM provider: name, description, a JSON Schema for inputs, and a
handler callable.  Adapters (openai, anthropic, gemini, xai) translate
ToolSpecs into provider-specific tool definitions at registration time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """One provider-neutral tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    group: str = ""


def _prop(
    type_: str,
    description: str,
    *,
    default: Optional[Any] = None,
    enum: Optional[list] = None,
) -> Dict[str, Any]:
    """Shorthand for building a single JSON Schema property."""
    d: Dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        d["enum"] = enum
    if default is not None:
        d["default"] = default
    return d


def parse_columns(columns: Optional[str]) -> Optional[List[str]]:
    """Normalise a columns param that LLMs may send as JSON string, CSV, or list."""
    if columns is None:
        return None
    if isinstance(columns, list):
        return columns
    try:
        parsed = json.loads(columns)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return [c.strip() for c in columns.split(",") if c.strip()]
