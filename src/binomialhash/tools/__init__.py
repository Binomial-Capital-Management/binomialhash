"""Provider-neutral tool definitions for BinomialHash.

Usage::

    from binomialhash import BinomialHash
    from binomialhash.tools import get_all_tools

    bh = BinomialHash()
    specs = get_all_tools(bh)          # list of 68 ToolSpec objects
    retrieval = get_tools_by_group(bh, "retrieval")  # data-access tools
    export = get_tools_by_group(bh, "export")         # CSV / Markdown / artifact tools

Each ToolSpec carries a name, description, JSON Schema for inputs, and a
handler callable.  Pass these to an adapter (``binomialhash.adapters``)
to convert into provider-specific tool registrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import ToolSpec, _prop
from .export import _make_export_specs
from .manifold import _make_manifold_specs
from .retrieval import _make_retrieval_specs
from .stats import _make_stats_specs

if TYPE_CHECKING:
    from ..core import BinomialHash


def get_all_tools(bh: "BinomialHash") -> List[ToolSpec]:
    """Return all 68 BinomialHash tool specs bound to *bh*."""
    return (
        _make_retrieval_specs(bh)
        + _make_stats_specs(bh)
        + _make_manifold_specs(bh)
        + _make_export_specs(bh)
    )


def get_tools_by_group(bh: "BinomialHash", group: str) -> List[ToolSpec]:
    """Return only the tool specs matching *group* (retrieval, stats, manifold, export)."""
    return [t for t in get_all_tools(bh) if t.group == group]


__all__ = [
    "ToolSpec",
    "get_all_tools",
    "get_tools_by_group",
]
