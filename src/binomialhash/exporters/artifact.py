"""Downloadable artifact wrapper for chat frontends.

Bundles exported data with a filename and MIME type so frontend
renderers can create download links or inline previews.

Usage::

    from binomialhash.exporters.artifact import build_artifact

    art = build_artifact(rows, columns, col_types,
                         format="csv", label="market_data")
    # {
    #     "type": "artifact",
    #     "filename": "market_data.csv",
    #     "mime_type": "text/csv",
    #     "content": "ticker,price,volume\\nAAPL,189.5,...",
    #     "row_count": 150,
    # }

Your React chat component checks ``item.type === "artifact"`` and
renders a download button.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .csv import export_csv
from .markdown import export_markdown
from .rows import export_rows

_FORMAT_META = {
    "csv": {"ext": "csv", "mime": "text/csv"},
    "markdown": {"ext": "md", "mime": "text/markdown"},
    "json": {"ext": "json", "mime": "application/json"},
    "jsonl": {"ext": "jsonl", "mime": "application/x-ndjson"},
}


def build_artifact(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    *,
    format: str = "csv",
    label: str = "export",
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = True,
    max_rows: int = 500,
    total_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a downloadable artifact from slot data.

    Parameters
    ----------
    rows / columns / col_types:
        Slot data from BinomialHash.
    format:
        One of ``"csv"``, ``"markdown"``, ``"json"``, ``"jsonl"``.
    label:
        Used to generate the filename (e.g. ``"market_data"`` ->
        ``"market_data.csv"``).
    select_columns / sort_by / sort_desc / max_rows:
        Forwarded to the underlying exporter.
    total_rows:
        Total available rows (for markdown footer note).

    Returns
    -------
    dict with keys: ``type``, ``filename``, ``mime_type``, ``content``,
    ``row_count``, ``format``.
    """
    meta = _FORMAT_META.get(format)
    if meta is None:
        raise ValueError(
            f"Unknown format '{format}'. "
            f"Choose from: {sorted(_FORMAT_META.keys())}"
        )

    common = dict(
        select_columns=select_columns,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )

    if format == "csv":
        content = export_csv(
            rows, columns, col_types, max_rows=max_rows, **common,
        )
        row_count = content.count("\n") - 1

    elif format == "markdown":
        content = export_markdown(
            rows, columns, col_types,
            max_rows=max_rows,
            total_rows=total_rows,
            label=label,
            **common,
        )
        row_count = min(len(rows), max_rows)

    elif format == "json":
        exported = export_rows(
            rows, columns, col_types, limit=max_rows, **common,
        )
        content = json.dumps(exported, indent=2, default=str)
        row_count = len(exported)

    elif format == "jsonl":
        exported = export_rows(
            rows, columns, col_types, limit=max_rows, **common,
        )
        content = "\n".join(json.dumps(r, default=str) for r in exported)
        row_count = len(exported)

    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
    filename = f"{safe_label}.{meta['ext']}"

    return {
        "type": "artifact",
        "filename": filename,
        "mime_type": meta["mime"],
        "content": content,
        "row_count": row_count,
        "format": format,
    }
