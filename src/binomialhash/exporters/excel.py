"""Export slot data to header+values matrix (Excel-ready format)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..predicates import sort_rows


def export_excel_batch(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    key: str,
    label: str,
    row_count: int,
    *,
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = True,
    max_rows: int = 200,
) -> Dict[str, Any]:
    """Return a dict with ``headers`` and ``values`` ready for Excel output."""
    if sort_by and sort_by in col_types:
        rows = sort_rows(rows, sort_by, col_types[sort_by], sort_desc)
    rows = rows[:max_rows]
    headers = list(select_columns) if select_columns else list(columns)
    values = [[r.get(c) for c in headers] for r in rows]
    return {
        "key": key,
        "label": label,
        "headers": headers,
        "values": values,
        "total_exported": len(values),
        "total_available": row_count,
    }
