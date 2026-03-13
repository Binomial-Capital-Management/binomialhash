"""Export slot data as clean row dicts.

Standalone row export with column selection, sorting, and pagination.
Returns plain Python dicts — no BH metadata overhead — suitable for
frontend table components, DataFrame construction, or further piping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..predicates import sort_rows


def export_rows(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    *,
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = True,
    offset: int = 0,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return a clean list of row dicts from slot data.

    Parameters
    ----------
    rows / columns / col_types:
        Slot data from BinomialHash.
    select_columns:
        Subset of columns to include (``None`` = all).
    sort_by:
        Column to sort by.
    sort_desc:
        Sort direction.
    offset:
        Skip this many rows (after sorting).
    limit:
        Maximum rows to return.
    """
    if sort_by and sort_by in col_types:
        rows = sort_rows(rows, sort_by, col_types[sort_by], sort_desc)

    sliced = rows[offset: offset + min(limit, 5000)]

    if select_columns:
        col_set = set(select_columns)
        return [{k: v for k, v in r.items() if k in col_set} for r in sliced]
    return [dict(r) for r in sliced]
