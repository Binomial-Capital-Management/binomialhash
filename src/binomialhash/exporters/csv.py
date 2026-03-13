"""Export slot data to CSV string.

Pure-stdlib implementation using the ``csv`` module.  Returns a plain
string so callers can write it to a file, stream it as a download, or
embed it in a chat artifact.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

from ..predicates import sort_rows


def export_csv(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    *,
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = True,
    max_rows: int = 500,
    include_header: bool = True,
) -> str:
    """Render *rows* as a CSV string.

    Parameters
    ----------
    rows / columns / col_types:
        Slot data from BinomialHash.
    select_columns:
        Subset of columns to include (``None`` = all).
    sort_by:
        Column to sort by before export.
    sort_desc:
        Sort direction.
    max_rows:
        Hard cap on exported rows.
    include_header:
        Whether to emit a header row.
    """
    if sort_by and sort_by in col_types:
        rows = sort_rows(rows, sort_by, col_types[sort_by], sort_desc)
    rows = rows[: min(max_rows, 5000)]

    headers = list(select_columns) if select_columns else list(columns)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    if include_header:
        writer.writerow(headers)
    for row in rows:
        writer.writerow([row.get(c, "") for c in headers])
    return buf.getvalue()
