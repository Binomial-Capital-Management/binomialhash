"""Export slot data to a GitHub Flavored Markdown table.

Designed for inline rendering in chat UIs (React, Slack, etc.).
Numbers are right-aligned, strings left-aligned.  Long cell values
are truncated to keep the table readable in constrained viewports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..predicates import sort_rows
from ..schema import T_NUMERIC


def _fmt_cell(value: Any, max_width: int = 40) -> str:
    if value is None:
        return ""
    s = str(value).replace("|", "\\|")
    if len(s) > max_width:
        return s[: max_width - 1] + "\u2026"
    return s


def _align_marker(col: str, col_types: Dict[str, str]) -> str:
    if col_types.get(col) == T_NUMERIC:
        return "---:"
    return ":---"


def export_markdown(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    *,
    select_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_desc: bool = True,
    max_rows: int = 50,
    max_cell_width: int = 40,
    total_rows: Optional[int] = None,
    label: Optional[str] = None,
) -> str:
    """Render *rows* as a Markdown table string.

    Parameters
    ----------
    rows / columns / col_types:
        Slot data from BinomialHash.
    select_columns:
        Subset of columns to include (``None`` = all).
    sort_by:
        Column to sort by before rendering.
    sort_desc:
        Sort direction.
    max_rows:
        Row cap for the table (keeps chat messages reasonable).
    max_cell_width:
        Truncate cell values wider than this.
    total_rows:
        Total available rows (for the footer note).
    label:
        Dataset label (for the caption).
    """
    if sort_by and sort_by in col_types:
        rows = sort_rows(rows, sort_by, col_types[sort_by], sort_desc)

    capped = min(max_rows, 200)
    display_rows = rows[:capped]
    headers = list(select_columns) if select_columns else list(columns)

    lines: List[str] = []

    header_line = "| " + " | ".join(headers) + " |"
    align_line = "| " + " | ".join(_align_marker(h, col_types) for h in headers) + " |"
    lines.append(header_line)
    lines.append(align_line)

    for row in display_rows:
        cells = [_fmt_cell(row.get(c), max_cell_width) for c in headers]
        lines.append("| " + " | ".join(cells) + " |")

    actual_total = total_rows if total_rows is not None else len(rows)
    if len(display_rows) < actual_total:
        lines.append("")
        lines.append(
            f"*Showing {len(display_rows)} of {actual_total} rows"
            + (f" from **{label}**" if label else "")
            + ".*"
        )

    return "\n".join(lines)
