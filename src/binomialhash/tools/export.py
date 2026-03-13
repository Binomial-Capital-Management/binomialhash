"""Export tool specs — CSV, Markdown, and downloadable artifacts.

These tools let LLM agents produce frontend-compatible output from
stored BH datasets: inline markdown tables for chat rendering, CSV
downloads, JSON exports, etc.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import ToolSpec, _prop, parse_columns as _parse_columns

if TYPE_CHECKING:
    from ..core import BinomialHash


def _export_csv_handler(bh: "BinomialHash", key: str, columns: Optional[str] = None,
                        sort_by: Optional[str] = None, sort_desc: bool = True,
                        max_rows: int = 500) -> Any:
    slot = bh._get_slot(key)
    if slot is None:
        return {"error": f"Key '{key}' not found."}
    from ..exporters.csv import export_csv
    return {
        "type": "artifact",
        "filename": f"{slot.label}.csv",
        "mime_type": "text/csv",
        "content": export_csv(
            slot.rows, slot.columns, slot.col_types,
            select_columns=_parse_columns(columns),
            sort_by=sort_by, sort_desc=sort_desc, max_rows=max_rows,
        ),
        "format": "csv",
    }


def _export_markdown_handler(bh: "BinomialHash", key: str, columns: Optional[str] = None,
                             sort_by: Optional[str] = None, sort_desc: bool = True,
                             max_rows: int = 50) -> Any:
    slot = bh._get_slot(key)
    if slot is None:
        return {"error": f"Key '{key}' not found."}
    from ..exporters.markdown import export_markdown
    return export_markdown(
        slot.rows, slot.columns, slot.col_types,
        select_columns=_parse_columns(columns),
        sort_by=sort_by, sort_desc=sort_desc, max_rows=max_rows,
        total_rows=slot.row_count, label=slot.label,
    )


def _export_artifact_handler(bh: "BinomialHash", key: str, format: str = "csv",
                             columns: Optional[str] = None, sort_by: Optional[str] = None,
                             sort_desc: bool = True, max_rows: int = 500) -> Any:
    slot = bh._get_slot(key)
    if slot is None:
        return {"error": f"Key '{key}' not found."}
    from ..exporters.artifact import build_artifact
    return build_artifact(
        slot.rows, slot.columns, slot.col_types,
        format=format, label=slot.label,
        select_columns=_parse_columns(columns),
        sort_by=sort_by, sort_desc=sort_desc, max_rows=max_rows,
        total_rows=slot.row_count,
    )


def _make_export_specs(bh: "BinomialHash") -> List[ToolSpec]:
    return [
        ToolSpec(
            name="bh_to_csv",
            description=(
                "Export a stored dataset as CSV.  Returns an artifact object "
                "with filename, MIME type, and CSV content string that the "
                "frontend can render as a download link."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "columns": _prop("string", "Column names as JSON array string (omit for all)."),
                    "sort_by": _prop("string", "Column to sort by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "max_rows": _prop("integer", "Maximum rows (default 500, max 5000).", default=500),
                },
                "required": ["key"],
            },
            handler=lambda key, columns=None, sort_by=None, sort_desc=True, max_rows=500: (
                _export_csv_handler(bh, key, columns, sort_by, sort_desc, max_rows)
            ),
            group="export",
        ),
        ToolSpec(
            name="bh_to_markdown",
            description=(
                "Render a stored dataset as a Markdown table for inline display "
                "in the chat.  Numbers are right-aligned, strings left-aligned.  "
                "Use this when the user asks to 'show' or 'display' data."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "columns": _prop("string", "Column names as JSON array string (omit for all)."),
                    "sort_by": _prop("string", "Column to sort by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "max_rows": _prop("integer", "Maximum rows to display (default 50, max 200).", default=50),
                },
                "required": ["key"],
            },
            handler=lambda key, columns=None, sort_by=None, sort_desc=True, max_rows=50: (
                _export_markdown_handler(bh, key, columns, sort_by, sort_desc, max_rows)
            ),
            group="export",
        ),
        ToolSpec(
            name="bh_export",
            description=(
                "Export a stored dataset as a downloadable artifact.  Supports "
                "multiple formats: 'csv', 'markdown', 'json', 'jsonl'.  Returns "
                "an artifact object with filename, MIME type, and content that "
                "the frontend renders as a download button."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "format": _prop(
                        "string",
                        "Export format.",
                        enum=["csv", "markdown", "json", "jsonl"],
                        default="csv",
                    ),
                    "columns": _prop("string", "Column names as JSON array string (omit for all)."),
                    "sort_by": _prop("string", "Column to sort by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "max_rows": _prop("integer", "Maximum rows (default 500, max 5000).", default=500),
                },
                "required": ["key"],
            },
            handler=lambda key, format="csv", columns=None, sort_by=None, sort_desc=True, max_rows=500: (
                _export_artifact_handler(bh, key, format, columns, sort_by, sort_desc, max_rows)
            ),
            group="export",
        ),
    ]
