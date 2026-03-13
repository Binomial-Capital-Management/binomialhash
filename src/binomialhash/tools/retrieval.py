"""Retrieval and data-access tool specs.

Covers: bh_retrieve, bh_aggregate, bh_query, bh_schema, bh_group_by,
bh_to_excel.  Each factory takes a BinomialHash instance and returns a
list of ToolSpec objects whose handlers delegate to the corresponding
BH methods.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import ToolSpec, _prop, parse_columns as _parse_columns

if TYPE_CHECKING:
    from ..core import BinomialHash


def _make_retrieval_specs(bh: "BinomialHash") -> List[ToolSpec]:
    return [
        ToolSpec(
            name="bh_retrieve",
            description=(
                "Retrieve a page of rows from a stored dataset.  Supports "
                "optional sorting, column projection, and offset-based "
                "pagination.  Row cap protects context budget."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "BH dataset key returned by ingest."),
                    "offset": _prop("integer", "Row offset (0-based).", default=0),
                    "limit": _prop("integer", "Max rows to return (max 50).", default=25),
                    "sort_by": _prop("string", "Column name to sort by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "columns": _prop("string", "Column names as JSON array string (omit for all)."),
                },
                "required": ["key"],
            },
            handler=lambda key, offset=0, limit=25, sort_by=None, sort_desc=True, columns=None: (
                bh.retrieve(key, offset, limit, sort_by, sort_desc, _parse_columns(columns))
            ),
            group="retrieval",
        ),
        ToolSpec(
            name="bh_aggregate",
            description=(
                "Compute a scalar aggregate on a stored dataset column.  "
                "Returns the result directly — no rows enter the context.  "
                "Funcs: sum, mean, median, min, max, std, count, count_distinct."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "column": _prop("string", "Column name to aggregate."),
                    "func": _prop(
                        "string", "Aggregation function.",
                        enum=["sum", "mean", "median", "min", "max", "std", "count", "count_distinct"],
                    ),
                },
                "required": ["key", "column", "func"],
            },
            handler=lambda key, column, func: bh.aggregate(key, column, func),
            group="retrieval",
        ),
        ToolSpec(
            name="bh_query",
            description=(
                "Filter and retrieve rows with a WHERE clause.  Supports "
                "compound AND/OR conditions and operators: =, !=, >, <, >=, "
                "<=, in, not_in, contains."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "where_json": _prop(
                        "string",
                        "JSON filter spec.  Single: {\"column\":\"vol\",\"op\":\">\",\"value\":1e6}.  "
                        "Compound: {\"and\":[...]} or {\"or\":[...]}.  Nesting allowed.",
                    ),
                    "sort_by": _prop("string", "Column to sort by after filtering."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "limit": _prop("integer", "Max rows to return (max 50).", default=25),
                    "columns": _prop("string", "Column names as JSON array string."),
                },
                "required": ["key", "where_json"],
            },
            handler=lambda key, where_json, sort_by=None, sort_desc=True, limit=25, columns=None: (
                bh.query(key, where_json, sort_by, sort_desc, limit, _parse_columns(columns))
            ),
            group="retrieval",
        ),
        ToolSpec(
            name="bh_schema",
            description=(
                "Get full schema info for a stored dataset.  Returns column "
                "names, inferred types (numeric/string/date/bool), and per-column "
                "statistics (min, max, mean, unique count, etc.)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                },
                "required": ["key"],
            },
            handler=lambda key: bh.schema(key),
            group="retrieval",
        ),
        ToolSpec(
            name="bh_group_by",
            description=(
                "Group rows by column(s) and aggregate — like SQL GROUP BY.  "
                "Returns grouped results with computed aggregates."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "group_cols": _prop("string", "Column(s) to group by as JSON array string."),
                    "agg_json": _prop(
                        "string",
                        "JSON list of aggregations.  Each: {\"column\":\"...\", \"func\":\"...\", \"alias\":\"...\"}.",
                    ),
                    "sort_by": _prop("string", "Column or alias to sort results by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "limit": _prop("integer", "Max groups to return (max 50).", default=50),
                },
                "required": ["key", "group_cols", "agg_json"],
            },
            handler=lambda key, group_cols, agg_json, sort_by=None, sort_desc=True, limit=50: (
                bh.group_by(key, _parse_columns(group_cols) or [], agg_json, sort_by, sort_desc, limit)
            ),
            group="retrieval",
        ),
        ToolSpec(
            name="bh_to_excel",
            description=(
                "Export a dataset as headers + values arrays for batch Excel "
                "writing.  Returns {headers, values, total_exported} so you can "
                "dump an entire dataset in two write calls."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "columns": _prop("string", "Column names as JSON array string (omit for all)."),
                    "sort_by": _prop("string", "Column to sort by."),
                    "sort_desc": _prop("boolean", "Sort descending.", default=True),
                    "max_rows": _prop("integer", "Maximum rows to export (default 200, max 500).", default=200),
                },
                "required": ["key"],
            },
            handler=lambda key, columns=None, sort_by=None, sort_desc=True, max_rows=200: (
                bh.to_excel_batch(key, _parse_columns(columns), sort_by, sort_desc, max_rows)
            ),
            group="retrieval",
        ),
    ]
