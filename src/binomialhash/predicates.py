"""Predicate building, sorting, and row shaping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .schema import T_NUMERIC, T_STRING
from .stats import to_float_permissive

_CMP_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a is not None and b is not None and a > b,
    "<": lambda a, b: a is not None and b is not None and a < b,
    ">=": lambda a, b: a is not None and b is not None and a >= b,
    "<=": lambda a, b: a is not None and b is not None and a <= b,
    "contains": lambda a, b: b is not None and a is not None and str(b).lower() in str(a).lower(),
}


@dataclass(frozen=True)
class QueryBuildPolicy:
    """Explicit predicate-builder limits.

    Defaults are correctness-first and do not silently truncate user queries.
    Callers may opt into bounded parsing explicitly.
    """

    max_depth: Optional[int] = 20
    max_clauses_per_node: Optional[int] = None


DEFAULT_QUERY_BUILD_POLICY = QueryBuildPolicy()


def build_leaf_predicate(col: str, op: str, val: Any, col_type: str) -> Optional[Callable[[Dict[str, Any]], bool]]:
    """Build a predicate for a single {column, op, value} clause."""
    if op == "in":
        values = set(val) if isinstance(val, list) else {val}
        return lambda row, _values=values: row.get(col) in _values
    if op == "not_in":
        values = set(val) if isinstance(val, list) else {val}
        return lambda row, _values=values: row.get(col) not in _values
    cmp_fn = _CMP_OPS.get(op)
    if cmp_fn is None:
        return None
    if col_type == T_NUMERIC:
        float_val = to_float_permissive(val)
        return lambda row, _cmp=cmp_fn, _target=float_val: _cmp(to_float_permissive(row.get(col)), _target)
    return lambda row, _cmp=cmp_fn, _target=val: _cmp(row.get(col), _target)


def build_predicate(
    where: Dict[str, Any],
    col_types: Dict[str, str],
    depth: int = 0,
    policy: QueryBuildPolicy = DEFAULT_QUERY_BUILD_POLICY,
) -> Optional[Callable[[Dict[str, Any]], bool]]:
    """Build a predicate supporting compound AND/OR."""
    if policy.max_depth is not None and depth > policy.max_depth:
        return None
    for logic_op, combiner in (("and", all), ("or", any)):
        if logic_op in where:
            subs = where[logic_op]
            if not isinstance(subs, list):
                return None
            if policy.max_clauses_per_node is not None:
                subs = subs[: policy.max_clauses_per_node]
            predicates = []
            for sub in subs:
                predicate = build_predicate(sub, col_types, depth + 1, policy)
                if predicate is None:
                    return None
                predicates.append(predicate)
            return lambda row, _preds=predicates, _combiner=combiner: _combiner(p(row) for p in _preds)
    return build_leaf_predicate(
        where.get("column", ""),
        where.get("op", "="),
        where.get("value"),
        col_types.get(where.get("column", ""), T_STRING),
    )


def sort_rows(rows: List[Dict[str, Any]], col: str, col_type: str, desc: bool) -> List[Dict[str, Any]]:
    """Sort rows by column, type-aware."""
    if col_type == T_NUMERIC:
        return sorted(
            rows,
            key=lambda row: (
                (v := to_float_permissive(row.get(col))) is None,
                v or 0,
            ),
            reverse=desc,
        )
    return sorted(rows, key=lambda row: str(row.get(col, "")), reverse=desc)


def apply_sort_slice_project(
    rows: List[Dict[str, Any]],
    slot: Any,
    sort_by: Optional[str],
    sort_desc: bool,
    limit: int,
    columns: Optional[List[str]],
    max_retrieve_rows: int,
) -> List[Dict[str, Any]]:
    """Sort, slice to limit, and optionally project columns."""
    if sort_by and sort_by in slot.col_types:
        rows = sort_rows(rows, sort_by, slot.col_types[sort_by], sort_desc)
    sliced = rows[: min(limit, max_retrieve_rows)]
    if columns:
        column_set = set(columns)
        sliced = [{k: v for k, v in row.items() if k in column_set} for row in sliced]
    return sliced


def filter_rows_by_condition(
    rows: List[Dict[str, Any]],
    column: str,
    op: str,
    value: Any,
) -> List[Dict[str, Any]]:
    """Filter rows by a single {column, op, value} condition.

    Handles numeric comparison with fallback to string equality/inequality.
    """
    result = []
    for row in rows:
        rv = row.get(column)
        if rv is None:
            continue
        try:
            fv = float(rv)
            target_v = float(value)
        except (ValueError, TypeError):
            if op == "=" and str(rv) == str(value):
                result.append(row)
            elif op == "!=" and str(rv) != str(value):
                result.append(row)
            continue

        match = False
        if op == ">" and fv > target_v:
            match = True
        elif op == ">=" and fv >= target_v:
            match = True
        elif op == "<" and fv < target_v:
            match = True
        elif op == "<=" and fv <= target_v:
            match = True
        elif op == "=" and abs(fv - target_v) < 1e-9:
            match = True
        elif op == "!=" and abs(fv - target_v) >= 1e-9:
            match = True
        if match:
            result.append(row)
    return result


__all__ = [
    "DEFAULT_QUERY_BUILD_POLICY",
    "QueryBuildPolicy",
    "apply_sort_slice_project",
    "build_leaf_predicate",
    "build_predicate",
    "filter_rows_by_condition",
    "sort_rows",
]
