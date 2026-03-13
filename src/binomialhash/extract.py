"""Row extraction and raw-structure inspection helpers for BinomialHash."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NestingProfile:
    """Structural topology of the raw JSON before flattening."""

    max_depth: int
    total_nodes: int
    total_leaves: int
    branching_by_depth: Dict[int, float]
    array_lengths_by_depth: Dict[int, List[int]]
    path_signature: str
    nested_key_count: int


def analyze_nesting(data: Any, _depth: int = 0, _max_depth: int = 12) -> NestingProfile:
    """Walk raw parsed JSON and capture structural topology."""
    nodes = 0
    leaves = 0
    branching: Dict[int, List[int]] = {}
    array_lengths: Dict[int, List[int]] = {}
    signature_parts: List[str] = []

    def walk(obj: Any, depth: int) -> None:
        nonlocal nodes, leaves
        if depth > _max_depth:
            return
        nodes += 1
        if isinstance(obj, dict):
            child_count = len(obj)
            branching.setdefault(depth, []).append(child_count)
            if depth < 4:
                signature_parts.append(f"d{depth}:dict({child_count})")
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, list):
            array_lengths.setdefault(depth, []).append(len(obj))
            if depth < 4:
                signature_parts.append(f"d{depth}:list({len(obj)})")
            for item in obj[:200]:
                walk(item, depth + 1)
        else:
            leaves += 1

    walk(data, _depth)
    avg_branching = {depth: round(sum(values) / len(values), 2) for depth, values in branching.items()}
    truncated_arrays = {depth: sorted(set(values))[:10] for depth, values in array_lengths.items()}
    nested_keys = sum(1 for depth, values in branching.items() if depth >= 1 for _ in values)
    max_depth = max(list(branching.keys()) + list(array_lengths.keys()) + [0])
    return NestingProfile(
        max_depth=max_depth,
        total_nodes=nodes,
        total_leaves=leaves,
        branching_by_depth=avg_branching,
        array_lengths_by_depth=truncated_arrays,
        path_signature="→".join(signature_parts[:8]) if signature_parts else "flat",
        nested_key_count=nested_keys,
    )


def find_largest_list(data: Dict[str, Any], depth: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    """Walk up to 3 levels deep to find the largest list-of-dicts."""
    best_key, best_list = "", []
    for key, value in data.items():
        if isinstance(value, list) and len(value) > len(best_list) and value and isinstance(value[0], dict):
            best_key, best_list = key, value
        elif isinstance(value, dict) and depth < 2:
            child_key, child_list = find_largest_list(value, depth + 1)
            if len(child_list) > len(best_list):
                best_key, best_list = f"{key}.{child_key}", child_list
    return best_key, best_list


def flatten_row(row: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts into dot-separated keys."""
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_row(value, full_key))
        else:
            flat[full_key] = value
    return flat


def parse_embedded_jsonish(value: Any) -> Any:
    """Best-effort parse for embedded stringified JSON objects/lists."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one extracted row by parsing embedded JSON-ish string values."""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        parsed = parse_embedded_jsonish(value)
        if isinstance(parsed, dict):
            for flat_key, flat_value in flatten_row(parsed, key).items():
                out[flat_key] = flat_value
        else:
            out[key] = parsed
    return out


def is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(
        isinstance(item, dict) for item in value[: min(len(value), 5)]
    )


def explode_embedded_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If rows contain an embedded list-of-dicts column (or stringified version of one),
    explode it into a flat row set while carrying parent scalar metadata columns.
    """
    if not rows:
        return rows

    candidate_stats: Dict[str, Dict[str, float]] = {}
    sample_rows = rows[: min(len(rows), 50)]
    for row in sample_rows:
        for column, raw_value in row.items():
            value = parse_embedded_jsonish(raw_value)
            if is_list_of_dicts(value):
                stat = candidate_stats.setdefault(column, {"rows": 0, "items": 0})
                stat["rows"] += 1
                stat["items"] += len(value)

    if not candidate_stats:
        return rows

    best_col = max(candidate_stats, key=lambda column: (candidate_stats[column]["rows"], candidate_stats[column]["items"]))
    best = candidate_stats[best_col]
    if best["rows"] < max(2, len(sample_rows) * 0.3):
        return rows

    exploded: List[Dict[str, Any]] = []
    for row in rows:
        value = parse_embedded_jsonish(row.get(best_col))
        if not is_list_of_dicts(value):
            continue
        base = {}
        for key, base_value in row.items():
            if key == best_col:
                continue
            parsed = parse_embedded_jsonish(base_value)
            if isinstance(parsed, dict):
                base.update(flatten_row(parsed, key))
            else:
                base[key] = parsed
        for item in value:
            flat_item = flatten_row(item, best_col)
            merged = dict(base)
            merged.update(flat_item)
            exploded.append(merged)

    return exploded or rows


def extract_rows(data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find the main list-of-dicts in parsed JSON by structure, not key names."""
    meta: Dict[str, Any] = {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        rows = [normalize_row(row) for row in data]
        rows = explode_embedded_table(rows)
        if rows and any(isinstance(value, dict) for value in rows[0].values()):
            rows = [flatten_row(row) for row in rows]
        return rows, meta

    if isinstance(data, dict):
        best_key, best_list = find_largest_list(data)
        if len(best_list) >= 2:
            meta = {key: value for key, value in data.items() if not isinstance(value, (list, dict))}
            rows = [normalize_row(row) for row in best_list]
            rows = explode_embedded_table(rows)
            if rows and any(isinstance(value, dict) for value in rows[0].values()):
                rows = [flatten_row(row) for row in rows]
            logger.info(
                "[BH] extract_rows found %d rows at path '%s' (%d cols)",
                len(rows),
                best_key,
                len(rows[0]) if rows else 0,
            )
            return rows, meta

    return [], meta
