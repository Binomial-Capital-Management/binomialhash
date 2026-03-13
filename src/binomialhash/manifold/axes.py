"""Axis discovery helpers for BinomialHash manifold construction."""

import math
from typing import Any, Dict, List, Tuple

from .structures import ManifoldAxis


def identify_axes(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    col_stats: Dict[str, Dict[str, Any]],
) -> Tuple[List[ManifoldAxis], List[str]]:
    """Separate columns into parameter-space axes vs numeric field values."""
    n = len(rows)
    if n == 0:
        return [], []

    axes: List[ManifoldAxis] = []
    fields: List[str] = []

    uniques: Dict[str, List[Any]] = {}
    unique_counts: Dict[str, int] = {}
    value_counts: Dict[str, Dict[str, int]] = {}

    for col in columns:
        seen = set()
        vals: List[Any] = []
        freq: Dict[str, int] = {}
        for r in rows:
            v = r.get(col)
            if v is None or v == "":
                continue
            sv = str(v)
            freq[sv] = freq.get(sv, 0) + 1
            if sv not in seen:
                seen.add(sv)
                vals.append(v)
        uniques[col] = vals
        unique_counts[col] = len(vals)
        value_counts[col] = freq

    def _entropy_norm(freq: Dict[str, int]) -> float:
        total = sum(freq.values())
        if total <= 0 or len(freq) <= 1:
            return 0.0
        probs = [v / total for v in freq.values()]
        h = -sum(p * math.log(max(p, 1e-12), 2) for p in probs)
        h_max = math.log(len(freq), 2)
        return h / h_max if h_max > 0 else 0.0

    def _distinct_tuple(cols: List[str]) -> int:
        if not cols:
            return 1
        s = set()
        for r in rows:
            s.add(tuple(str(r.get(c, "")) for c in cols))
        return len(s)

    seed_axes: List[str] = []
    for col in columns:
        ct = col_types.get(col, "string")
        u = unique_counts.get(col, 0)
        if u < 2:
            continue
        if ct in {"date", "string", "bool"}:
            seed_axes.append(col)

    selected_axes: List[str] = []
    for col in seed_axes:
        u = unique_counts[col]
        if 2 <= u <= max(2, int(math.sqrt(n) * 6)):
            selected_axes.append(col)

    numeric_candidates = [
        c for c in columns if col_types.get(c) == "numeric" and unique_counts.get(c, 0) >= 2
    ]
    best_num_axis = None
    best_score = -1.0

    base_prod = 1
    for c in selected_axes:
        base_prod *= max(unique_counts.get(c, 1), 1)

    for c in numeric_candidates:
        u = unique_counts[c]
        freq = value_counts[c]
        repeats = n / max(u, 1)
        ent = _entropy_norm(freq)

        cols_for_cov = selected_axes + [c]
        distinct_tuples = _distinct_tuple(cols_for_cov)
        expected = max(base_prod * u, 1)
        coverage = distinct_tuples / expected

        score = coverage * math.log1p(repeats) * (0.5 + 0.5 * ent) * math.log1p(u)
        if score > best_score:
            best_score = score
            best_num_axis = c

    if best_num_axis is not None:
        test_prod = base_prod * max(unique_counts.get(best_num_axis, 1), 1)
        test_occupancy = n / max(test_prod, 1)
        if test_occupancy >= 0.3:
            selected_axes.append(best_num_axis)

    for col in selected_axes:
        vals = uniques[col][:]
        ct = col_types.get(col, "string")
        if ct == "numeric":
            axis_type = "numeric_ordered"
            ordered = True
            try:
                vals = sorted(vals, key=lambda x: float(x))
            except (ValueError, TypeError):
                vals = sorted(vals, key=str)
        elif ct == "date":
            axis_type = "temporal"
            ordered = True
            vals = sorted(vals, key=str)
        else:
            axis_type = "categorical"
            ordered = False
        axes.append(
            ManifoldAxis(
                column=col,
                values=vals,
                ordered=ordered,
                axis_type=axis_type,
                size=len(vals),
            )
        )

    selected_axis_set = {a.column for a in axes}
    fields = [c for c in columns if col_types.get(c) == "numeric" and c not in selected_axis_set]

    if not fields:
        numeric_axes = [a for a in axes if a.axis_type == "numeric_ordered"]
        numeric_axes.sort(key=lambda a: a.size, reverse=True)
        while numeric_axes and not fields:
            demote = numeric_axes.pop(0)
            axes = [a for a in axes if a.column != demote.column]
            fields.append(demote.column)

    axes.sort(key=lambda a: a.size)
    return axes[:6], fields[:20]
