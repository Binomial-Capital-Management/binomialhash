"""Objective-driven insight extraction over tabular data.

Provides: driver discovery, residual surprises, regime boundary detection,
branch divergence scoring, and counterfactual direction guidance.

All functions operate on raw rows — they do not depend on the manifold grid.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from .stats import (
    bucket_index,
    fit_linear,
    numeric_column_values,
    pearson_corr,
    quantile_edges,
    to_float_permissive,
)


def _to_float(v: Any) -> Optional[float]:
    return to_float_permissive(v)


def discover_best_driver(
    rows: List[Dict[str, Any]],
    target: str,
    drivers: List[str],
    min_samples: int = 30,
) -> Optional[Dict[str, Any]]:
    """Fit univariate linear models and return the best driver by R2."""
    target_pairs = numeric_column_values(rows, target)
    if len(target_pairs) < min_samples:
        return None
    target_by_idx = dict(target_pairs)

    best: Optional[Dict[str, Any]] = None
    for driver in drivers:
        pairs = numeric_column_values(rows, driver)
        xs: List[float] = []
        ys: List[float] = []
        for idx, x in pairs:
            y = target_by_idx.get(idx)
            if y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < min_samples:
            continue
        slope, intercept, r2 = fit_linear(xs, ys)
        corr = pearson_corr(xs, ys)
        candidate = {
            "driver": driver,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "corr": corr,
            "samples": len(xs),
        }
        if best is None or candidate["r2"] > best["r2"]:
            best = candidate
    return best


def compute_surprises(
    rows: List[Dict[str, Any]],
    target: str,
    driver: str,
    slope: float,
    intercept: float,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the *top_k* strongest residual contradictions to a fitted law."""
    residuals: List[float] = []
    temp: List[Tuple[int, float, float, float]] = []
    for idx, row in enumerate(rows):
        x = _to_float(row.get(driver))
        y = _to_float(row.get(target))
        if x is None or y is None:
            continue
        expected = slope * x + intercept
        resid = y - expected
        residuals.append(resid)
        temp.append((idx, y, expected, resid))

    resid_std = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0
    surprises: List[Dict[str, Any]] = []
    for idx, y, expected, resid in sorted(temp, key=lambda t: abs(t[3]), reverse=True)[: max(top_k, 1)]:
        z = (resid / resid_std) if resid_std > 1e-9 else 0.0
        surprises.append({
            "row_index": idx,
            "actual": round(y, 6),
            "expected": round(expected, 6),
            "residual": round(resid, 6),
            "residual_z": round(z, 4),
        })
    return surprises


def compute_regime_boundaries(
    rows: List[Dict[str, Any]],
    target: str,
    driver: str,
    bins: int = 10,
    z_threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """Detect abrupt jumps in target means across driver quantile buckets."""
    x_values = [v for _, v in numeric_column_values(rows, driver)]
    edges = quantile_edges(x_values, bins=bins)
    n_buckets = max(len(edges) - 1, 1)
    bucket_sums = [0.0] * n_buckets
    bucket_counts = [0] * n_buckets

    for row in rows:
        x = _to_float(row.get(driver))
        y = _to_float(row.get(target))
        if x is None or y is None:
            continue
        b = bucket_index(x, edges)
        bucket_sums[b] += y
        bucket_counts[b] += 1

    means = [
        bucket_sums[i] / bucket_counts[i] if bucket_counts[i] else 0.0
        for i in range(n_buckets)
    ]
    deltas = [means[i + 1] - means[i] for i in range(len(means) - 1)] if len(means) > 1 else []
    delta_std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0

    boundaries: List[Dict[str, Any]] = []
    for i, d in enumerate(deltas):
        z = (d / delta_std) if delta_std > 1e-9 else 0.0
        if abs(z) >= z_threshold:
            hi = round(edges[i + 2], 6) if i + 2 < len(edges) else round(edges[-1], 6)
            boundaries.append({
                "between_bucket": [i, i + 1],
                "driver_range": [round(edges[i], 6), hi],
                "target_jump": round(d, 6),
                "jump_z": round(z, 4),
            })
    return boundaries


def compute_branch_divergence(
    rows: List[Dict[str, Any]],
    target: str,
    numeric_cols: List[str],
    *,
    target_bins: int = 6,
    context_limit: int = 8,
    min_rows: int = 20,
    min_values: int = 10,
) -> List[Dict[str, Any]]:
    """Score how much latent context spreads within each target quantile bucket."""
    target_pairs = numeric_column_values(rows, target)
    t_values = [v for _, v in target_pairs]
    t_edges = quantile_edges(t_values, bins=target_bins)
    context_cols = [c for c in numeric_cols if c != target][:context_limit]

    divergences: List[Dict[str, Any]] = []
    for b in range(max(len(t_edges) - 1, 1)):
        members = []
        for row in rows:
            y = _to_float(row.get(target))
            if y is None:
                continue
            if bucket_index(y, t_edges) == b:
                members.append(row)
        if len(members) < min_rows:
            continue

        spread_scores: List[float] = []
        for c in context_cols:
            vals = [v for v in (_to_float(r.get(c)) for r in members) if v is not None]
            if len(vals) < min_values:
                continue
            vmin, vmax = min(vals), max(vals)
            mean_abs = abs(sum(vals) / len(vals)) or 1.0
            spread_scores.append((vmax - vmin) / mean_abs)

        if spread_scores:
            hi = round(t_edges[b + 1], 6) if b + 1 < len(t_edges) else round(t_edges[-1], 6)
            divergences.append({
                "target_quantile_bucket": b,
                "target_range": [round(t_edges[b], 6), hi],
                "rows": len(members),
                "latent_spread_score": round(sum(spread_scores) / len(spread_scores), 6),
            })
    divergences.sort(key=lambda x: x["latent_spread_score"], reverse=True)
    return divergences


def build_counterfactual(
    rows: List[Dict[str, Any]],
    target: str,
    driver: str,
    model: Dict[str, Any],
    objective: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a counterfactual suggestion from a fitted linear law."""
    slope = model["slope"]
    intercept = model["intercept"]
    goal = str(objective.get("goal", "maximize")).lower()

    cf: Dict[str, Any] = {
        "driver": driver,
        "target": target,
        "law": {
            "formula": f"{target} \u2248 {slope:.6f} * {driver} + {intercept:.6f}",
            "r2": round(model["r2"], 4),
            "corr": round(model["corr"], 4),
        },
    }

    if goal == "minimize":
        direction = "increase" if slope < 0 else "decrease"
    else:
        direction = "increase" if slope > 0 else "decrease"

    cf["suggested_action"] = {
        "goal": goal,
        "driver_direction": direction,
        "rationale": "Direction chosen from fitted slope sign under stated objective.",
    }

    target_value = _to_float(objective.get("target_value"))
    target_pairs = numeric_column_values(rows, target)
    if target_pairs and target_value is not None and abs(slope) > 1e-9:
        current_mean = sum(v for _, v in target_pairs) / len(target_pairs)
        required_driver = (target_value - intercept) / slope
        driver_pairs = numeric_column_values(rows, driver)
        driver_mean = sum(v for _, v in driver_pairs) / len(driver_pairs) if driver_pairs else 0.0
        cf["reach_value_estimate"] = {
            "current_target_mean": round(current_mean, 6),
            "desired_target_value": round(target_value, 6),
            "required_driver_value": round(required_driver, 6),
            "delta_driver_from_mean": round(required_driver - driver_mean, 6),
        }
    return cf


def compute_insights(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    objective: Dict[str, Any],
    *,
    top_k: int = 5,
    driver_limit: int = 20,
    driver_bins: int = 10,
    regime_z_threshold: float = 1.5,
    target_bins: int = 6,
    branch_context_limit: int = 8,
    branch_min_rows: int = 20,
    branch_min_values: int = 10,
) -> Dict[str, Any]:
    """Full objective-driven insight pipeline over tabular rows.

    Returns a dict with ``discovered_law``, ``insights`` (surprises,
    regime_boundaries, branch_divergence, counterfactual), and
    ``method_notes``.  Returns ``{"error": ...}`` on failure.
    """
    from .schema import T_NUMERIC

    numeric_cols = [c for c in columns if col_types.get(c) == T_NUMERIC]
    if not numeric_cols:
        return {"error": "No numeric columns available for insights."}

    target = objective.get("target")
    if target not in numeric_cols:
        target = numeric_cols[0]

    target_pairs = numeric_column_values(rows, target)
    if len(target_pairs) < 30:
        return {"error": f"Insufficient numeric data for target '{target}' (need >=30 rows)."}

    drivers = [c for c in numeric_cols if c != target][:driver_limit]
    best = discover_best_driver(rows, target, drivers)
    if best is None:
        return {"error": f"Could not fit a law for target '{target}'."}

    driver = best["driver"]
    goal = str(objective.get("goal", "maximize")).lower()

    surprises = compute_surprises(rows, target, driver, best["slope"], best["intercept"], top_k)
    boundaries = compute_regime_boundaries(rows, target, driver, bins=driver_bins, z_threshold=regime_z_threshold)
    divergence = compute_branch_divergence(
        rows, target, numeric_cols,
        target_bins=target_bins,
        context_limit=branch_context_limit,
        min_rows=branch_min_rows,
        min_values=branch_min_values,
    )
    counterfactual = build_counterfactual(rows, target, driver, best, objective)

    return {
        "objective": {
            "target": target,
            "goal": goal,
            "target_value": objective.get("target_value"),
        },
        "discovered_law": {
            "driver": driver,
            "target": target,
            "slope": round(best["slope"], 6),
            "intercept": round(best["intercept"], 6),
            "r2": round(best["r2"], 4),
            "corr": round(best["corr"], 4),
            "samples": best["samples"],
        },
        "insights": {
            "surprises": surprises,
            "regime_boundaries": boundaries[: max(top_k, 1)],
            "branch_divergence": divergence[: max(top_k, 1)],
            "counterfactual": counterfactual,
        },
        "method_notes": [
            "Insights are objective-driven and deterministic from fitted structural laws.",
            "No hardcoded shape labels are used; transitions emerge from data quantiles and jumps.",
            "Branch divergence flags non-equivalent contexts at similar headline target levels.",
        ],
    }
