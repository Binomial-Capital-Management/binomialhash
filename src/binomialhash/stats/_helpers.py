"""Shared helpers, numeric coercion, aggregation, and StatsPolicy."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

_NUMERIC_FUNCS = frozenset(("sum", "mean", "median", "min", "max", "std"))
_ALL_AGG_FUNCS = _NUMERIC_FUNCS | frozenset(("count", "count_distinct"))


@dataclass(frozen=True)
class StatsPolicy:
    """Named statistical policy values for BH analysis routines."""

    error_preview_column_limit: int = 20
    regression_min_extra_samples: int = 2
    partial_corr_min_extra_samples: int = 3
    pca_default_component_count: int = 3
    pca_default_field_count: int = 10
    pca_min_field_count: int = 2
    pca_min_complete_rows: int = 5
    pca_top_loading_count: int = 5
    dependency_default_top_k: int = 8
    dependency_min_extra_rows: int = 3
    dependency_score_partial_weight: float = 0.45
    dependency_score_raw_weight: float = 0.35
    dependency_score_coef_weight: float = 0.20
    dependency_score_coef_clip: float = 1.0
    dependency_rank_cap: int = 50
    solver_default_top_k: int = 5
    solver_solution_cap: int = 50

    distribution_default_bins: int = 15
    distribution_min_values: int = 5
    outlier_default_zscore: float = 3.0
    outlier_default_iqr_multiplier: float = 1.5
    outlier_max_flagged: int = 50
    benford_min_values: int = 50
    vif_high_threshold: float = 10.0
    effective_dim_default_k: int = 5
    effective_dim_min_fields: int = 3
    effective_dim_min_rows: int = 10

    rank_corr_min_samples: int = 5
    chi_squared_default_bins: int = 10
    chi_squared_min_expected: float = 1.0
    anova_min_group_size: int = 2
    anova_min_groups: int = 2
    mi_default_bins: int = 10
    mi_min_samples: int = 20
    hsic_default_permutations: int = 100
    hsic_min_samples: int = 10
    copula_default_tail: float = 0.05
    copula_min_samples: int = 30

    polynomial_min_samples: int = 10
    interaction_default_top_k: int = 5
    interaction_min_samples: int = 15
    sparse_default_max_features: int = 10
    sparse_cv_folds: int = 5
    sparse_n_alphas: int = 20
    sparse_max_iter: int = 1000
    importance_n_shuffles: int = 10
    ib_default_clusters: int = 5
    ib_default_beta: float = 1.0
    ib_max_iter: int = 100

    cluster_max_k: int = 8
    cluster_max_iter: int = 100
    cluster_n_init: int = 5
    spectral_default_neighbors: int = 10
    spectral_default_components: int = 5
    ica_max_iter: int = 200
    ica_tol: float = 1e-4
    graphical_default_alpha: float = 0.01
    graphical_glasso_max_iter: int = 100
    topology_max_points: int = 200
    topology_n_thresholds: int = 50

    causal_default_alpha: float = 0.05
    causal_max_conditioning: int = 3
    causal_min_samples: int = 20
    te_default_bins: int = 8
    te_max_lag: int = 5
    te_surrogates: int = 100
    do_default_bins: int = 5
    do_min_per_stratum: int = 5
    synth_min_pre_periods: int = 5

    acf_max_lag: int = 20
    acf_min_samples: int = 30
    changepoint_min_segment: int = 10
    changepoint_default_threshold: float = 2.0
    rolling_default_window: int = 20
    phase_max_embedding: int = 10
    phase_fnn_threshold: float = 15.0
    recurrence_default_embed: int = 3

    entropy_max_scale: int = 10
    entropy_default_embed: int = 2
    renorm_max_scale: int = 5


DEFAULT_STATS_POLICY = StatsPolicy()


def to_float_permissive(v: Any) -> Optional[float]:
    """Coerce a value to float by stripping non-numeric chars discovered from the value."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except ValueError:
            pass
        cleaned = "".join(c for c in v if c.isdigit() or c in ".+-eE")
        if cleaned:
            try:
                f = float(cleaned)
                if math.isfinite(f):
                    return f
            except ValueError:
                pass
    return None


def agg_numeric(nums: List[float], func: str) -> Optional[float]:
    """Compute a numeric aggregate on a non-empty list."""
    assert nums, "nums must not be empty"
    if func == "sum":
        return round(sum(nums), 4)
    if func == "mean":
        return round(sum(nums) / len(nums), 4)
    if func == "min":
        return min(nums)
    if func == "max":
        return max(nums)
    if func == "median":
        sorted_values = sorted(nums)
        mid = len(sorted_values) // 2
        return (
            sorted_values[mid]
            if len(sorted_values) % 2
            else round((sorted_values[mid - 1] + sorted_values[mid]) / 2, 4)
        )
    if func == "std":
        mean = sum(nums) / len(nums)
        return round((sum((x - mean) ** 2 for x in nums) / len(nums)) ** 0.5, 4)
    return None


def run_agg(rows: List[Dict[str, Any]], col: str, func: str) -> Any:
    """Run a single aggregation over rows. Shared by aggregate() and group_by()."""
    if func in _NUMERIC_FUNCS:
        nums = [n for n in (to_float_permissive(row.get(col)) for row in rows) if n is not None]
        return agg_numeric(nums, func) if nums else None
    if func == "count":
        return len([row for row in rows if row.get(col) is not None and row.get(col) != ""])
    if func == "count_distinct":
        return len(set(str(row.get(col)) for row in rows if row.get(col) is not None))
    return None


def numeric_column_values(rows: List[Dict[str, Any]], column: str) -> List[Tuple[int, float]]:
    """Return (row_idx, numeric_value) pairs for rows where value parses as float."""
    out: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        val = to_float_permissive(row.get(column))
        if val is not None:
            out.append((idx, val))
    return out


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation with safe fallbacks for constant vectors."""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = den_x * den_y
    if denominator == 0:
        return 0.0
    return max(min(numerator / denominator, 1.0), -1.0)


def fit_linear(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Fit y = slope*x + intercept. Returns slope, intercept, r2."""
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return 0.0, mean_y, 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    corr = pearson_corr(xs, ys)
    return slope, intercept, corr * corr


def quantile_edges(values: List[float], bins: int) -> List[float]:
    """Build deterministic quantile edges from sorted values."""
    if not values:
        return []
    sorted_values = sorted(values)
    n = len(sorted_values)
    edges: List[float] = []
    for b in range(bins + 1):
        pos = int(round((n - 1) * (b / bins)))
        edges.append(sorted_values[pos])
    for idx in range(1, len(edges)):
        if edges[idx] < edges[idx - 1]:
            edges[idx] = edges[idx - 1]
    return edges


def bucket_index(val: float, edges: List[float]) -> int:
    """Return bucket index in [0, len(edges)-2]."""
    if len(edges) < 2:
        return 0
    for idx in range(len(edges) - 1):
        if val <= edges[idx + 1]:
            return idx
    return len(edges) - 2


def normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def extract_numeric_pairs(
    rows: List[Dict[str, Any]], col_a: str, col_b: str,
) -> Tuple[List[float], List[float]]:
    """Extract paired numeric values for two columns, skipping missing."""
    xs, ys = [], []
    for row in rows:
        a = to_float_permissive(row.get(col_a))
        b = to_float_permissive(row.get(col_b))
        if a is not None and b is not None:
            xs.append(a)
            ys.append(b)
    return xs, ys


def extract_numeric_matrix(
    rows: List[Dict[str, Any]], fields: List[str],
) -> List[List[float]]:
    """Extract rows where all fields are numeric. Returns list of float-lists."""
    out: List[List[float]] = []
    for row in rows:
        vals: List[float] = []
        ok = True
        for f in fields:
            v = to_float_permissive(row.get(f))
            if v is None:
                ok = False
                break
            vals.append(v)
        if ok:
            out.append(vals)
    return out


def spearman_rank(vals: List[float]) -> List[float]:
    """Return rank array (1-based, averaging ties)."""
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy in nats from a count histogram."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def ols_r2(xs_all: List[List[float]], ys: List[float]) -> float:
    """Quick OLS R-squared using normal equations with Gaussian elimination."""
    n = len(ys)
    p = len(xs_all[0]) if xs_all else 0
    if n < p + 2:
        return 0.0
    xtx = [[0.0] * (p + 1) for _ in range(p + 1)]
    xty = [0.0] * (p + 1)
    for i in range(n):
        row_ext = [1.0] + xs_all[i]
        for a in range(p + 1):
            xty[a] += row_ext[a] * ys[i]
            for b in range(p + 1):
                xtx[a][b] += row_ext[a] * row_ext[b]
    aug = [xtx[i][:] + [xty[i]] for i in range(p + 1)]
    for col in range(p + 1):
        max_row = col
        for ri in range(col + 1, p + 1):
            if abs(aug[ri][col]) > abs(aug[max_row][col]):
                max_row = ri
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return 0.0
        for ri in range(col + 1, p + 1):
            factor = aug[ri][col] / aug[col][col]
            for j in range(col, p + 2):
                aug[ri][j] -= factor * aug[col][j]
    beta = [0.0] * (p + 1)
    for i in range(p, -1, -1):
        beta[i] = aug[i][p + 1]
        for j in range(i + 1, p + 1):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]
    mean_y = sum(ys) / n
    ss_res = sum((ys[i] - (beta[0] + sum(beta[j + 1] * xs_all[i][j] for j in range(p)))) ** 2 for i in range(n))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
