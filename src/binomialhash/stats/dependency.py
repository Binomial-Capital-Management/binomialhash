"""Dependency mapping and independence tests."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

from ..schema import T_NUMERIC
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    bucket_index,
    extract_numeric_pairs,
    normal_cdf,
    np,
    pearson_corr,
    quantile_edges,
    shannon_entropy,
    spearman_rank,
    to_float_permissive,
)


def rank_corr_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    method: str = "spearman",
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Spearman rank correlation matrix with Pearson comparison."""
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 numeric fields."}

    from ._helpers import extract_numeric_matrix
    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < policy.rank_corr_min_samples:
        return {"error": f"Not enough complete rows ({n})."}

    pairs = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            xs = [mat[r][i] for r in range(n)]
            ys = [mat[r][j] for r in range(n)]
            p_corr = pearson_corr(xs, ys)
            rx = spearman_rank(xs)
            ry = spearman_rank(ys)
            s_corr = pearson_corr(rx, ry)
            div = abs(s_corr - p_corr)
            pairs.append({
                "field_a": fields[i], "field_b": fields[j],
                "spearman": round(s_corr, 4), "pearson": round(p_corr, 4),
                "divergence": round(div, 4),
            })
    pairs.sort(key=lambda x: x["divergence"], reverse=True)
    # Empirically tuned: Spearman-Pearson divergence > 0.15 suggests nonlinearity.
    nonlinear = [p for p in pairs if p["divergence"] > 0.15]

    return {
        "fields": fields, "samples": n, "method": method,
        "pairs": pairs, "nonlinear_pairs": nonlinear,
    }


def chi_squared_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field_a: str,
    field_b: str,
    bins: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Chi-squared test of independence between two columns."""
    if field_a not in col_types or field_b not in col_types:
        return {"error": f"Field(s) not found."}
    bins = bins or policy.chi_squared_default_bins

    vals_a, vals_b = [], []
    for r in rows:
        a, b = r.get(field_a), r.get(field_b)
        if a is not None and b is not None:
            vals_a.append(a)
            vals_b.append(b)
    n = len(vals_a)
    if n < 10:
        return {"error": f"Not enough paired values ({n})."}

    def _categorize(vals, b):
        nums = [to_float_permissive(v) for v in vals]
        if sum(1 for x in nums if x is not None) > n * 0.8:
            clean = [x for x in nums if x is not None]
            edges = quantile_edges(clean, b)
            return [str(bucket_index(x, edges)) if x is not None else "NA" for x in nums]
        return [str(v) for v in vals]

    cat_a = _categorize(vals_a, bins)
    cat_b = _categorize(vals_b, bins)
    labels_a = sorted(set(cat_a))
    labels_b = sorted(set(cat_b))
    ra = len(labels_a)
    rb = len(labels_b)
    if ra < 2 or rb < 2:
        return {"error": "Each field must have at least 2 distinct categories."}

    idx_a = {v: i for i, v in enumerate(labels_a)}
    idx_b = {v: i for i, v in enumerate(labels_b)}
    table = [[0] * rb for _ in range(ra)]
    for a, b in zip(cat_a, cat_b):
        table[idx_a[a]][idx_b[b]] += 1

    row_sums = [sum(table[i]) for i in range(ra)]
    col_sums = [sum(table[i][j] for i in range(ra)) for j in range(rb)]
    chi2 = 0.0
    for i in range(ra):
        for j in range(rb):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected

    dof = (ra - 1) * (rb - 1)
    from .quality import _chi2_pvalue
    p_value = _chi2_pvalue(chi2, dof) if dof > 0 else 1.0
    cramers_v = math.sqrt(chi2 / (n * (min(ra, rb) - 1))) if n > 0 and min(ra, rb) > 1 else 0.0

    return {
        "field_a": field_a, "field_b": field_b, "samples": n,
        "chi_squared": round(chi2, 4), "dof": dof,
        "p_value": round(p_value, 6), "cramers_v": round(cramers_v, 4),
        "categories_a": ra, "categories_b": rb,
    }


def anova_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    group_field: str,
    target_field: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """One-way ANOVA: does a categorical group explain variance in a numeric target?"""
    if group_field not in col_types or target_field not in col_types:
        return {"error": "Field(s) not found."}

    groups: Dict[str, List[float]] = {}
    for r in rows:
        g = r.get(group_field)
        v = to_float_permissive(r.get(target_field))
        if g is not None and v is not None:
            groups.setdefault(str(g), []).append(v)

    groups = {k: v for k, v in groups.items() if len(v) >= policy.anova_min_group_size}
    if len(groups) < policy.anova_min_groups:
        return {"error": f"Need at least {policy.anova_min_groups} groups with >= {policy.anova_min_group_size} members."}

    all_vals = [v for vs in groups.values() for v in vs]
    grand_mean = sum(all_vals) / len(all_vals)
    n_total = len(all_vals)
    k = len(groups)

    ss_between = sum(len(vs) * (sum(vs) / len(vs) - grand_mean) ** 2 for vs in groups.values())
    ss_within = sum(sum((v - sum(vs) / len(vs)) ** 2 for v in vs) for vs in groups.values())

    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0 or df_between <= 0:
        return {"error": "Not enough degrees of freedom."}

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within if ms_within > 1e-12 else 0.0
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 1e-12 else 0.0

    p_value = _f_pvalue(f_stat, df_between, df_within)

    group_stats = []
    for name, vs in sorted(groups.items()):
        m = sum(vs) / len(vs)
        s = (sum((v - m) ** 2 for v in vs) / len(vs)) ** 0.5
        group_stats.append({"group": name, "n": len(vs), "mean": round(m, 4), "std": round(s, 4)})

    return {
        "group_field": group_field, "target_field": target_field,
        "n_groups": k, "samples": n_total,
        "f_statistic": round(f_stat, 4), "p_value": round(p_value, 6),
        "eta_squared": round(eta_squared, 4), "group_stats": group_stats,
    }


def _f_pvalue(f: float, df1: int, df2: int) -> float:
    """Approximate F-distribution p-value using the Wilson-Hilferty transform."""
    if f <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    a = 2.0 / (9.0 * df1)
    b = 2.0 / (9.0 * df2)
    z = ((1.0 - b) * f ** (1.0 / 3.0) - (1.0 - a)) / math.sqrt(b * f ** (2.0 / 3.0) + a)
    return max(0.0, min(1.0, 1.0 - normal_cdf(z)))


def mutual_info_matrix_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    bins: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Pairwise mutual information matrix for selected columns."""
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 numeric fields."}

    bins = bins or policy.mi_default_bins
    from ._helpers import extract_numeric_matrix
    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < policy.mi_min_samples:
        return {"error": f"Not enough complete rows ({n})."}

    col_edges = []
    for i in range(len(fields)):
        col_vals = sorted(mat[r][i] for r in range(n))
        col_edges.append(quantile_edges(col_vals, bins))

    def _mi(ci: int, cj: int) -> float:
        joint = [[0] * bins for _ in range(bins)]
        for r in range(n):
            bi = bucket_index(mat[r][ci], col_edges[ci])
            bj = bucket_index(mat[r][cj], col_edges[cj])
            joint[bi][bj] += 1
        marg_i = [sum(joint[bi]) for bi in range(bins)]
        marg_j = [sum(joint[bi][bj] for bi in range(bins)) for bj in range(bins)]
        h_i = shannon_entropy(marg_i)
        h_j = shannon_entropy(marg_j)
        h_ij = shannon_entropy([joint[bi][bj] for bi in range(bins) for bj in range(bins)])
        return max(0.0, h_i + h_j - h_ij)

    pairs = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            mi = _mi(i, j)
            xs = [mat[r][i] for r in range(n)]
            ys = [mat[r][j] for r in range(n)]
            p_corr = abs(pearson_corr(xs, ys))
            h_i = shannon_entropy([sum(1 for r in range(n) if bucket_index(mat[r][i], col_edges[i]) == b) for b in range(bins)])
            h_j = shannon_entropy([sum(1 for r in range(n) if bucket_index(mat[r][j], col_edges[j]) == b) for b in range(bins)])
            # NMI = MI/min(H(X),H(Y)) for scale-invariant comparison.
            nmi = mi / min(h_i, h_j) if min(h_i, h_j) > 1e-12 else 0.0
            pairs.append({
                "field_a": fields[i], "field_b": fields[j],
                "mutual_info": round(mi, 6), "normalized_mi": round(nmi, 4),
                "pearson_abs": round(p_corr, 4),
                "nonlinear_flag": nmi > 0.1 and nmi > p_corr * 1.5,
            })
    pairs.sort(key=lambda x: x["mutual_info"], reverse=True)
    nonlinear = [p for p in pairs if p["nonlinear_flag"]]

    return {
        "fields": fields, "samples": n, "bins": bins,
        "pairs": pairs, "nonlinear_pairs": nonlinear,
    }


def hsic_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field_a: str,
    field_b: str,
    kernel: str = "gaussian",
    n_permutations: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Hilbert-Schmidt Independence Criterion with permutation test."""
    if np is None:
        return {"error": "numpy required for HSIC."}
    xs, ys = extract_numeric_pairs(rows, field_a, field_b)
    n = len(xs)
    if n < policy.hsic_min_samples:
        return {"error": f"Not enough paired values ({n})."}

    n_perm = n_permutations or policy.hsic_default_permutations
    xa = np.array(xs).reshape(-1, 1)
    ya = np.array(ys).reshape(-1, 1)

    def _kernel_matrix(data):
        sq = np.sum(data ** 2, axis=1, keepdims=True)
        dists = sq + sq.T - 2 * data @ data.T
        sigma = float(np.median(np.sqrt(np.maximum(dists, 0))))
        if sigma < 1e-12:
            sigma = 1.0
        return np.exp(-dists / (2 * sigma ** 2))

    kx = _kernel_matrix(xa)
    ky = _kernel_matrix(ya)
    h = np.eye(n) - np.ones((n, n)) / n
    hkx = h @ kx @ h
    hky = h @ ky @ h
    hsic_stat = float(np.trace(hkx @ hky)) / (n ** 2)

    null_dist = []
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        ky_perm = ky[perm][:, perm]
        hky_perm = h @ ky_perm @ h
        null_dist.append(float(np.trace(hkx @ hky_perm)) / (n ** 2))

    p_value = sum(1 for s in null_dist if s >= hsic_stat) / max(len(null_dist), 1)
    max_hsic = math.sqrt(float(np.trace(hkx @ hkx)) * float(np.trace(hky @ hky))) / (n ** 2)
    normalized = hsic_stat / max_hsic if max_hsic > 1e-12 else 0.0

    from ._helpers import spearman_rank
    rx = spearman_rank(xs)
    ry = spearman_rank(ys)

    return {
        "field_a": field_a, "field_b": field_b, "samples": n,
        "hsic_statistic": round(hsic_stat, 8), "p_value": round(p_value, 4),
        "normalized_hsic": round(normalized, 4),
        "is_dependent": p_value < 0.05,
        "comparison": {
            "pearson": round(pearson_corr(xs, ys), 4),
            "spearman": round(pearson_corr(rx, ry), 4),
        },
    }


def copula_tail_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field_a: str,
    field_b: str,
    tail_threshold: Optional[float] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Tail dependence via empirical copula."""
    xs, ys = extract_numeric_pairs(rows, field_a, field_b)
    n = len(xs)
    if n < policy.copula_min_samples:
        return {"error": f"Not enough paired values ({n})."}

    tail = tail_threshold or policy.copula_default_tail
    rx = spearman_rank(xs)
    ry = spearman_rank(ys)
    ux = [r / (n + 1) for r in rx]
    uy = [r / (n + 1) for r in ry]

    # P(U > 1-t, V > 1-t) / t
    upper_count = sum(1 for u, v in zip(ux, uy) if u > 1 - tail and v > 1 - tail)
    upper_tail = upper_count / (n * tail) if n * tail > 0 else 0.0

    # P(U < t, V < t) / t
    lower_count = sum(1 for u, v in zip(ux, uy) if u < tail and v < tail)
    lower_tail = lower_count / (n * tail) if n * tail > 0 else 0.0

    # Kendall's tau
    concordant = discordant = 0
    # Kendall's tau: O(n^2) pairs; cap at 500 per point to avoid quadratic blowup.
    for i in range(n):
        for j in range(i + 1, min(i + 500, n)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total_pairs = concordant + discordant
    kendall = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    thresholds = [0.01, 0.05, 0.10, 0.25, 0.50]
    profile = []
    for t in thresholds:
        uc = sum(1 for u, v in zip(ux, uy) if u > 1 - t and v > 1 - t)
        lc = sum(1 for u, v in zip(ux, uy) if u < t and v < t)
        profile.append({
            "threshold": t,
            "upper": round(uc / (n * t), 4) if n * t > 0 else 0,
            "lower": round(lc / (n * t), 4) if n * t > 0 else 0,
        })

    return {
        "field_a": field_a, "field_b": field_b, "samples": n,
        "upper_tail_dependence": round(upper_tail, 4),
        "lower_tail_dependence": round(lower_tail, 4),
        "tail_asymmetry": round(upper_tail - lower_tail, 4),
        "kendall_tau": round(kendall, 4),
        "spearman_rho": round(pearson_corr(
            spearman_rank(xs), spearman_rank(ys)
        ), 4),
        "pearson_r": round(pearson_corr(xs, ys), 4),
        "quantile_profile": profile,
    }
