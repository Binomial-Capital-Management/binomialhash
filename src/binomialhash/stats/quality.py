"""Data quality profiling and diagnostics."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from ..schema import T_NUMERIC
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    extract_numeric_matrix,
    np,
    pearson_corr,
    to_float_permissive,
)


def _quantile(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def distribution_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    bins: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Full distributional profile of a single numeric column."""
    if field not in col_types:
        return {"error": f"Field '{field}' not found."}
    vals = [v for v in (to_float_permissive(r.get(field)) for r in rows) if v is not None]
    n = len(vals)
    if n < policy.distribution_min_values:
        return {"error": f"Not enough numeric values ({n}) for distribution profile."}

    bins = bins or policy.distribution_default_bins
    sv = sorted(vals)
    mean = sum(vals) / n
    median_val = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in vals) / n
    std = var ** 0.5 if var > 0 else 0.0

    if std > 1e-12:
        skewness = sum(((x - mean) / std) ** 3 for x in vals) / n
        kurtosis = sum(((x - mean) / std) ** 4 for x in vals) / n - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    quantiles = {
        "p5": round(_quantile(sv, 0.05), 6),
        "p25": round(_quantile(sv, 0.25), 6),
        "p50": round(_quantile(sv, 0.50), 6),
        "p75": round(_quantile(sv, 0.75), 6),
        "p95": round(_quantile(sv, 0.95), 6),
    }

    lo_val, hi_val = sv[0], sv[-1]
    bin_width = (hi_val - lo_val) / bins if hi_val > lo_val else 1.0
    hist_counts = [0] * bins
    hist_edges = [round(lo_val + i * bin_width, 6) for i in range(bins + 1)]
    for v in vals:
        idx = min(int((v - lo_val) / bin_width), bins - 1) if bin_width > 0 else 0
        hist_counts[idx] += 1

    # Jarque-Bera: JB = n/6*(S^2 + K^2/4); p-value ~ exp(-JB/2) under chi2(2).
    jb = n / 6.0 * (skewness ** 2 + kurtosis ** 2 / 4.0)
    normality_pvalue = round(math.exp(-jb / 2.0), 6)

    # Empirically tuned skewness/kurtosis cutoffs for shape classification.
    if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
        shape = "normal"
    elif skewness > 1.0:
        shape = "right_skewed"
    elif skewness < -1.0:
        shape = "left_skewed"
    elif kurtosis > 3.0:
        shape = "heavy_tailed"
    elif abs(skewness) < 0.3 and kurtosis < -1.0:
        shape = "uniform"
    else:
        peaks = sum(
            1 for i in range(1, bins - 1)
            if hist_counts[i] > hist_counts[i - 1] and hist_counts[i] > hist_counts[i + 1]
        )
        shape = "bimodal" if peaks >= 2 else "moderate_skew"

    return {
        "field": field, "count": n,
        "mean": round(mean, 6), "median": round(median_val, 6), "std": round(std, 6),
        "skewness": round(skewness, 4), "kurtosis": round(kurtosis, 4),
        "min": round(sv[0], 6), "max": round(sv[-1], 6),
        "quantiles": quantiles,
        "histogram": {"edges": hist_edges, "counts": hist_counts},
        "normality_pvalue": normality_pvalue, "shape_label": shape,
    }


def outliers_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    method: str = "both",
    threshold: Optional[float] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Flag anomalous rows via z-score, IQR, or both."""
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:20]
    fields = [f for f in fields if f in numeric_cols]
    if not fields:
        return {"error": "No valid numeric fields."}

    z_thresh = threshold or policy.outlier_default_zscore
    iqr_mult = policy.outlier_default_iqr_multiplier
    field_summaries = []
    row_scores: Dict[int, float] = {}

    for fld in fields:
        vals = [(i, v) for i, r in enumerate(rows) for v in [to_float_permissive(r.get(fld))] if v is not None]
        if len(vals) < 5:
            continue
        numbers = [v for _, v in vals]
        mean = sum(numbers) / len(numbers)
        std = (sum((x - mean) ** 2 for x in numbers) / len(numbers)) ** 0.5
        sv = sorted(numbers)
        q1 = _quantile(sv, 0.25)
        q3 = _quantile(sv, 0.75)
        iqr = q3 - q1
        lo_fence = q1 - iqr_mult * iqr
        hi_fence = q3 + iqr_mult * iqr
        n_outliers = 0
        n_high = 0
        n_low = 0
        for idx, v in vals:
            is_outlier = False
            if method in ("zscore", "both") and std > 1e-12 and abs(v - mean) / std > z_thresh:
                is_outlier = True
            if method in ("iqr", "both") and (v < lo_fence or v > hi_fence):
                is_outlier = True
            if is_outlier:
                n_outliers += 1
                severity = abs(v - mean) / std if std > 1e-12 else 0
                row_scores[idx] = row_scores.get(idx, 0) + severity
                if v > mean:
                    n_high += 1
                else:
                    n_low += 1
        field_summaries.append({
            "field": fld, "outlier_count": n_outliers,
            "high": n_high, "low": n_low, "total_values": len(vals),
        })

    top_flagged = sorted(row_scores.items(), key=lambda x: x[1], reverse=True)[:policy.outlier_max_flagged]
    total_outlier_fields = sum(fs["outlier_count"] for fs in field_summaries)
    total_values = sum(fs["total_values"] for fs in field_summaries)
    quality_score = round(1.0 - total_outlier_fields / max(total_values, 1), 4)

    return {
        "method": method, "z_threshold": z_thresh, "iqr_multiplier": iqr_mult,
        "field_summaries": field_summaries,
        "top_flagged_rows": [{"row_index": idx, "severity": round(s, 4)} for idx, s in top_flagged],
        "data_quality_score": quality_score,
    }


def benford_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Benford's Law first-digit test."""
    if field not in col_types:
        return {"error": f"Field '{field}' not found."}
    vals = [abs(v) for v in (to_float_permissive(r.get(field)) for r in rows) if v is not None and v != 0]
    if len(vals) < policy.benford_min_values:
        return {"error": f"Need at least {policy.benford_min_values} non-zero values, got {len(vals)}."}

    observed = [0] * 10
    for v in vals:
        first_digit = int(str(v).lstrip("0.")[0]) if str(v).lstrip("0.") else 0
        if 1 <= first_digit <= 9:
            observed[first_digit] += 1
    total = sum(observed[1:])
    if total == 0:
        return {"error": "No valid leading digits found."}

    expected_freq = [0.0] + [math.log10(1 + 1.0 / d) for d in range(1, 10)]
    chi2 = 0.0
    digits = []
    for d in range(1, 10):
        obs_pct = observed[d] / total
        exp_pct = expected_freq[d]
        exp_count = exp_pct * total
        if exp_count > 0:
            chi2 += (observed[d] - exp_count) ** 2 / exp_count
        digits.append({
            "digit": d,
            "observed_pct": round(obs_pct, 4),
            "expected_pct": round(exp_pct, 4),
            "deviation": round(obs_pct - exp_pct, 4),
        })

    p_value = _chi2_pvalue(chi2, 8)
    if p_value > 0.10:
        label = "conforms"
    elif p_value > 0.01:
        label = "suspicious"
    else:
        label = "fails"

    return {
        "field": field, "sample_size": total,
        "chi_squared": round(chi2, 4), "p_value": round(p_value, 6),
        "conformity_label": label, "digits": digits,
    }


def _chi2_pvalue(x: float, dof: int) -> float:
    """Approximate chi-squared p-value using the Wilson-Hilferty normal approximation."""
    if x <= 0:
        return 1.0
    z = ((x / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    from ._helpers import normal_cdf
    return max(0.0, min(1.0, 1.0 - normal_cdf(z)))


def vif_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Variance Inflation Factor for numeric columns."""
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:20]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 numeric fields for VIF."}

    mat = extract_numeric_matrix(rows, fields)
    if len(mat) < len(fields) + 2:
        return {"error": f"Not enough complete rows ({len(mat)}) for VIF."}

    results = []
    for i, target_f in enumerate(fields):
        others = [j for j in range(len(fields)) if j != i]
        xs = [[mat[r][j] for j in others] for r in range(len(mat))]
        ys = [mat[r][i] for r in range(len(mat))]
        from ._helpers import ols_r2
        r2 = ols_r2(xs, ys)
        # VIF = 1/(1-R^2); cap at 999 when near-perfect collinearity.
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 - 1e-12 else 999.0
        results.append({
            "field": target_f,
            "vif": round(vif, 2),
            "r2_with_others": round(r2, 4),
            "collinear": vif > policy.vif_high_threshold,
        })
    results.sort(key=lambda x: x["vif"], reverse=True)
    drop = [r["field"] for r in results if r["collinear"]]

    return {
        "fields": fields, "samples": len(mat),
        "vif_results": results, "drop_candidates": drop,
        "threshold": policy.vif_high_threshold,
    }


def effective_dimension_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    method: str = "both",
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Intrinsic dimensionality via participation ratio and MLE."""
    if np is None:
        return {"error": "numpy required for effective_dimension."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:30]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < policy.effective_dim_min_fields:
        return {"error": f"Need at least {policy.effective_dim_min_fields} numeric fields."}

    mat = extract_numeric_matrix(rows, fields)
    if len(mat) < policy.effective_dim_min_rows:
        return {"error": f"Not enough complete rows ({len(mat)})."}

    x = np.array(mat, dtype=float)
    x = x - x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds < 1e-12] = 1.0
    x = x / stds
    cov = np.cov(x, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    total = float(eigenvalues.sum())
    cumvar = np.cumsum(eigenvalues) / total if total > 1e-12 else np.zeros_like(eigenvalues)

    pr = float(total ** 2 / (eigenvalues ** 2).sum()) if (eigenvalues ** 2).sum() > 1e-12 else 1.0

    # Levina-Bickel MLE: k-NN distance estimator for intrinsic dimension.
    mle_dim = None
    if method in ("mle", "both"):
        k = min(policy.effective_dim_default_k, len(mat) - 1)
        if k >= 2:
            from numpy.linalg import norm
            dists = np.array([[norm(x[i] - x[j]) for j in range(len(x))] for i in range(len(x))])
            dims = []
            for i in range(len(x)):
                sorted_d = np.sort(dists[i])[1:k + 1]
                if sorted_d[-1] > 1e-12 and all(d > 1e-12 for d in sorted_d[:-1]):
                    dim_i = (k - 1) / sum(math.log(sorted_d[-1] / d) for d in sorted_d[:-1])
                    dims.append(dim_i)
            mle_dim = round(sum(dims) / len(dims), 2) if dims else None

    result: Dict[str, Any] = {
        "fields": fields, "n_fields": len(fields), "samples": len(mat),
        "participation_ratio": round(pr, 2),
        "eigenvalue_spectrum": [round(float(e), 6) for e in eigenvalues[:min(15, len(eigenvalues))]],
        "cumulative_variance": [round(float(c), 4) for c in cumvar[:min(15, len(cumvar))]],
        "redundancy_ratio": round(len(fields) / pr, 2) if pr > 0.1 else None,
    }
    if mle_dim is not None:
        result["mle_dimension"] = mle_dim
    return result
