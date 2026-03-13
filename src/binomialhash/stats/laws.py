"""Scale, symmetry, and universal laws."""

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
    shannon_entropy,
    to_float_permissive,
)


def entropy_spectrum_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    max_scale: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Multi-scale sample entropy (complexity profile)."""
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    paired = []
    for r in rows:
        v = to_float_permissive(r.get(field))
        o = r.get(order_by)
        if v is not None and o is not None:
            paired.append((o, v))
    paired.sort(key=lambda x: x[0])
    vals = [p[1] for p in paired]
    n = len(vals)
    if n < 50:
        return {"error": f"Not enough ordered values ({n})."}

    ms = max_scale or policy.entropy_max_scale
    m = embedding_dim or policy.entropy_default_embed
    r_tol = 0.15  # tolerance as fraction of std

    def _sample_entropy(series, m_val, r_val):
        n_s = len(series)
        if n_s < m_val + 2:
            return 0.0

        def _templates(dim):
            return [series[i:i + dim] for i in range(n_s - dim)]

        def _count_matches(templates, r_thresh):
            count = 0
            nt = len(templates)
            for i in range(nt):
                for j in range(i + 1, nt):
                    if all(abs(templates[i][k] - templates[j][k]) <= r_thresh for k in range(len(templates[i]))):
                        count += 1
            return count

        templates_m = _templates(m_val)
        templates_m1 = _templates(m_val + 1)
        b = _count_matches(templates_m, r_val)
        a = _count_matches(templates_m1, r_val)
        if b == 0:
            return 0.0
        return -math.log(a / b) if a > 0 else 0.0

    std = (sum((v - sum(vals) / n) ** 2 for v in vals) / n) ** 0.5
    r_threshold = r_tol * std if std > 1e-12 else 0.15

    spectrum = []
    for scale in range(1, ms + 1):
        coarsened_n = n // scale
        if coarsened_n < m + 5:
            break
        coarsened = [sum(vals[i * scale:(i + 1) * scale]) / scale for i in range(coarsened_n)]
        se = _sample_entropy(coarsened, m, r_threshold)
        spectrum.append({"scale": scale, "sample_entropy": round(se, 4), "series_length": coarsened_n})

    if len(spectrum) >= 3:
        entropies = [s["sample_entropy"] for s in spectrum]
        if entropies[0] > entropies[-1] * 1.5:
            complexity_type = "decreasing"
        elif entropies[-1] > entropies[0] * 1.5:
            complexity_type = "increasing"
        elif max(entropies) / (min(entropies) + 1e-12) < 1.3:
            complexity_type = "scale_invariant"
        else:
            complexity_type = "mixed"
    else:
        complexity_type = "insufficient_data"

    return {
        "field": field, "samples": n,
        "embedding_dim": m, "tolerance": round(r_threshold, 4),
        "entropy_spectrum": spectrum,
        "complexity_type": complexity_type,
    }


def renormalization_flow_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    max_scale: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Multiscale coarse-graining: how correlations change under averaging."""
    if np is None:
        return {"error": "numpy required for renormalization flow."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:10]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < 20:
        return {"error": f"Not enough rows ({n})."}

    ms = max_scale or policy.renorm_max_scale
    x = np.array(mat, dtype=float)

    scales = []
    for scale in range(1, ms + 1):
        block_n = n // (2 ** (scale - 1))
        if block_n < 10:
            break
        block_size = 2 ** (scale - 1)
        coarsened = np.array([x[i * block_size:(i + 1) * block_size].mean(axis=0) for i in range(block_n)])
        corr = np.corrcoef(coarsened, rowvar=False)
        upper = corr[np.triu_indices(len(fields), k=1)]
        mean_corr = float(np.mean(np.abs(upper)))
        max_corr = float(np.max(np.abs(upper)))
        stds = coarsened.std(axis=0)
        mean_std = float(np.mean(stds))

        scales.append({
            "scale": scale, "block_size": block_size, "n_blocks": block_n,
            "mean_abs_corr": round(mean_corr, 4),
            "max_abs_corr": round(max_corr, 4),
            "mean_std": round(mean_std, 4),
        })

    if len(scales) >= 2:
        corr_trend = scales[-1]["mean_abs_corr"] - scales[0]["mean_abs_corr"]
        if corr_trend > 0.1:
            flow_type = "coupling_strengthens"
        elif corr_trend < -0.1:
            flow_type = "coupling_weakens"
        else:
            flow_type = "scale_invariant"
    else:
        flow_type = "insufficient_data"
        corr_trend = 0.0

    fixed_point_corr = scales[-1]["mean_abs_corr"] if scales else 0

    return {
        "fields": fields, "samples": n,
        "scales": scales,
        "flow_type": flow_type,
        "correlation_trend": round(corr_trend, 4),
        "fixed_point_correlation": round(fixed_point_corr, 4),
    }


def symmetry_scan_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Detect invariances: translation, scaling, reflection, permutation symmetries."""
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < 20:
        return {"error": f"Not enough rows ({n})."}

    symmetries = []

    # Translation invariance: are differences more structured than raw?
    for i in range(len(fields)):
        vals = [mat[r][i] for r in range(n)]
        diffs = [vals[j] - vals[j - 1] for j in range(1, n)]
        mean_raw = sum(vals) / n
        std_raw = (sum((v - mean_raw) ** 2 for v in vals) / n) ** 0.5
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)) ** 0.5
        cv_raw = std_raw / abs(mean_raw) if abs(mean_raw) > 1e-12 else float("inf")
        cv_diff = std_diff / abs(mean_diff) if abs(mean_diff) > 1e-12 else float("inf")
        if cv_diff < cv_raw * 0.5:
            symmetries.append({
                "type": "translation_invariance", "field": fields[i],
                "cv_raw": round(cv_raw, 4), "cv_diff": round(cv_diff, 4),
            })

    # Scale invariance: log-transformed data more normal?
    for i in range(len(fields)):
        vals = [mat[r][i] for r in range(n)]
        positive = [v for v in vals if v > 0]
        if len(positive) < n * 0.8:
            continue
        log_vals = [math.log(v) for v in positive]
        log_mean = sum(log_vals) / len(log_vals)
        log_std = (sum((v - log_mean) ** 2 for v in log_vals) / len(log_vals)) ** 0.5
        raw_mean = sum(positive) / len(positive)
        raw_std = (sum((v - raw_mean) ** 2 for v in positive) / len(positive)) ** 0.5
        if log_std > 1e-12 and raw_std > 1e-12:
            log_skew = sum(((v - log_mean) / log_std) ** 3 for v in log_vals) / len(log_vals)
            raw_skew = sum(((v - raw_mean) / raw_std) ** 3 for v in positive) / len(positive)
            if abs(log_skew) < abs(raw_skew) * 0.5:
                symmetries.append({
                    "type": "scale_invariance", "field": fields[i],
                    "raw_skew": round(raw_skew, 4), "log_skew": round(log_skew, 4),
                })

    # Reflection symmetry: distributions symmetric around mean?
    for i in range(len(fields)):
        vals = [mat[r][i] for r in range(n)]
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        if std < 1e-12:
            continue
        skew = sum(((v - mean) / std) ** 3 for v in vals) / n
        if abs(skew) < 0.2:
            symmetries.append({
                "type": "reflection_symmetry", "field": fields[i],
                "skewness": round(skew, 4),
            })

    # Permutation symmetry: do pairs of columns have similar distributions?
    exchangeable_pairs = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            vals_i = sorted(mat[r][i] for r in range(n))
            vals_j = sorted(mat[r][j] for r in range(n))
            ks = max(abs(vals_i[k] - vals_j[k]) for k in range(n)) / max(
                max(abs(v) for v in vals_i + vals_j), 1e-12
            )
            if ks < 0.1:
                exchangeable_pairs.append({
                    "type": "permutation_symmetry",
                    "field_a": fields[i], "field_b": fields[j],
                    "ks_distance": round(ks, 4),
                })

    symmetries.extend(exchangeable_pairs)

    return {
        "fields": fields, "samples": n,
        "symmetries": symmetries,
        "total_symmetries_found": len(symmetries),
    }
