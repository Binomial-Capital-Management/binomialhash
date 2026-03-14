"""Temporal dynamics analysis."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

try:
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    pdist = None
    squareform = None

from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    extract_numeric_matrix,
    np,
    to_float_permissive,
)


def _ordered_values(rows, field, order_by):
    """Extract and sort numeric values by an ordering column."""
    paired = []
    for r in rows:
        v = to_float_permissive(r.get(field))
        o = r.get(order_by)
        if v is not None and o is not None:
            paired.append((o, v))
    paired.sort(key=lambda x: x[0])
    return [p[1] for p in paired], [p[0] for p in paired]


def autocorrelation_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    max_lag: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """ACF at multiple lags with significance and periodicity detection."""
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    vals, _ = _ordered_values(rows, field, order_by)
    n = len(vals)
    if n < policy.acf_min_samples:
        return {"error": f"Not enough ordered values ({n})."}

    ml = max_lag or min(policy.acf_max_lag, n // 3)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    if var < 1e-12:
        return {"error": "Constant series — no autocorrelation."}

    acf_vals = []
    # ~2/sqrt(n) approximate 95% CI under white-noise null.
    sig_threshold = 2.0 / math.sqrt(n)
    significant_lags = []
    for lag in range(1, ml + 1):
        cov = sum((vals[i] - mean) * (vals[i - lag] - mean) for i in range(lag, n)) / n
        acf = cov / var
        acf_vals.append({"lag": lag, "acf": round(acf, 4)})
        if abs(acf) > sig_threshold:
            significant_lags.append(lag)

    # Detect dominant period from first peak in ACF
    dominant_period = None
    for i in range(1, len(acf_vals) - 1):
        if (acf_vals[i]["acf"] > acf_vals[i - 1]["acf"] and
                acf_vals[i]["acf"] > acf_vals[i + 1]["acf"] and
                acf_vals[i]["acf"] > sig_threshold):
            dominant_period = acf_vals[i]["lag"]
            break

    # Slow ACF decay suggests non-stationarity; 0.3 is heuristic.
    decay_rate = abs(acf_vals[0]["acf"] - acf_vals[min(4, len(acf_vals) - 1)]["acf"]) if len(acf_vals) > 4 else 1.0
    is_stationary = decay_rate > 0.3

    return {
        "field": field, "samples": n,
        "acf_values": acf_vals, "significant_lags": significant_lags,
        "significance_threshold": round(sig_threshold, 4),
        "dominant_period": dominant_period,
        "is_stationary_hint": is_stationary,
    }


def changepoints_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    min_segment: Optional[int] = None,
    threshold: Optional[float] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """CUSUM-based structural break detection."""
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    vals, order_vals = _ordered_values(rows, field, order_by)
    n = len(vals)
    if n < 20:
        return {"error": f"Not enough ordered values ({n})."}

    ms = min_segment or policy.changepoint_min_segment
    thresh = threshold or policy.changepoint_default_threshold

    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    if std < 1e-12:
        return {"field": field, "samples": n, "changepoints": []}

    # CUSUM
    cusum_pos = [0.0]
    cusum_neg = [0.0]
    for v in vals:
        cusum_pos.append(max(0, cusum_pos[-1] + (v - mean) / std))
        cusum_neg.append(min(0, cusum_neg[-1] + (v - mean) / std))

    changepoints = []
    i = ms
    while i < n - ms:
        segment_before = vals[max(0, i - ms):i]
        segment_after = vals[i:min(n, i + ms)]
        mean_before = sum(segment_before) / len(segment_before)
        mean_after = sum(segment_after) / len(segment_after)
        magnitude = abs(mean_after - mean_before) / std

        if magnitude > thresh:
            confidence = min(1.0, magnitude / (thresh * 2))
            changepoints.append({
                "index": i,
                "order_value": str(order_vals[i]) if i < len(order_vals) else None,
                "mean_before": round(mean_before, 4),
                "mean_after": round(mean_after, 4),
                "magnitude": round(magnitude, 4),
                "confidence": round(confidence, 4),
            })
            i += ms
        else:
            i += 1

    return {
        "field": field, "samples": n,
        "changepoints": changepoints,
        "global_mean": round(mean, 4), "global_std": round(std, 4),
    }


def rolling_analysis_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    window: Optional[int] = None,
    field_b: Optional[str] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Windowed statistics over ordered data."""
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    w = window or policy.rolling_default_window
    vals, order_vals = _ordered_values(rows, field, order_by)
    n = len(vals)
    if n < w + 5:
        return {"error": f"Not enough values ({n}) for window {w}."}

    vals_b = None
    if field_b:
        vals_b_raw, _ = _ordered_values(rows, field_b, order_by)
        if len(vals_b_raw) == n:
            vals_b = vals_b_raw

    windows = []
    for i in range(n - w + 1):
        seg = vals[i:i + w]
        m = sum(seg) / w
        s = (sum((v - m) ** 2 for v in seg) / w) ** 0.5
        entry = {
            "start_index": i,
            "mean": round(m, 4), "std": round(s, 4),
            "min": round(min(seg), 4), "max": round(max(seg), 4),
        }
        if vals_b:
            seg_b = vals_b[i:i + w]
            from ._helpers import pearson_corr
            entry["rolling_corr"] = round(pearson_corr(seg, seg_b), 4)
        windows.append(entry)

    step = max(1, len(windows) // 50)
    sampled = windows[::step]

    means = [w_entry["mean"] for w_entry in windows]
    if len(means) > 2:
        from ._helpers import fit_linear
        xs = list(range(len(means)))
        slope, _, _ = fit_linear([float(x) for x in xs], means)
        trend_slope = round(slope, 6)
    else:
        trend_slope = 0.0

    # Volatility clustering: autocorrelation of squared changes
    stds = [w_entry["std"] for w_entry in windows]
    vol_acf = 0.0
    if len(stds) > 5:
        m_std = sum(stds) / len(stds)
        var_std = sum((s - m_std) ** 2 for s in stds) / len(stds)
        if var_std > 1e-12:
            cov1 = sum((stds[i] - m_std) * (stds[i - 1] - m_std) for i in range(1, len(stds))) / len(stds)
            vol_acf = cov1 / var_std

    result: Dict[str, Any] = {
        "field": field, "samples": n, "window": w,
        "windows_sampled": sampled,
        "trend_slope": trend_slope,
        "volatility_clustering": round(vol_acf, 4),
    }

    if vals_b and field_b:
        corrs = [w_entry.get("rolling_corr", 0) for w_entry in windows if "rolling_corr" in w_entry]
        sign_changes = sum(1 for i in range(1, len(corrs)) if corrs[i] * corrs[i - 1] < 0)
        result["field_b"] = field_b
        result["correlation_sign_changes"] = sign_changes
        result["correlation_stability"] = round(1.0 - sign_changes / max(len(corrs), 1), 4)

    return result


def phase_space_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    max_embedding_dim: Optional[int] = None,
    tau: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Takens' delay embedding attractor reconstruction."""
    if np is None:
        return {"error": "numpy required for phase space analysis."}
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    vals, _ = _ordered_values(rows, field, order_by)
    n = len(vals)
    if n < 50:
        return {"error": f"Not enough values ({n})."}

    max_ed = max_embedding_dim or policy.phase_max_embedding

    # Auto tau: first zero-crossing of ACF
    if tau is None:
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        if var < 1e-12:
            return {"error": "Constant series."}
        tau = 1
        for lag in range(1, n // 3):
            acf = sum((vals[i] - mean) * (vals[i - lag] - mean) for i in range(lag, n)) / (n * var)
            if acf <= 0:
                tau = lag
                break

    # False Nearest Neighbors for embedding dimension
    x_arr = np.array(vals)
    best_dim = 2
    fnn_results = []
    for dim in range(1, max_ed + 1):
        delay_len = n - (dim - 1) * tau
        if delay_len < 20:
            break
        embedded = np.array([[x_arr[i + j * tau] for j in range(dim)] for i in range(delay_len)])
        n_check = min(200, delay_len)
        fnn_count = 0
        for idx in range(n_check):
            dists = np.sqrt(np.sum((embedded - embedded[idx]) ** 2, axis=1))
            dists[idx] = float("inf")
            nn_idx = int(np.argmin(dists))
            nn_dist = dists[nn_idx]
            if nn_dist < 1e-12:
                continue
            if dim < max_ed:
                next_delay_len = n - dim * tau
                if idx < next_delay_len and nn_idx < next_delay_len:
                    extra = abs(x_arr[idx + dim * tau] - x_arr[nn_idx + dim * tau])
                    ratio = extra / nn_dist
                    # FNN: ratio > threshold indicates false neighbor (dim too low).
                    if ratio > policy.phase_fnn_threshold:
                        fnn_count += 1
        fnn_pct = fnn_count / n_check if n_check > 0 else 0
        fnn_results.append({"dim": dim, "fnn_pct": round(fnn_pct, 4)})
        if fnn_pct < 0.05:
            best_dim = dim
            break
        best_dim = dim + 1

    best_dim = min(best_dim, max_ed)

    # Largest Lyapunov exponent (Rosenstein method)
    delay_len = n - (best_dim - 1) * tau
    if delay_len < 30:
        lyap = 0.0
    else:
        embedded = np.array([[x_arr[i + j * tau] for j in range(best_dim)] for i in range(delay_len)])
        divergences = []
        n_sample = min(100, delay_len - 10)
        for idx in range(n_sample):
            dists = np.sqrt(np.sum((embedded - embedded[idx]) ** 2, axis=1))
            dists[idx] = float("inf")
            for jj in range(max(0, idx - tau), min(delay_len, idx + tau + 1)):
                dists[jj] = float("inf")
            nn_idx = int(np.argmin(dists))
            if dists[nn_idx] < 1e-12:
                continue
            max_steps = min(10, delay_len - max(idx, nn_idx) - 1)
            for step in range(1, max_steps + 1):
                if idx + step < delay_len and nn_idx + step < delay_len:
                    d = float(np.sqrt(np.sum((embedded[idx + step] - embedded[nn_idx + step]) ** 2)))
                    if d > 1e-12:
                        divergences.append((step, math.log(d / dists[nn_idx])))

        if divergences:
            from ._helpers import fit_linear
            steps_d = [d[0] for d in divergences]
            log_divs = [d[1] for d in divergences]
            lyap, _, _ = fit_linear([float(s) for s in steps_d], log_divs)
        else:
            lyap = 0.0

    if lyap > 0.1:
        attractor = "chaotic"
    elif lyap < -0.1:
        attractor = "fixed_point"
    else:
        acf1 = 0.0
        mean_v = sum(vals) / n
        var_v = sum((v - mean_v) ** 2 for v in vals) / n
        if var_v > 1e-12:
            acf1 = sum((vals[i] - mean_v) * (vals[i - 1] - mean_v) for i in range(1, n)) / (n * var_v)
        attractor = "limit_cycle" if abs(acf1) > 0.5 else "stochastic"

    prediction_horizon = round(1.0 / lyap, 2) if lyap > 0.01 else None

    return {
        "field": field, "samples": n, "tau": tau,
        "optimal_embedding_dim": best_dim,
        "lyapunov_exponent": round(lyap, 4),
        "attractor_type": attractor,
        "prediction_horizon": prediction_horizon,
        "fnn_results": fnn_results,
    }


def ergodicity_test_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field: str,
    order_by: str,
    window_sizes_json: Optional[str] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Test whether time averages converge to ensemble averages."""
    if field not in col_types or order_by not in col_types:
        return {"error": "Field(s) not found."}

    vals, _ = _ordered_values(rows, field, order_by)
    n = len(vals)
    if n < 30:
        return {"error": f"Not enough values ({n})."}

    try:
        windows = json.loads(window_sizes_json) if window_sizes_json else [10, 25, 50, 100]
    except (json.JSONDecodeError, TypeError):
        windows = [10, 25, 50, 100]
    windows = [w for w in windows if w < n]

    ensemble_mean = sum(vals) / n
    ensemble_var = sum((v - ensemble_mean) ** 2 for v in vals) / n

    time_avgs = []
    for w in windows:
        chunk_means = [sum(vals[i:i + w]) / w for i in range(0, n - w + 1, max(1, w // 2))]
        spread = (sum((m - ensemble_mean) ** 2 for m in chunk_means) / len(chunk_means)) ** 0.5 if chunk_means else 0
        time_avgs.append({
            "window": w,
            "mean_of_means": round(sum(chunk_means) / len(chunk_means), 4) if chunk_means else None,
            "spread_of_means": round(spread, 4),
            "n_chunks": len(chunk_means),
        })

    # Ergodicity: spread of time-averages should scale as 1/sqrt(window).
    if len(time_avgs) >= 2 and time_avgs[0]["spread_of_means"] > 1e-12:
        expected_ratio = math.sqrt(windows[0] / windows[-1])
        actual_ratio = time_avgs[-1]["spread_of_means"] / time_avgs[0]["spread_of_means"] if time_avgs[0]["spread_of_means"] > 1e-12 else 1
        ergo_ratio = actual_ratio / expected_ratio if expected_ratio > 1e-12 else 1
    else:
        ergo_ratio = 1.0

    is_ergodic = 0.5 < ergo_ratio < 2.0

    if is_ergodic:
        implication = "Time averages converge to ensemble averages. Standard statistics are valid."
    else:
        implication = "Non-ergodic: time averages do NOT represent the ensemble. Sample statistics may be misleading."

    return {
        "field": field, "samples": n,
        "ensemble_mean": round(ensemble_mean, 4),
        "ensemble_std": round(ensemble_var ** 0.5, 4),
        "time_averages_by_window": time_avgs,
        "ergodicity_ratio": round(ergo_ratio, 4),
        "is_ergodic": is_ergodic,
        "implication": implication,
    }


def recurrence_analysis_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    order_by: str,
    threshold: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Recurrence Quantification Analysis."""
    if np is None or pdist is None:
        return {"error": "numpy and scipy required for recurrence analysis."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    if order_by not in col_types:
        return {"error": f"Order field '{order_by}' not found."}

    numeric_cols = [c for c, t in col_types.items() if c in fields]
    if not numeric_cols:
        return {"error": "No valid numeric fields."}

    all_fields = numeric_cols + [order_by]
    paired = []
    for r in rows:
        o = r.get(order_by)
        vs = [to_float_permissive(r.get(f)) for f in numeric_cols]
        if o is not None and all(v is not None for v in vs):
            paired.append((o, vs))
    paired.sort(key=lambda x: x[0])
    n = len(paired)
    if n < 30:
        return {"error": f"Not enough rows ({n})."}

    ed = embedding_dim or policy.recurrence_default_embed
    data = np.array([p[1] for p in paired], dtype=float)

    if data.shape[1] == 1 and ed > 1:
        col = data[:, 0]
        delay_n = n - (ed - 1)
        if delay_n < 20:
            return {"error": "Not enough points after embedding."}
        embedded = np.array([[col[i + j] for j in range(ed)] for i in range(delay_n)])
    else:
        embedded = data
        delay_n = n

    if delay_n > 500:
        idx = np.linspace(0, delay_n - 1, 500, dtype=int)
        embedded = embedded[idx]
        delay_n = 500

    dists = squareform(pdist(embedded, metric='euclidean'))

    if threshold is None:
        flat_dists = dists[np.triu_indices(delay_n, k=1)]
        threshold = float(np.percentile(flat_dists, 10))

    rec = (dists <= threshold).astype(int)
    np.fill_diagonal(rec, 0)

    total_possible = delay_n * (delay_n - 1)
    rr = float(rec.sum()) / total_possible if total_possible > 0 else 0

    # Diagonal lines (determinism)
    diag_lengths = []
    for offset in range(1, delay_n):
        diag = np.diag(rec, offset)
        length = 0
        for val in diag:
            if val:
                length += 1
            elif length > 1:
                diag_lengths.append(length)
                length = 0
        if length > 1:
            diag_lengths.append(length)

    total_diag_points = sum(diag_lengths)
    total_rec_points = int(rec.sum()) // 2
    determinism = total_diag_points / total_rec_points if total_rec_points > 0 else 0
    max_diag = max(diag_lengths) if diag_lengths else 0

    # Vertical lines (laminarity)
    vert_lengths = []
    for col in range(delay_n):
        length = 0
        for row in range(delay_n):
            if rec[row, col]:
                length += 1
            elif length > 1:
                vert_lengths.append(length)
                length = 0
        if length > 1:
            vert_lengths.append(length)

    total_vert_points = sum(vert_lengths)
    laminarity = total_vert_points / total_rec_points if total_rec_points > 0 else 0
    trapping = sum(vert_lengths) / len(vert_lengths) if vert_lengths else 0

    from ._helpers import shannon_entropy
    if diag_lengths:
        max_len = max(diag_lengths)
        len_hist = [0] * (max_len + 1)
        for dl in diag_lengths:
            len_hist[dl] += 1
        entropy_diag = shannon_entropy(len_hist)
    else:
        entropy_diag = 0.0

    return {
        "fields": numeric_cols, "samples": delay_n,
        "embedding_dim": ed, "threshold": round(threshold, 4),
        "recurrence_rate": round(rr, 4),
        "determinism": round(determinism, 4),
        "laminarity": round(laminarity, 4),
        "trapping_time": round(trapping, 4),
        "max_diagonal_length": max_diag,
        "entropy_of_diagonals": round(entropy_diag, 4),
    }
