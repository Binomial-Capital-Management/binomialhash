"""Causal inference analysis."""

from __future__ import annotations

import json
import math
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schema import T_NUMERIC
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    bucket_index,
    extract_numeric_matrix,
    normal_cdf,
    np,
    pearson_corr,
    quantile_edges,
    shannon_entropy,
    to_float_permissive,
)


def causal_graph_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    alpha: Optional[float] = None,
    max_conditioning_set: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """PC algorithm: discover causal DAG from conditional independence tests."""
    if np is None:
        return {"error": "numpy required for causal graph."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    p = len(fields)
    if p < 3:
        return {"error": "Need at least 3 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < policy.causal_min_samples:
        return {"error": f"Not enough rows ({n})."}

    alpha_val = alpha or policy.causal_default_alpha
    max_cond = max_conditioning_set or policy.causal_max_conditioning

    x = np.array(mat, dtype=float)
    corr_mat = np.corrcoef(x, rowvar=False)

    def _partial_corr(i: int, j: int, cond: List[int]) -> float:
        if not cond:
            return float(corr_mat[i, j])
        idx = [i, j] + cond
        sub = corr_mat[np.ix_(idx, idx)]
        try:
            prec = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            return 0.0
        d = np.sqrt(abs(prec[0, 0] * prec[1, 1]))
        return -prec[0, 1] / d if d > 1e-12 else 0.0

    # Fisher z-transform for partial correlation p-value under null.
    def _fisher_z_pvalue(r: float, n_pts: int, cond_size: int) -> float:
        dof = n_pts - cond_size - 3
        if dof < 1 or abs(r) >= 1.0:
            return 1.0
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
        return 2.0 * (1.0 - normal_cdf(abs(z)))

    adj: Dict[int, Set[int]] = {i: set(range(p)) - {i} for i in range(p)}
    sep_sets: Dict[Tuple[int, int], List[int]] = {}

    for cond_size in range(max_cond + 1):
        for i in range(p):
            for j in list(adj[i]):
                if j <= i:
                    continue
                neighbors = sorted(adj[i] - {j})
                if len(neighbors) < cond_size:
                    continue
                for cond in combinations(neighbors, cond_size):
                    cond_list = list(cond)
                    pc = _partial_corr(i, j, cond_list)
                    pv = _fisher_z_pvalue(pc, n, len(cond_list))
                    if pv > alpha_val:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        key = (min(i, j), max(i, j))
                        sep_sets[key] = cond_list
                        break

    directed: Set[Tuple[int, int]] = set()
    for i in range(p):
        for j in adj[i]:
            for k in adj[j]:
                if k == i or k in adj[i]:
                    continue
                key = (min(i, k), max(i, k))
                sep = sep_sets.get(key, [])
                if j not in sep:
                    directed.add((i, j))
                    directed.add((k, j))

    # Meek R1: orient v-structures and propagate via rules.
    for i in range(p):
        for j in adj[i]:
            if (i, j) in directed and (j, i) not in directed:
                for k in adj[j]:
                    if k != i and k not in adj[i] and (j, k) not in directed and (k, j) not in directed:
                        directed.add((j, k))

    edges = []
    seen = set()
    for i in range(p):
        for j in adj[i]:
            if j > i:
                pair = (i, j)
                if pair in seen:
                    continue
                seen.add(pair)
                if (i, j) in directed and (j, i) not in directed:
                    direction = f"{fields[i]} -> {fields[j]}"
                elif (j, i) in directed and (i, j) not in directed:
                    direction = f"{fields[j]} -> {fields[i]}"
                else:
                    direction = f"{fields[i]} -- {fields[j]}"
                pc = _partial_corr(i, j, [])
                edges.append({
                    "field_a": fields[i], "field_b": fields[j],
                    "direction": direction,
                    "marginal_corr": round(float(corr_mat[i, j]), 4),
                })

    sep_list = []
    for (i, j), cond in sep_sets.items():
        sep_list.append({
            "field_a": fields[i], "field_b": fields[j],
            "separating_set": [fields[c] for c in cond],
        })

    return {
        "fields": fields, "samples": n, "alpha": alpha_val,
        "edges": edges, "separated_pairs": sep_list[:20],
        "graph_density": round(2 * len(edges) / (p * (p - 1)), 4) if p > 1 else 0,
    }


def transfer_entropy_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    source_field: str,
    target_field: str,
    order_by: str,
    max_lag: Optional[int] = None,
    bins: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Directional information flow via transfer entropy."""
    for f in [source_field, target_field, order_by]:
        if f not in col_types:
            return {"error": f"Field '{f}' not found."}

    paired = []
    for r in rows:
        s = to_float_permissive(r.get(source_field))
        t = to_float_permissive(r.get(target_field))
        o = r.get(order_by)
        if s is not None and t is not None and o is not None:
            paired.append((o, s, t))
    paired.sort(key=lambda x: x[0])
    n = len(paired)
    if n < 30:
        return {"error": f"Not enough ordered values ({n})."}

    ml = max_lag or policy.te_max_lag
    b = bins or policy.te_default_bins
    source_vals = [p[1] for p in paired]
    target_vals = [p[2] for p in paired]

    s_edges = quantile_edges(sorted(source_vals), b)
    t_edges = quantile_edges(sorted(target_vals), b)
    s_bins = [bucket_index(v, s_edges) for v in source_vals]
    t_bins = [bucket_index(v, t_edges) for v in target_vals]

    def _te(src_b, tgt_b, lag):
        """TE(src -> tgt) at given lag."""
        valid_n = n - lag
        if valid_n < 20:
            return 0.0
        # Joint: (tgt_future, tgt_past, src_past)
        joint = {}
        marg_tt = {}
        marg_tts = {}
        marg_t = {}
        for i in range(lag, n):
            tf = tgt_b[i]
            tp = tgt_b[i - lag]
            sp = src_b[i - lag]
            k3 = (tf, tp, sp)
            k2_tt = (tf, tp)
            k2_ts = (tp, sp)
            joint[k3] = joint.get(k3, 0) + 1
            marg_tt[k2_tt] = marg_tt.get(k2_tt, 0) + 1
            marg_tts[k2_ts] = marg_tts.get(k2_ts, 0) + 1
            marg_t[tp] = marg_t.get(tp, 0) + 1

        te = 0.0
        for (tf, tp, sp), count in joint.items():
            p_joint = count / valid_n
            p_tt = marg_tt.get((tf, tp), 0) / valid_n
            p_ts = marg_tts.get((tp, sp), 0) / valid_n
            p_t = marg_t.get(tp, 0) / valid_n
            if p_tt > 0 and p_ts > 0 and p_t > 0:
                ratio = (p_joint * p_t) / (p_tt * p_ts)
                if ratio > 0:
                    te += p_joint * math.log(ratio)
        return max(te, 0.0)

    best_lag = 1
    best_te = 0.0
    lag_results = []
    for lag in range(1, ml + 1):
        te_st = _te(s_bins, t_bins, lag)
        te_ts = _te(t_bins, s_bins, lag)
        lag_results.append({
            "lag": lag,
            "te_source_to_target": round(te_st, 6),
            "te_target_to_source": round(te_ts, 6),
            "net_flow": round(te_st - te_ts, 6),
        })
        if te_st > best_te:
            best_te = te_st
            best_lag = lag

    te_observed = best_te
    null_count = 0
    for _ in range(policy.te_surrogates):
        shuffled_s = s_bins[:]
        random.shuffle(shuffled_s)
        te_null = _te(shuffled_s, t_bins, best_lag)
        if te_null >= te_observed:
            null_count += 1
    p_value = null_count / policy.te_surrogates

    net = lag_results[best_lag - 1]["net_flow"]
    if net > 0.01:
        direction = f"{source_field} -> {target_field}"
    elif net < -0.01:
        direction = f"{target_field} -> {source_field}"
    else:
        direction = "no_clear_direction"

    return {
        "source": source_field, "target": target_field,
        "samples": n, "best_lag": best_lag,
        "te_source_to_target": round(te_observed, 6),
        "p_value": round(p_value, 4),
        "direction_hint": direction,
        "lag_results": lag_results,
    }


def do_estimate_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    treatment: str,
    outcome: str,
    confounders_json: str,
    method: str = "regress",
    bins_count: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Backdoor-criterion causal effect estimation."""
    try:
        confounders = json.loads(confounders_json) if confounders_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid confounders_json."}
    for f in [treatment, outcome] + confounders:
        if f not in col_types:
            return {"error": f"Field '{f}' not found."}

    all_fields = [outcome, treatment] + confounders
    mat = extract_numeric_matrix(rows, all_fields)
    n = len(mat)
    if n < 20:
        return {"error": f"Not enough rows ({n})."}

    ys = [mat[r][0] for r in range(n)]
    xs_treat = [mat[r][1] for r in range(n)]
    naive_corr = pearson_corr(xs_treat, ys)
    sx = sum(xs_treat) / n
    sy = sum(ys) / n
    num = sum((x - sx) * (y - sy) for x, y in zip(xs_treat, ys))
    den = sum((x - sx) ** 2 for x in xs_treat)
    naive_slope = num / den if abs(den) > 1e-12 else 0.0

    if method == "regress":
        from ._helpers import ols_r2
        xs_full = [[mat[r][j] for j in range(1, len(all_fields))] for r in range(n)]
        p = len(all_fields) - 1
        xtx = [[0.0] * (p + 1) for _ in range(p + 1)]
        xty = [0.0] * (p + 1)
        for i in range(n):
            row_ext = [1.0] + xs_full[i]
            for a in range(p + 1):
                xty[a] += row_ext[a] * ys[i]
                for b_idx in range(p + 1):
                    xtx[a][b_idx] += row_ext[a] * row_ext[b_idx]
        aug = [xtx[i][:] + [xty[i]] for i in range(p + 1)]
        for col in range(p + 1):
            max_row = col
            for ri in range(col + 1, p + 1):
                if abs(aug[ri][col]) > abs(aug[max_row][col]):
                    max_row = ri
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-12:
                return {"error": "Singular matrix in backdoor regression."}
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
        ate = beta[1]

        boot_ates = []
        for _ in range(200):
            idx = [random.randint(0, n - 1) for _ in range(n)]
            by = [ys[i] for i in idx]
            bx = [xs_full[i] for i in idx]
            bp = len(bx[0])
            bxtx = [[0.0] * (bp + 1) for _ in range(bp + 1)]
            bxty = [0.0] * (bp + 1)
            for i in range(n):
                re = [1.0] + bx[i]
                for a in range(bp + 1):
                    bxty[a] += re[a] * by[i]
                    for b_idx in range(bp + 1):
                        bxtx[a][b_idx] += re[a] * re[b_idx]
            baug = [bxtx[i][:] + [bxty[i]] for i in range(bp + 1)]
            ok = True
            for col in range(bp + 1):
                mr = col
                for ri in range(col + 1, bp + 1):
                    if abs(baug[ri][col]) > abs(baug[mr][col]):
                        mr = ri
                baug[col], baug[mr] = baug[mr], baug[col]
                if abs(baug[col][col]) < 1e-12:
                    ok = False
                    break
                for ri in range(col + 1, bp + 1):
                    f = baug[ri][col] / baug[col][col]
                    for j in range(col, bp + 2):
                        baug[ri][j] -= f * baug[col][j]
            if not ok:
                continue
            bb = [0.0] * (bp + 1)
            for i in range(bp, -1, -1):
                bb[i] = baug[i][bp + 1]
                for j in range(i + 1, bp + 1):
                    bb[i] -= baug[i][j] * bb[j]
                bb[i] /= baug[i][i]
            boot_ates.append(bb[1])

        boot_ates.sort()
        ci_lo = boot_ates[int(len(boot_ates) * 0.025)] if boot_ates else ate
        ci_hi = boot_ates[int(len(boot_ates) * 0.975)] if boot_ates else ate

        return {
            "treatment": treatment, "outcome": outcome,
            "confounders": confounders, "method": "regression",
            "samples": n,
            "causal_effect_ate": round(ate, 6),
            "confidence_interval": [round(ci_lo, 6), round(ci_hi, 6)],
            "naive_correlation": round(naive_corr, 4),
            "naive_slope": round(naive_slope, 6),
            "confounding_bias": round(naive_slope - ate, 4) if abs(naive_slope) > 1e-12 else 0,
        }
    else:
        return {"error": f"Unknown method '{method}'. Use 'regress'."}


def counterfactual_impact_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    outcome_field: str,
    time_field: str,
    unit_field: str,
    treated_unit: str,
    intervention_time: str,
    donor_units_json: Optional[str] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Synthetic control: estimate counterfactual via donor weighting."""
    for f in [outcome_field, time_field, unit_field]:
        if f not in col_types:
            return {"error": f"Field '{f}' not found."}

    try:
        donor_units = json.loads(donor_units_json) if donor_units_json else None
    except (json.JSONDecodeError, TypeError):
        donor_units = None

    unit_data: Dict[str, Dict[str, float]] = {}
    for r in rows:
        u = str(r.get(unit_field, ""))
        t = str(r.get(time_field, ""))
        v = to_float_permissive(r.get(outcome_field))
        if u and t and v is not None:
            unit_data.setdefault(u, {})[t] = v

    if treated_unit not in unit_data:
        return {"error": f"Treated unit '{treated_unit}' not found."}

    all_units = list(unit_data.keys())
    if donor_units:
        donors = [u for u in donor_units if u in unit_data and u != treated_unit]
    else:
        donors = [u for u in all_units if u != treated_unit]
    if not donors:
        return {"error": "No donor units available."}

    all_times = sorted(set(t for u_data in unit_data.values() for t in u_data.keys()))
    try:
        split_idx = all_times.index(intervention_time)
    except ValueError:
        int_val = to_float_permissive(intervention_time)
        if int_val is not None:
            split_idx = next((i for i, t in enumerate(all_times) if to_float_permissive(t) is not None and to_float_permissive(t) >= int_val), len(all_times))
        else:
            return {"error": f"Intervention time '{intervention_time}' not found in data."}

    pre_times = all_times[:split_idx]
    post_times = all_times[split_idx:]

    if len(pre_times) < policy.synth_min_pre_periods:
        return {"error": f"Not enough pre-intervention periods ({len(pre_times)})."}

    treated_pre = [unit_data[treated_unit].get(t) for t in pre_times]
    donor_pre = [[unit_data[d].get(t) for t in pre_times] for d in donors]

    valid = [i for i in range(len(pre_times))
             if treated_pre[i] is not None and all(donor_pre[d][i] is not None for d in range(len(donors)))]
    if len(valid) < policy.synth_min_pre_periods:
        return {"error": "Not enough overlapping pre-periods."}

    y_pre = [treated_pre[i] for i in valid]
    x_pre = [[donor_pre[d][i] for d in range(len(donors))] for i in valid]

    from ._helpers import ols_r2
    if np is None:
        return {"error": "numpy required for synthetic control."}
    X = np.array(x_pre, dtype=float)
    Y = np.array(y_pre, dtype=float)
    # Synthetic control: non-negative donor weights summing to 1.
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    beta = np.clip(beta, 0, None)
    beta_sum = beta.sum()
    if beta_sum > 1e-12:
        weights = beta / beta_sum
    else:
        weights = np.ones(len(donors)) / len(donors)

    pred_pre = X @ weights
    rmse_pre = float(np.sqrt(np.mean((Y - pred_pre) ** 2)))

    impacts = []
    for t in post_times:
        actual = unit_data[treated_unit].get(t)
        donor_vals = [unit_data[d].get(t) for d in donors]
        if actual is not None and all(v is not None for v in donor_vals):
            synthetic = sum(w * v for w, v in zip(weights, donor_vals))
            impacts.append({
                "time": t,
                "actual": round(actual, 4),
                "synthetic": round(float(synthetic), 4),
                "impact": round(actual - float(synthetic), 4),
            })

    cumulative = sum(imp["impact"] for imp in impacts)
    donor_weights = [{"unit": d, "weight": round(float(w), 4)} for d, w in zip(donors, weights)]
    donor_weights.sort(key=lambda x: abs(x["weight"]), reverse=True)

    return {
        "treated_unit": treated_unit, "outcome": outcome_field,
        "pre_periods": len(pre_times), "post_periods": len(post_times),
        "pre_fit_rmse": round(rmse_pre, 4),
        "post_impacts": impacts[:50],
        "cumulative_impact": round(cumulative, 4),
        "donor_weights": donor_weights[:10],
    }
