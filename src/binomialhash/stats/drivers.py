"""Driver discovery and feature selection."""

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
    extract_numeric_matrix,
    extract_numeric_pairs,
    np,
    ols_r2,
    pearson_corr,
    quantile_edges,
    shannon_entropy,
    to_float_permissive,
)


def polynomial_test_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field_x: str,
    field_y: str,
    max_degree: int = 3,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Fit polynomial regressions of degree 1-3 and compare R-squared."""
    xs, ys = extract_numeric_pairs(rows, field_x, field_y)
    n = len(xs)
    if n < policy.polynomial_min_samples:
        return {"error": f"Not enough paired values ({n})."}
    max_degree = min(max_degree, 3)

    results = {}
    for deg in range(1, max_degree + 1):
        if n < deg + 2:
            break
        xs_poly = [[x ** p for p in range(1, deg + 1)] for x in xs]
        r2 = ols_r2(xs_poly, ys)
        results[f"degree_{deg}_r2"] = round(r2, 4)

    linear = results.get("degree_1_r2", 0)
    best_deg = 1
    best_r2 = linear
    for deg in range(2, max_degree + 1):
        r2 = results.get(f"degree_{deg}_r2", 0)
        # Require R^2 gain > 0.02 to avoid overfitting to noise.
        if r2 > best_r2 + 0.02:
            best_deg = deg
            best_r2 = r2

    return {
        "field_x": field_x, "field_y": field_y, "samples": n,
        **results,
        "best_degree": best_deg,
        "curvature_gain": round(best_r2 - linear, 4),
    }


def interaction_screen_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    target: str,
    candidates_json: str,
    top_k: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Test pairs of candidates for non-additive effects on target."""
    try:
        candidates = json.loads(candidates_json) if candidates_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid candidates_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if target not in numeric_cols:
        return {"error": f"Target '{target}' must be numeric."}
    candidates = [c for c in candidates if c in numeric_cols and c != target]
    if len(candidates) < 2:
        return {"error": "Need at least 2 candidate fields."}

    top_k = top_k or policy.interaction_default_top_k
    all_fields = [target] + candidates
    mat = extract_numeric_matrix(rows, all_fields)
    n = len(mat)
    if n < policy.interaction_min_samples:
        return {"error": f"Not enough rows ({n})."}

    ys = [mat[r][0] for r in range(n)]
    interactions = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ci, cj = i + 1, j + 1
            xs_a = [[mat[r][ci]] for r in range(n)]
            xs_b = [[mat[r][cj]] for r in range(n)]
            xs_additive = [[mat[r][ci], mat[r][cj]] for r in range(n)]
            xs_ab = [[mat[r][ci], mat[r][cj], mat[r][ci] * mat[r][cj]] for r in range(n)]
            r2_a = ols_r2(xs_a, ys)
            r2_b = ols_r2(xs_b, ys)
            r2_additive = ols_r2(xs_additive, ys)
            r2_joint = ols_r2(xs_ab, ys)
            strength = r2_joint - r2_additive
            # Synergy/suppression thresholds tuned to avoid flagging noise.
            if r2_joint > r2_a and r2_joint > r2_b:
                itype = "synergy" if strength > 0.02 else "additive"
            else:
                itype = "suppression" if strength < -0.02 else "additive"
            interactions.append({
                "field_a": candidates[i], "field_b": candidates[j],
                "r2_a": round(r2_a, 4), "r2_b": round(r2_b, 4),
                "r2_joint": round(r2_joint, 4),
                "interaction_strength": round(strength, 4),
                "interaction_type": itype,
            })

    interactions.sort(key=lambda x: abs(x["interaction_strength"]), reverse=True)
    return {
        "target": target, "samples": n,
        "interactions": interactions[:top_k],
        "total_pairs_tested": len(interactions),
    }


def sparse_drivers_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    target: str,
    candidates_json: str,
    alpha: Optional[float] = None,
    max_features: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """LASSO (L1) variable selection via coordinate descent."""
    if np is None:
        return {"error": "numpy required for LASSO."}
    try:
        candidates = json.loads(candidates_json) if candidates_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid candidates_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if target not in numeric_cols:
        return {"error": f"Target '{target}' must be numeric."}
    candidates = [c for c in candidates if c in numeric_cols and c != target]
    if not candidates:
        return {"error": "No valid candidate fields."}

    max_feat = max_features or policy.sparse_default_max_features
    all_fields = [target] + candidates
    mat = extract_numeric_matrix(rows, all_fields)
    n = len(mat)
    if n < 10:
        return {"error": f"Not enough rows ({n})."}

    x = np.array([[mat[r][j] for j in range(1, len(all_fields))] for r in range(n)])
    y = np.array([mat[r][0] for r in range(n)])
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    xn = (x - x_mean) / x_std
    y_mean = y.mean()
    yn = y - y_mean

    def _lasso_cd(xn, yn, alpha_val, max_iter):
        p = xn.shape[1]
        beta = np.zeros(p)
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r = yn - xn @ beta + xn[:, j] * beta[j]
                rho = float(xn[:, j] @ r) / n
                beta[j] = math.copysign(max(abs(rho) - alpha_val, 0), rho)
            if np.max(np.abs(beta - beta_old)) < 1e-6:
                break
        return beta

    if alpha is None:
        alpha_max = float(np.max(np.abs(xn.T @ yn))) / n
        # Log-spaced alpha path from max (all zeros) down to 1% of max.
        alphas = np.logspace(math.log10(alpha_max), math.log10(alpha_max * 0.01), policy.sparse_n_alphas)
        best_alpha = alphas[0]
        best_score = -1e9
        fold_size = max(n // policy.sparse_cv_folds, 5)
        for a in alphas:
            scores = []
            for fold in range(min(policy.sparse_cv_folds, n // fold_size)):
                start = fold * fold_size
                end_idx = start + fold_size
                x_train = np.vstack([xn[:start], xn[end_idx:]])
                y_train = np.concatenate([yn[:start], yn[end_idx:]])
                x_test = xn[start:end_idx]
                y_test = yn[start:end_idx]
                beta = _lasso_cd(x_train, y_train, float(a), policy.sparse_max_iter)
                pred = x_test @ beta
                ss_res = float(np.sum((y_test - pred) ** 2))
                ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                scores.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0)
            avg = sum(scores) / len(scores) if scores else 0
            if avg > best_score:
                best_score = avg
                best_alpha = float(a)
        alpha = best_alpha

    beta_final = _lasso_cd(xn, yn, alpha, policy.sparse_max_iter)
    pred = xn @ beta_final
    ss_res = float(np.sum((yn - pred) ** 2))
    ss_tot = float(np.sum(yn ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

    coefs_original = beta_final / x_std
    selected = []
    eliminated = []
    for j, c in enumerate(candidates):
        if abs(beta_final[j]) > 1e-8:
            selected.append({"field": c, "coefficient": round(float(coefs_original[j]), 6)})
        else:
            eliminated.append(c)
    selected.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return {
        "target": target, "samples": n,
        "alpha": round(alpha, 6), "r2": round(r2, 4),
        "selected_features": selected[:max_feat],
        "eliminated_features": eliminated,
        "sparsity_ratio": round(len(selected) / len(candidates), 4) if candidates else 0,
    }


def feature_importance_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    target: str,
    candidates_json: str,
    n_shuffles: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Permutation-based model-agnostic feature importance."""
    try:
        candidates = json.loads(candidates_json) if candidates_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid candidates_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if target not in numeric_cols:
        return {"error": f"Target '{target}' must be numeric."}
    candidates = [c for c in candidates if c in numeric_cols and c != target]
    if not candidates:
        return {"error": "No valid candidate fields."}

    n_shuf = n_shuffles or policy.importance_n_shuffles
    all_fields = [target] + candidates
    mat = extract_numeric_matrix(rows, all_fields)
    n = len(mat)
    if n < 15:
        return {"error": f"Not enough rows ({n})."}

    ys = [mat[r][0] for r in range(n)]
    xs_all = [[mat[r][j] for j in range(1, len(all_fields))] for r in range(n)]
    baseline_r2 = ols_r2(xs_all, ys)

    results = []
    for j, cand in enumerate(candidates):
        drop_sum = 0.0
        for _ in range(n_shuf):
            shuffled = [row[:] for row in xs_all]
            col_vals = [shuffled[r][j] for r in range(n)]
            random.shuffle(col_vals)
            for r in range(n):
                shuffled[r][j] = col_vals[r]
            drop_sum += baseline_r2 - ols_r2(shuffled, ys)
        importance = drop_sum / n_shuf
        results.append({
            "field": cand,
            "importance_score": round(importance, 6),
            "baseline_r2": round(baseline_r2, 4),
        })
    results.sort(key=lambda x: x["importance_score"], reverse=True)

    return {
        "target": target, "samples": n,
        "baseline_r2": round(baseline_r2, 4),
        "importances": results,
    }


def information_bottleneck_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    input_fields_json: str,
    target_field: str,
    beta: Optional[float] = None,
    n_clusters: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Information Bottleneck: optimal compression preserving target info."""
    try:
        fields = json.loads(input_fields_json) if input_fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid input_fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if target_field not in numeric_cols:
        return {"error": f"Target '{target_field}' must be numeric."}
    fields = [f for f in fields if f in numeric_cols and f != target_field]
    if not fields:
        return {"error": "No valid input fields."}

    beta = beta or policy.ib_default_beta
    nc = n_clusters or policy.ib_default_clusters
    bins = 8
    all_fields = fields + [target_field]
    mat = extract_numeric_matrix(rows, all_fields)
    n = len(mat)
    if n < 20:
        return {"error": f"Not enough rows ({n})."}

    n_input = len(fields)
    input_bins = []
    for j in range(n_input):
        col_vals = sorted(mat[r][j] for r in range(n))
        edges = quantile_edges(col_vals, bins)
        input_bins.append([bucket_index(mat[r][j], edges) for r in range(n)])
    target_vals = sorted(mat[r][n_input] for r in range(n))
    target_edges = quantile_edges(target_vals, bins)
    target_bins = [bucket_index(mat[r][n_input], target_edges) for r in range(n)]

    # Composite input state: bin-tuple -> int
    state_map: Dict[tuple, int] = {}
    states: List[int] = []
    for r in range(n):
        key = tuple(input_bins[j][r] for j in range(n_input))
        if key not in state_map:
            state_map[key] = len(state_map)
        states.append(state_map[key])
    n_states = len(state_map)

    # p(x), p(y|x)
    px = [0.0] * n_states
    pyx = [[0.0] * bins for _ in range(n_states)]
    for r in range(n):
        s = states[r]
        t = target_bins[r]
        px[s] += 1.0
        pyx[s][t] += 1.0
    for s in range(n_states):
        if px[s] > 0:
            for t in range(bins):
                pyx[s][t] /= px[s]
    total_px = sum(px)
    px = [p / total_px for p in px]

    # IB clustering: assign each input state to one of nc clusters
    assignments = [i % nc for i in range(n_states)]
    for _ in range(policy.ib_max_iter):
        pc = [0.0] * nc
        pyc = [[0.0] * bins for _ in range(nc)]
        for s in range(n_states):
            c = assignments[s]
            pc[c] += px[s]
            for t in range(bins):
                pyc[c][t] += px[s] * pyx[s][t]
        for c in range(nc):
            if pc[c] > 0:
                for t in range(bins):
                    pyc[c][t] /= pc[c]

        changed = False
        for s in range(n_states):
            if px[s] < 1e-12:
                continue
            best_c = assignments[s]
            best_cost = float("inf")
            for c in range(nc):
                kl = sum(
                    pyx[s][t] * (math.log(pyx[s][t] / pyc[c][t]) if pyx[s][t] > 1e-12 and pyc[c][t] > 1e-12 else 0)
                    for t in range(bins)
                )
                # IB cost: -I(T;Y) + (1/beta)*I(T;X); higher beta favors compression.
                cost = kl - (1.0 / beta) * math.log(max(pc[c], 1e-12))
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
            if best_c != assignments[s]:
                assignments[s] = best_c
                changed = True
        if not changed:
            break

    # MI(T;Y) and MI(T;X)
    row_clusters = [assignments[states[r]] for r in range(n)]
    pc_final = [0] * nc
    pyc_final = [[0] * bins for _ in range(nc)]
    for r in range(n):
        c = row_clusters[r]
        pc_final[c] += 1
        pyc_final[c][target_bins[r]] += 1
    h_t = shannon_entropy(pc_final)
    pt = [sum(1 for r in range(n) if target_bins[r] == t) for t in range(bins)]
    h_y = shannon_entropy(pt)
    h_ty = shannon_entropy([pyc_final[c][t] for c in range(nc) for t in range(bins)])
    mi_ty = max(0.0, h_t + h_y - h_ty)

    h_xy = shannon_entropy([sum(1 for r in range(n) if states[r] == s and target_bins[r] == t) for s in range(n_states) for t in range(bins)])
    h_x = shannon_entropy([sum(1 for r in range(n) if states[r] == s) for s in range(n_states)])
    mi_xy = max(0.0, h_x + h_y - h_xy)
    preserved = mi_ty / mi_xy if mi_xy > 1e-12 else 0.0

    profiles = []
    for c in range(nc):
        members = [r for r in range(n) if row_clusters[r] == c]
        if not members:
            continue
        profile = {"cluster": c, "size": len(members)}
        for j, f in enumerate(fields):
            fvals = [mat[r][j] for r in members]
            profile[f"{f}_mean"] = round(sum(fvals) / len(fvals), 4)
        profiles.append(profile)

    return {
        "input_fields": fields, "target": target_field,
        "samples": n, "n_clusters": nc, "beta": beta,
        "preserved_info": round(preserved, 4),
        "compression_ratio": round(n_states / max(nc, 1), 2),
        "mi_compressed_target": round(mi_ty, 6),
        "mi_full_target": round(mi_xy, 6),
        "cluster_profiles": profiles,
    }
