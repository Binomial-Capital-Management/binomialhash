"""Regression, partial correlation, PCA, dependency screen, and solver."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..schema import T_NUMERIC
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    np,
    pearson_corr,
    to_float_permissive,
)


def regress_dataset(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    target: str,
    drivers_json: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Multivariate OLS over rows using a target and driver list."""
    try:
        drivers = json.loads(drivers_json) if drivers_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid drivers_json."}
    if not isinstance(drivers, list) or not drivers:
        return {"error": "drivers_json must be a non-empty list of column names."}

    all_cols = [target] + drivers
    for col in all_cols:
        if col not in col_types:
            return {
                "error": f"Column '{col}' not found. Available: {columns[:policy.error_preview_column_limit]}"
            }

    xs_all: List[List[float]] = []
    ys: List[float] = []
    for row in rows:
        yv = to_float_permissive(row.get(target))
        if yv is None:
            continue
        xrow = []
        skip = False
        for driver in drivers:
            xv = to_float_permissive(row.get(driver))
            if xv is None:
                skip = True
                break
            xrow.append(xv)
        if skip:
            continue
        xs_all.append(xrow)
        ys.append(yv)

    n = len(ys)
    p = len(drivers)
    if n < p + policy.regression_min_extra_samples:
        return {"error": f"Not enough complete rows ({n}) for {p} drivers."}

    # Build normal equations X'X β = X'y manually to avoid numpy dependency.
    xtx = [[0.0] * (p + 1) for _ in range(p + 1)]
    xty = [0.0] * (p + 1)
    for i in range(n):
        row_ext = [1.0] + xs_all[i]
        for a in range(p + 1):
            xty[a] += row_ext[a] * ys[i]
            for b in range(p + 1):
                xtx[a][b] += row_ext[a] * row_ext[b]

    # Partial pivoting for numerical stability; singular matrix implies collinear drivers.
    aug = [xtx[i][:] + [xty[i]] for i in range(p + 1)]
    for col in range(p + 1):
        max_row = col
        for row_idx in range(col + 1, p + 1):
            if abs(aug[row_idx][col]) > abs(aug[max_row][col]):
                max_row = row_idx
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return {"error": "Singular matrix — drivers may be collinear."}
        for row_idx in range(col + 1, p + 1):
            factor = aug[row_idx][col] / aug[col][col]
            for j in range(col, p + 2):
                aug[row_idx][j] -= factor * aug[col][j]

    beta = [0.0] * (p + 1)
    for i in range(p, -1, -1):
        beta[i] = aug[i][p + 1]
        for j in range(i + 1, p + 1):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]

    intercept = beta[0]
    coefficients = beta[1:]
    mean_y = sum(ys) / n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        predicted = intercept + sum(coefficients[j] * xs_all[i][j] for j in range(p))
        ss_res += (ys[i] - predicted) ** 2
        ss_tot += (ys[i] - mean_y) ** 2
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    adj_r2 = 1.0 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

    driver_results = []
    for j, driver in enumerate(drivers):
        single_corr = pearson_corr([xs_all[i][j] for i in range(n)], ys)
        driver_results.append(
            {
                "driver": driver,
                "coefficient": round(coefficients[j], 6),
                "individual_correlation": round(single_corr, 4),
            }
        )
    driver_results.sort(key=lambda item: abs(item["coefficient"]), reverse=True)

    return {
        "target": target,
        "samples": n,
        "intercept": round(intercept, 6),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
        "drivers": driver_results,
    }


def partial_correlate_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    field_a: str,
    field_b: str,
    controls_json: str,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Partial correlation of A and B after removing control effects."""
    try:
        controls = json.loads(controls_json) if controls_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid controls_json."}
    if not isinstance(controls, list):
        return {"error": "controls_json must be a list of column names."}

    for col in [field_a, field_b] + controls:
        if col not in col_types:
            return {"error": f"Column '{col}' not found."}

    a_vals: List[float] = []
    b_vals: List[float] = []
    ctrl_vals: List[List[float]] = []
    for row in rows:
        av = to_float_permissive(row.get(field_a))
        bv = to_float_permissive(row.get(field_b))
        if av is None or bv is None:
            continue
        cvs: List[float] = []
        skip = False
        for control in controls:
            cv = to_float_permissive(row.get(control))
            if cv is None:
                skip = True
                break
            cvs.append(cv)
        if skip:
            continue
        a_vals.append(av)
        b_vals.append(bv)
        ctrl_vals.append(cvs)

    n = len(a_vals)
    if n < len(controls) + policy.partial_corr_min_extra_samples:
        return {"error": f"Not enough rows ({n}) for partial correlation."}

    if not controls:
        raw = pearson_corr(a_vals, b_vals)
        return {
            "field_a": field_a,
            "field_b": field_b,
            "controls": [],
            "partial_correlation": round(raw, 4),
            "raw_correlation": round(raw, 4),
            "samples": n,
        }

    # Frisch-Waugh-Lovell: OLS-regress each variable on controls, then correlate residuals.
    def residuals(vals: List[float], ctrl: List[List[float]]) -> List[float]:
        p = len(controls)
        xtx = [[0.0] * (p + 1) for _ in range(p + 1)]
        xty = [0.0] * (p + 1)
        for i in range(n):
            row_ext = [1.0] + ctrl[i]
            for a in range(p + 1):
                xty[a] += row_ext[a] * vals[i]
                for b in range(p + 1):
                    xtx[a][b] += row_ext[a] * row_ext[b]
        aug = [xtx[r][:] + [xty[r]] for r in range(p + 1)]
        for col in range(p + 1):
            max_row = col
            for row_idx in range(col + 1, p + 1):
                if abs(aug[row_idx][col]) > abs(aug[max_row][col]):
                    max_row = row_idx
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-12:
                return vals
            for row_idx in range(col + 1, p + 1):
                factor = aug[row_idx][col] / aug[col][col]
                for j in range(col, p + 2):
                    aug[row_idx][j] -= factor * aug[col][j]
        beta = [0.0] * (p + 1)
        for i in range(p, -1, -1):
            beta[i] = aug[i][p + 1]
            for j in range(i + 1, p + 1):
                beta[i] -= aug[i][j] * beta[j]
            beta[i] /= aug[i][i]
        return [
            vals[i] - (beta[0] + sum(beta[j + 1] * ctrl[i][j] for j in range(p)))
            for i in range(n)
        ]

    resid_a = residuals(a_vals, ctrl_vals)
    resid_b = residuals(b_vals, ctrl_vals)
    partial = pearson_corr(resid_a, resid_b)
    raw = pearson_corr(a_vals, b_vals)
    return {
        "field_a": field_a,
        "field_b": field_b,
        "controls": controls,
        "partial_correlation": round(partial, 4),
        "raw_correlation": round(raw, 4),
        "samples": n,
        "change": round(partial - raw, 4),
    }


def pca_surface_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    n_components: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """PCA on selected numeric fields."""
    if np is None:
        return {"error": "numpy not available for PCA."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[: policy.pca_default_field_count]
    if not isinstance(fields, list) or len(fields) < policy.pca_min_field_count:
        return {
            "error": f"fields_json must resolve to a list of at least {policy.pca_min_field_count} numeric columns."
        }
    fields = [field for field in fields if field in numeric_cols]
    if len(fields) < policy.pca_min_field_count:
        return {"error": f"Need at least {policy.pca_min_field_count} valid numeric fields for PCA."}

    complete_rows: List[List[float]] = []
    for row in rows:
        vals: List[float] = []
        skip = False
        for field in fields:
            value = to_float_permissive(row.get(field))
            if value is None:
                skip = True
                break
            vals.append(value)
        if not skip:
            complete_rows.append(vals)
    if len(complete_rows) < policy.pca_min_complete_rows:
        return {"error": f"Not enough complete rows ({len(complete_rows)}) for PCA."}

    x = np.array(complete_rows, dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds < 1e-12] = 1.0
    xz = (x - means) / stds
    cov = np.cov(xz, rowvar=False)
    vals_e, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals_e)[::-1]
    vals_e = vals_e[order]
    vecs = vecs[:, order]
    requested_components = (
        policy.pca_default_component_count if n_components is None else int(n_components)
    )
    n_comp = max(1, min(requested_components, len(fields)))
    total_var = float(vals_e.sum()) if float(vals_e.sum()) > 1e-12 else 1.0

    components = []
    for i in range(n_comp):
        loadings = [
            {"field": fields[j], "loading": round(float(vecs[j, i]), 6)}
            for j in range(len(fields))
        ]
        loadings.sort(key=lambda item: abs(item["loading"]), reverse=True)
        components.append(
            {
                "component": i + 1,
                "eigenvalue": round(float(vals_e[i]), 6),
                "explained_variance_ratio": round(float(vals_e[i] / total_var), 6),
                "top_loadings": loadings[: policy.pca_top_loading_count],
            }
        )

    scores = xz @ vecs[:, :n_comp]
    score_summary = []
    for i in range(n_comp):
        col = scores[:, i]
        score_summary.append(
            {
                "component": i + 1,
                "mean": round(float(col.mean()), 6),
                "std": round(float(col.std()), 6),
                "min": round(float(col.min()), 6),
                "max": round(float(col.max()), 6),
            }
        )

    return {
        "fields": fields,
        "samples": int(x.shape[0]),
        "components": components,
        "score_summary": score_summary,
        "method_notes": [
            "PCA uses standardized numeric fields.",
            "Explained variance ratios sum over the selected fields only.",
        ],
    }


def dependency_screen_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    target: str,
    candidates_json: str,
    controls_json: str = "[]",
    top_k: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Rank candidate drivers using raw corr, partial corr, and coefficient strength."""
    if np is None:
        return {"error": "numpy not available for dependency screening."}
    try:
        candidates = json.loads(candidates_json) if candidates_json else []
        controls = json.loads(controls_json) if controls_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid candidates_json or controls_json."}
    if not isinstance(candidates, list) or not candidates:
        return {"error": "candidates_json must be a non-empty list."}
    if not isinstance(controls, list):
        return {"error": "controls_json must be a list."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if target not in numeric_cols:
        return {"error": f"Target '{target}' must be numeric."}
    candidates = [c for c in candidates if c in numeric_cols and c != target]
    controls = [c for c in controls if c in numeric_cols and c != target]
    if not candidates:
        return {"error": "No valid numeric candidates provided."}

    results = []
    for candidate in candidates:
        fields = [target, candidate] + [c for c in controls if c != candidate]
        complete_rows: List[List[float]] = []
        for row in rows:
            vals: List[float] = []
            ok = True
            for field in fields:
                value = to_float_permissive(row.get(field))
                if value is None:
                    ok = False
                    break
                vals.append(value)
            if ok:
                complete_rows.append(vals)
        if len(complete_rows) < len(fields) + policy.dependency_min_extra_rows:
            continue
        arr = np.array(complete_rows, dtype=float)
        y = arr[:, 0]
        x = arr[:, 1]
        raw = pearson_corr(x.tolist(), y.tolist())
        ctrl_mat = arr[:, 2:] if arr.shape[1] > 2 else np.empty((len(complete_rows), 0))
        partial = raw
        if ctrl_mat.shape[1] > 0:
            xa = np.column_stack([np.ones(len(complete_rows)), ctrl_mat])
            beta_y = np.linalg.lstsq(xa, y, rcond=None)[0]
            beta_x = np.linalg.lstsq(xa, x, rcond=None)[0]
            resid_y = y - xa @ beta_y
            resid_x = x - xa @ beta_x
            partial = pearson_corr(resid_x.tolist(), resid_y.tolist())
            xfull = np.column_stack([np.ones(len(complete_rows)), x, ctrl_mat])
        else:
            xfull = np.column_stack([np.ones(len(complete_rows)), x])
        coef = float(np.linalg.lstsq(xfull, y, rcond=None)[0][1])
        # Weighted blend of |partial|, |raw|, and |coef| balances different evidence types.
        score = (
            policy.dependency_score_partial_weight * abs(partial)
            + policy.dependency_score_raw_weight * abs(raw)
            + policy.dependency_score_coef_weight
            * min(abs(coef), policy.dependency_score_coef_clip)
        )
        results.append(
            {
                "candidate": candidate,
                "raw_correlation": round(raw, 6),
                "partial_correlation": round(partial, 6),
                "coefficient": round(coef, 6),
                "samples": len(complete_rows),
                "score": round(score, 6),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    requested_top_k = policy.dependency_default_top_k if top_k is None else int(top_k)
    return {
        "target": target,
        "controls": controls,
        "ranked_candidates": results[
            : max(1, min(requested_top_k, policy.dependency_rank_cap))
        ],
        "method_notes": [
            "Score blends |partial correlation|, |raw correlation|, and coefficient magnitude.",
            "Controls are applied to the partial-correlation and multivariate coefficient terms.",
        ],
    }


def solve_over_rows(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    target_field: str,
    goal_json: str,
    controllable_vars_json: str,
    constraints_json: str = "[]",
    top_k: Optional[int] = None,
    predicate_builder=None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Goal-seek over stored rows under optional constraints."""
    try:
        goal = json.loads(goal_json) if goal_json else {}
        controllable_vars = json.loads(controllable_vars_json) if controllable_vars_json else []
        constraints = json.loads(constraints_json) if constraints_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid goal/controllable_vars/constraints JSON."}
    if not isinstance(controllable_vars, list) or not controllable_vars:
        return {"error": "controllable_vars_json must be a non-empty list."}
    if not isinstance(constraints, list):
        return {"error": "constraints_json must be a list."}
    if predicate_builder is None:
        return {"error": "predicate_builder is required for solver constraints."}

    mode = str(goal.get("mode", "maximize")).lower()
    target_value = to_float_permissive(goal.get("target_value"))
    value_range = goal.get("range")

    predicates = []
    for cond in constraints:
        if not isinstance(cond, dict):
            return {"error": "Each constraint must be a condition dict."}
        pred = predicate_builder(cond, col_types)
        if pred is None:
            return {"error": f"Could not build predicate for constraint: {cond}"}
        predicates.append(pred)

    feasible = []
    for idx, row in enumerate(rows):
        tv = to_float_permissive(row.get(target_field))
        if tv is None:
            continue
        if not all(pred(row) for pred in predicates):
            continue
        score = tv
        if mode == "minimize":
            score = -tv
        elif mode == "hit_target" and target_value is not None:
            score = -abs(tv - target_value)
        elif mode == "range" and isinstance(value_range, list) and len(value_range) == 2:
            lo = to_float_permissive(value_range[0])
            hi = to_float_permissive(value_range[1])
            if lo is None or hi is None:
                return {"error": "goal.range must contain numeric values."}
            score = -abs(tv - ((lo + hi) / 2.0))
        feasible.append((score, idx, row))
    if not feasible:
        return {"error": "No feasible rows found under the given constraints."}

    feasible.sort(key=lambda item: item[0], reverse=True)
    requested_top_k = policy.solver_default_top_k if top_k is None else int(top_k)
    solutions = []
    for score, idx, row in feasible[: max(1, min(requested_top_k, policy.solver_solution_cap))]:
        candidate = {k: row.get(k) for k in controllable_vars if k in row}
        candidate[target_field] = row.get(target_field)
        candidate["_row_index"] = idx
        candidate["_score"] = round(score, 6)
        solutions.append(candidate)

    return {
        "target_field": target_field,
        "goal": goal,
        "constraints_applied": constraints,
        "candidate_count": len(feasible),
        "solutions": solutions,
        "method_notes": [
            "Solver ranks feasible stored rows rather than extrapolating beyond observed data.",
            "Use bh_manifold_slice, bh_orbit, or bh_dependency_screen to inspect the returned regions.",
        ],
    }
