"""Manifold navigation, inspection, and spatial reasoning methods for BinomialHash."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .insights import compute_insights
from .predicates import build_predicate, filter_rows_by_condition
from .schema import T_BOOL, T_DATE, T_NUMERIC, T_STRING

logger = logging.getLogger(__name__)


class _ManifoldMethodsMixin:
    """All manifold, navigation, and spatial reasoning methods.  Mixed into BinomialHash."""

    def _manifold_call(self, name: str, key: str, fn) -> Dict[str, Any]:
        """Shared wrapper for simple manifold-delegating methods."""
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        result = fn(slot)
        if isinstance(result, dict) and "error" in result:
            return result
        result = {"key": key, **result}
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] %s '%s' | %.1fms",
                    name, key, (time.perf_counter() - t0) * 1000)
        return result

    def _resolve_coord(self, manifold, inp: Any) -> Tuple[str, ...]:
        if isinstance(inp, dict):
            return tuple(str(inp.get(a.column, "")) for a in manifold.axes)
        if isinstance(inp, (list, tuple)):
            return tuple(str(c) for c in inp)
        return ()

    def _parse_json(self, raw: str, label: str = "JSON") -> Any:
        try:
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            return None

    # -- state & insights --

    def manifold_state(self, key: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}

        if slot.manifold is None:
            non_null_counts = {
                c: sum(1 for r in slot.rows if r.get(c) not in (None, ""))
                for c in slot.columns[:self._policy.manifold_non_null_preview_column_count]
            }
            numeric_cols = [c for c in slot.columns if slot.col_types.get(c) == T_NUMERIC]
            candidate_axes = [
                c for c in slot.columns
                if slot.col_types.get(c) in {T_STRING, T_DATE, T_BOOL}
                and non_null_counts.get(c, 0) >= 2
            ]
            return {
                "error": f"No manifold could be built for '{key}'.",
                "diagnostics": {
                    "row_count": slot.row_count,
                    "columns": slot.columns,
                    "numeric_columns": numeric_cols[:self._policy.manifold_diagnostic_preview_column_count],
                    "candidate_axis_columns": candidate_axes[:self._policy.manifold_diagnostic_preview_column_count],
                    "non_null_counts": {
                        k: non_null_counts[k]
                        for k in slot.columns[:self._policy.manifold_diagnostic_preview_column_count]
                    },
                    "note": "Common causes: wrong embedded table, too many nulls, only one axis, or no numeric fields after axis selection.",
                },
            }

        nesting_info: Dict[str, Any] = {}
        if slot.nesting is not None:
            n = slot.nesting
            nesting_info = {
                "max_depth": n.max_depth,
                "path_signature": n.path_signature,
                "branching_by_depth": n.branching_by_depth,
            }

        surface = slot.manifold.to_summary()
        result = {
            "key": key, "label": slot.label, "row_count": slot.row_count,
            "nesting_topology": nesting_info, "manifold": surface,
        }
        out_chars = len(json.dumps(result, default=str))
        self._track(0, out_chars)
        logger.info("[BH-perf] manifold_state '%s' | surface=%s genus=%d V=%d F=%s | %.1fms",
                    key, surface["surface"], surface["genus"],
                    surface["graph"]["V"], surface["face_complex"]["F"],
                    (time.perf_counter() - t0) * 1000)
        return result

    def manifold_insights(self, key: str, objective_json: str,
                          top_k: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        objective = self._parse_json(objective_json)
        if objective is None:
            return {"error": f"Invalid objective_json: {objective_json[:120]}"}

        requested_top_k = self._policy.manifold_insights_default_top_k if top_k is None else int(top_k)
        result = compute_insights(
            slot.rows, slot.columns, slot.col_types, objective,
            top_k=requested_top_k,
            driver_limit=self._policy.manifold_insights_driver_limit,
            driver_bins=self._policy.manifold_insights_driver_bins,
            regime_z_threshold=self._policy.manifold_insights_regime_z_threshold,
            target_bins=self._policy.manifold_insights_target_bins,
            branch_context_limit=self._policy.manifold_insights_branch_context_limit,
            branch_min_rows=self._policy.manifold_insights_branch_min_rows,
            branch_min_values=self._policy.manifold_insights_branch_min_values,
        )
        if "error" in result:
            return result
        result["key"] = key
        result["label"] = slot.label
        out_chars = len(json.dumps(result, default=str))
        self._track(0, out_chars)
        logger.info("[BH-perf] insights '%s' | %.1fms",
                    key, (time.perf_counter() - t0) * 1000)
        return result

    # -- point navigation --

    def manifold_navigate(self, key: str, coord_json: str,
                          target_field: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold built for '{key}'."}
        coord_input = self._parse_json(coord_json)
        if coord_input is None:
            return {"error": f"Invalid coord_json: {coord_json[:100]}"}
        coord = self._resolve_coord(slot.manifold, coord_input)
        result = slot.manifold.navigate(coord, target_field)
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] navigate '%s' coord=%s | %.1fms",
                    key, coord, (time.perf_counter() - t0) * 1000)
        return result

    def geodesic(self, key: str, start_json: str, end_json: str,
                 target_field: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        s = self._parse_json(start_json)
        e = self._parse_json(end_json)
        if s is None or e is None:
            return {"error": "Invalid start/end JSON."}
        start_coord = self._resolve_coord(slot.manifold, s)
        end_coord = self._resolve_coord(slot.manifold, e)
        result = slot.manifold.geodesic_path(start_coord, end_coord, target_field)
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] geodesic '%s' %s→%s | %.1fms",
                    key, start_coord, end_coord, (time.perf_counter() - t0) * 1000)
        return result

    def controlled_walk(self, key: str, walk_axis: str,
                        target_field: str) -> Dict[str, Any]:
        return self._manifold_call("walk", key,
            lambda s: s.manifold.controlled_walk(walk_axis, target_field))

    def orbit(self, key: str, center_json: str, radius: int,
              target_field: Optional[str] = None,
              resolution: Optional[int] = None, mode: str = "ring") -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        res = self._policy.orbit_default_resolution if resolution is None else int(resolution)
        center_inp = self._parse_json(center_json)
        if center_inp is None:
            return {"error": "Invalid center_json."}
        center = self._resolve_coord(slot.manifold, center_inp)
        result = slot.manifold.orbit(center, int(radius), target_field, res, str(mode))
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] orbit '%s' r=%s | %.1fms",
                    key, radius, (time.perf_counter() - t0) * 1000)
        return result

    def multiscale_view(self, key: str, center_json: str, radii_json: str,
                        target_field: Optional[str] = None,
                        resolution: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        res = self._policy.multiscale_default_resolution if resolution is None else int(resolution)
        center_inp = self._parse_json(center_json)
        radii = self._parse_json(radii_json)
        if center_inp is None or radii is None:
            return {"error": "Invalid center_json or radii_json."}
        if not isinstance(radii, list):
            return {"error": "radii_json must be a list of integers."}
        center = self._resolve_coord(slot.manifold, center_inp)
        result = slot.manifold.multiscale_view(center, radii, target_field, res)
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] multiscale '%s' radii=%s | %.1fms",
                    key, radii, (time.perf_counter() - t0) * 1000)
        return result

    # -- frontier & extremum tracing --

    def frontier(self, key: str, condition_a_json: str, condition_b_json: str,
                 target_field: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        cond_a = self._parse_json(condition_a_json)
        cond_b = self._parse_json(condition_b_json)
        if cond_a is None or cond_b is None:
            return {"error": "Invalid condition JSON."}
        gp_col_types = dict(slot.col_types)
        for ax in slot.manifold.axes:
            gp_col_types[ax.column] = T_STRING
        gp_col_types.update({"density": T_NUMERIC, "curvature": T_NUMERIC, "forman_ricci": T_NUMERIC})
        pred_a = build_predicate(cond_a, gp_col_types)
        pred_b = build_predicate(cond_b, gp_col_types)
        if pred_a is None or pred_b is None:
            return {"error": "Could not build predicates from condition JSON."}
        from .manifold.navigation import find_frontier_edges
        edges = find_frontier_edges(slot.manifold, pred_a, pred_b, target_field, limit)
        result = {"key": key, "target_field": target_field,
                  "frontier_edges": edges, "frontier_count": len(edges)}
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] frontier '%s' edges=%d | %.1fms",
                    key, len(edges), (time.perf_counter() - t0) * 1000)
        return result

    def basin(self, key: str, seed_json: str, target_field: str,
              direction: str = "descend") -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        seed_inp = self._parse_json(seed_json)
        if seed_inp is None:
            return {"error": "Invalid seed_json."}
        seed = self._resolve_coord(slot.manifold, seed_inp)
        result = slot.manifold.basin(seed, target_field, direction)
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] basin '%s' | %.1fms",
                    key, (time.perf_counter() - t0) * 1000)
        return result

    def ridge_trace(self, key: str, seed_json: str, target_field: str,
                    max_steps: int = 25) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        seed_inp = self._parse_json(seed_json)
        if seed_inp is None:
            return {"error": "Invalid seed_json."}
        seed = self._resolve_coord(slot.manifold, seed_inp)
        result = slot.manifold.trace_extremum(seed, target_field, "ridge", int(max_steps))
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] ridge '%s' | %.1fms",
                    key, (time.perf_counter() - t0) * 1000)
        return result

    def valley_trace(self, key: str, seed_json: str, target_field: str,
                     max_steps: int = 25) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        seed_inp = self._parse_json(seed_json)
        if seed_inp is None:
            return {"error": "Invalid seed_json."}
        seed = self._resolve_coord(slot.manifold, seed_inp)
        result = slot.manifold.trace_extremum(seed, target_field, "valley", int(max_steps))
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] valley '%s' | %.1fms",
                    key, (time.perf_counter() - t0) * 1000)
        return result

    def coverage_audit(self, key: str) -> Dict[str, Any]:
        return self._manifold_call("coverage", key,
            lambda s: s.manifold.coverage_audit())

    def wrap_audit(self, key: str, axis: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        from .manifold.diagnostics import boundary_wrap_diagnostics
        axes = slot.manifold.axes
        indices = [i for i, ax in enumerate(axes) if axis is None or ax.column == axis]
        diags = [boundary_wrap_diagnostics(slot.manifold.grid, axes, slot.manifold.field_columns, i)
                 for i in indices]
        result = {"key": key, "requested_axis": axis, "axes": diags,
                  "surface_confidence": slot.manifold.surface_confidence_obj}
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] wrap_audit '%s' axis=%s | %.1fms",
                    key, axis, (time.perf_counter() - t0) * 1000)
        return result

    # -- spatial reasoning --

    def heat_kernel(self, key: str, target_field: Optional[str] = None,
                    n_eigen: int = 20, top_k: int = 10) -> Dict[str, Any]:
        return self._manifold_call("heat_kernel", key,
            lambda s: s.manifold.heat_kernel(
                target_field=target_field, n_eigen=int(n_eigen),
                top_k_bottlenecks=int(top_k)))

    def reeb_graph(self, key: str, target_field: str,
                   n_levels: int = 20) -> Dict[str, Any]:
        return self._manifold_call("reeb_graph", key,
            lambda s: s.manifold.reeb_graph(target_field, n_levels=int(n_levels)))

    def vector_field(self, key: str, target_field: str,
                     top_k: int = 10) -> Dict[str, Any]:
        return self._manifold_call("vector_field", key,
            lambda s: s.manifold.vector_field(target_field, top_k=int(top_k)))

    def laplacian_spectrum(self, key: str, n_eigen: int = 15,
                           n_clusters: Optional[int] = None) -> Dict[str, Any]:
        return self._manifold_call("laplacian", key,
            lambda s: s.manifold.laplacian_spectrum(
                n_eigen=int(n_eigen), n_clusters=n_clusters))

    def scalar_harmonics(self, key: str, target_field: str,
                         n_modes: int = 10, top_k: int = 10) -> Dict[str, Any]:
        return self._manifold_call("harmonics", key,
            lambda s: s.manifold.scalar_harmonics(
                target_field, n_modes=int(n_modes), top_k_anomalies=int(top_k)))

    def diffusion_distance(self, key: str, landmarks_json: Optional[str] = None,
                           time_param: float = 1.0,
                           n_landmarks: int = 8) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        if slot.manifold is None:
            return {"error": f"No manifold for '{key}'."}
        landmark_coords = None
        if landmarks_json:
            try:
                raw = json.loads(landmarks_json)
                landmark_coords = [
                    tuple(str(v) for v in lm.values()) if isinstance(lm, dict)
                    else tuple(str(v) for v in lm)
                    for lm in raw
                ]
            except (json.JSONDecodeError, TypeError):
                return {"error": "Invalid landmarks_json."}
        from .manifold.spatial import diffusion_distance as _dd
        result = _dd(slot.manifold, landmark_coords=landmark_coords,
                     time_param=float(time_param), n_landmarks=int(n_landmarks))
        if "error" in result:
            return result
        result = {"key": key, **result}
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] diffusion_dist '%s' | %.1fms",
                    key, (time.perf_counter() - t0) * 1000)
        return result

    # -- slicing --

    def manifold_slice(self, key: str, condition_json: str,
                       target_field: Optional[str] = None) -> Dict[str, Any]:
        """Build sub-manifold from rows matching a condition."""
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        condition = self._parse_json(condition_json)
        if condition is None:
            return {"error": f"Invalid condition_json: {condition_json[:120]}"}

        col = condition.get("column")
        op = condition.get("op", ">")
        value = condition.get("value")
        if not col or value is None:
            return {"error": "condition_json needs 'column', 'op', and 'value'."}

        filtered_rows = filter_rows_by_condition(slot.rows, col, op, value)
        if len(filtered_rows) < 10:
            return {"error": f"Only {len(filtered_rows)} rows match — need at least 10."}

        from .manifold.builder import build_manifold
        sub_manifold = build_manifold(filtered_rows, slot.columns, slot.col_types, slot.col_stats)
        if sub_manifold is None:
            return {"error": "Could not build sub-manifold from filtered rows."}

        summary = sub_manifold.to_summary()
        summary["slice_condition"] = condition
        summary["slice_rows"] = len(filtered_rows)
        summary["parent_rows"] = slot.row_count
        self._track(0, len(json.dumps(summary, default=str)))
        logger.info("[BH-perf] slice '%s' %d→%d rows | %.1fms",
                    key, slot.row_count, len(filtered_rows),
                    (time.perf_counter() - t0) * 1000)
        return summary
