"""Regression tests for the extracted modules.

Covers:
  - insights.py          (driver discovery, surprises, regimes, divergence, counterfactual, pipeline)
  - predicates.py        (filter_rows_by_condition)
  - exporters/excel.py   (export_excel_batch)
  - exporters/chunks.py  (slot_to_chunk)
  - manifold/navigation.py (gridpoint_record, find_frontier_edges)
  - core.py promotion    (backward-compat shim, BinomialHash identity)
  - BinomialHash integration (manifold_insights, frontier, manifold_slice, to_excel_batch, to_chunks)
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Fixtures: deterministic tabular data
# ---------------------------------------------------------------------------

def _linear_dataset(n: int = 60, noise: float = 0.1, seed: int = 42) -> List[Dict[str, Any]]:
    """y = 2*x + 10, with controlled noise.  Also adds a 'category' string col."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x = float(i)
        y = 2.0 * x + 10.0 + rng.gauss(0, noise)
        z = rng.random() * 100  # unrelated numeric col
        rows.append({"x": x, "y": y, "z": z, "cat": f"group_{i % 3}"})
    return rows


ROWS = _linear_dataset()
COLUMNS = ["x", "y", "z", "cat"]
COL_TYPES = {"x": "numeric", "y": "numeric", "z": "numeric", "cat": "string"}
COL_STATS: Dict[str, Dict[str, Any]] = {
    "x": {"min": 0, "max": 59, "mean": 29.5},
    "y": {"min": 10, "max": 128, "mean": 69},
    "z": {"min": 0, "max": 100, "mean": 50},
    "cat": {"top_values": ["group_0", "group_1", "group_2"], "unique_count": 3},
}


def _regime_dataset(n: int = 120, seed: int = 7) -> List[Dict[str, Any]]:
    """Dataset with an abrupt regime change at x=60."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        x = float(i)
        y = 1.0 * x + rng.gauss(0, 0.5) if x < 60 else 5.0 * x + rng.gauss(0, 0.5)
        rows.append({"x": x, "y": y, "z": rng.random()})
    return rows


# ---------------------------------------------------------------------------
# 1. insights.py — discover_best_driver
# ---------------------------------------------------------------------------

class TestDiscoverBestDriver:
    def test_finds_strong_driver(self):
        from binomialhash.insights import discover_best_driver
        best = discover_best_driver(ROWS, "y", ["x", "z"])
        assert best is not None
        assert best["driver"] == "x"
        assert best["r2"] > 0.99, f"Expected near-perfect R2, got {best['r2']}"
        assert best["samples"] == len(ROWS)

    def test_returns_none_below_min_samples(self):
        from binomialhash.insights import discover_best_driver
        short = ROWS[:10]
        result = discover_best_driver(short, "y", ["x"], min_samples=30)
        assert result is None

    def test_returns_none_with_no_drivers(self):
        from binomialhash.insights import discover_best_driver
        result = discover_best_driver(ROWS, "y", [])
        assert result is None

    def test_picks_higher_r2_among_drivers(self):
        from binomialhash.insights import discover_best_driver
        best = discover_best_driver(ROWS, "y", ["z", "x"])
        assert best is not None
        assert best["driver"] == "x"

    def test_model_keys(self):
        from binomialhash.insights import discover_best_driver
        best = discover_best_driver(ROWS, "y", ["x"])
        assert best is not None
        for key in ("driver", "slope", "intercept", "r2", "corr", "samples"):
            assert key in best, f"Missing key: {key}"

    def test_slope_and_intercept_approximate(self):
        from binomialhash.insights import discover_best_driver
        best = discover_best_driver(ROWS, "y", ["x"])
        assert best is not None
        assert abs(best["slope"] - 2.0) < 0.1
        assert abs(best["intercept"] - 10.0) < 1.0


# ---------------------------------------------------------------------------
# 2. insights.py — compute_surprises
# ---------------------------------------------------------------------------

class TestComputeSurprises:
    def test_returns_top_k(self):
        from binomialhash.insights import compute_surprises
        surprises = compute_surprises(ROWS, "y", "x", slope=2.0, intercept=10.0, top_k=3)
        assert len(surprises) == 3

    def test_surprise_keys(self):
        from binomialhash.insights import compute_surprises
        surprises = compute_surprises(ROWS, "y", "x", slope=2.0, intercept=10.0, top_k=1)
        assert len(surprises) >= 1
        s = surprises[0]
        for key in ("row_index", "actual", "expected", "residual", "residual_z"):
            assert key in s

    def test_residual_is_actual_minus_expected(self):
        from binomialhash.insights import compute_surprises
        surprises = compute_surprises(ROWS, "y", "x", slope=2.0, intercept=10.0, top_k=5)
        for s in surprises:
            assert abs(s["residual"] - (s["actual"] - s["expected"])) < 1e-4

    def test_top_k_one_minimum(self):
        from binomialhash.insights import compute_surprises
        surprises = compute_surprises(ROWS, "y", "x", slope=2.0, intercept=10.0, top_k=0)
        assert len(surprises) >= 1

    def test_skips_none_values(self):
        from binomialhash.insights import compute_surprises
        rows_with_none = ROWS[:30] + [{"x": None, "y": 100}] + [{"x": 5, "y": None}]
        surprises = compute_surprises(rows_with_none, "y", "x", slope=2.0, intercept=10.0, top_k=2)
        for s in surprises:
            assert s["actual"] is not None
            assert s["expected"] is not None


# ---------------------------------------------------------------------------
# 3. insights.py — compute_regime_boundaries
# ---------------------------------------------------------------------------

class TestComputeRegimeBoundaries:
    def test_detects_regime_jump(self):
        from binomialhash.insights import compute_regime_boundaries
        rows = _regime_dataset()
        boundaries = compute_regime_boundaries(rows, "y", "x", bins=6, z_threshold=1.0)
        assert len(boundaries) >= 1, "Should detect at least one regime boundary"

    def test_boundary_keys(self):
        from binomialhash.insights import compute_regime_boundaries
        rows = _regime_dataset()
        boundaries = compute_regime_boundaries(rows, "y", "x", bins=6, z_threshold=1.0)
        if boundaries:
            b = boundaries[0]
            for key in ("between_bucket", "driver_range", "target_jump", "jump_z"):
                assert key in b

    def test_uniform_slope_has_uniform_jumps(self):
        from binomialhash.insights import compute_regime_boundaries
        boundaries = compute_regime_boundaries(ROWS, "y", "x", bins=6, z_threshold=1.5)
        if boundaries:
            jumps = [b["target_jump"] for b in boundaries]
            spread = max(jumps) - min(jumps)
            assert spread < max(jumps) * 0.5, "Uniform linear data should have roughly equal bucket jumps"

    def test_empty_if_no_numeric_data(self):
        from binomialhash.insights import compute_regime_boundaries
        rows = [{"x": "a", "y": "b"} for _ in range(60)]
        boundaries = compute_regime_boundaries(rows, "y", "x", bins=4)
        assert boundaries == []


# ---------------------------------------------------------------------------
# 4. insights.py — compute_branch_divergence
# ---------------------------------------------------------------------------

class TestComputeBranchDivergence:
    def test_returns_list(self):
        from binomialhash.insights import compute_branch_divergence
        result = compute_branch_divergence(
            ROWS, "y", ["x", "y", "z"], target_bins=3, min_rows=5, min_values=5,
        )
        assert isinstance(result, list)

    def test_divergence_keys(self):
        from binomialhash.insights import compute_branch_divergence
        result = compute_branch_divergence(
            ROWS, "y", ["x", "y", "z"], target_bins=3, min_rows=5, min_values=5,
        )
        if result:
            d = result[0]
            for key in ("target_quantile_bucket", "target_range", "rows", "latent_spread_score"):
                assert key in d

    def test_sorted_desc_by_spread(self):
        from binomialhash.insights import compute_branch_divergence
        result = compute_branch_divergence(
            ROWS, "y", ["x", "y", "z"], target_bins=3, min_rows=5, min_values=5,
        )
        if len(result) >= 2:
            assert result[0]["latent_spread_score"] >= result[1]["latent_spread_score"]

    def test_empty_when_too_few_rows(self):
        from binomialhash.insights import compute_branch_divergence
        result = compute_branch_divergence(
            ROWS[:5], "y", ["x", "y", "z"], target_bins=3, min_rows=100,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 5. insights.py — build_counterfactual
# ---------------------------------------------------------------------------

class TestBuildCounterfactual:
    def _model(self):
        return {"slope": 2.0, "intercept": 10.0, "r2": 0.99, "corr": 0.995}

    def test_maximize_positive_slope(self):
        from binomialhash.insights import build_counterfactual
        cf = build_counterfactual(ROWS, "y", "x", self._model(), {"goal": "maximize"})
        assert cf["suggested_action"]["driver_direction"] == "increase"

    def test_minimize_positive_slope(self):
        from binomialhash.insights import build_counterfactual
        cf = build_counterfactual(ROWS, "y", "x", self._model(), {"goal": "minimize"})
        assert cf["suggested_action"]["driver_direction"] == "decrease"

    def test_maximize_negative_slope(self):
        from binomialhash.insights import build_counterfactual
        model = {"slope": -3.0, "intercept": 100.0, "r2": 0.95, "corr": -0.97}
        cf = build_counterfactual(ROWS, "y", "x", model, {"goal": "maximize"})
        assert cf["suggested_action"]["driver_direction"] == "decrease"

    def test_minimize_negative_slope(self):
        from binomialhash.insights import build_counterfactual
        model = {"slope": -3.0, "intercept": 100.0, "r2": 0.95, "corr": -0.97}
        cf = build_counterfactual(ROWS, "y", "x", model, {"goal": "minimize"})
        assert cf["suggested_action"]["driver_direction"] == "increase"

    def test_reach_value_estimate(self):
        from binomialhash.insights import build_counterfactual
        cf = build_counterfactual(ROWS, "y", "x", self._model(), {"goal": "maximize", "target_value": 50.0})
        assert "reach_value_estimate" in cf
        est = cf["reach_value_estimate"]
        expected_driver = (50.0 - 10.0) / 2.0
        assert abs(est["required_driver_value"] - expected_driver) < 0.1

    def test_law_formula_format(self):
        from binomialhash.insights import build_counterfactual
        cf = build_counterfactual(ROWS, "y", "x", self._model(), {"goal": "maximize"})
        assert "law" in cf
        assert "≈" in cf["law"]["formula"]

    def test_default_goal_is_maximize(self):
        from binomialhash.insights import build_counterfactual
        cf = build_counterfactual(ROWS, "y", "x", self._model(), {})
        assert cf["suggested_action"]["goal"] == "maximize"


# ---------------------------------------------------------------------------
# 6. insights.py — compute_insights (full pipeline)
# ---------------------------------------------------------------------------

class TestComputeInsights:
    def test_full_pipeline_success(self):
        from binomialhash.insights import compute_insights
        result = compute_insights(ROWS, COLUMNS, COL_TYPES, {"target": "y"}, top_k=3)
        assert "error" not in result
        assert "discovered_law" in result
        assert "insights" in result
        assert "method_notes" in result
        assert result["discovered_law"]["driver"] == "x"
        assert len(result["insights"]["surprises"]) == 3

    def test_auto_selects_first_numeric_if_target_missing(self):
        from binomialhash.insights import compute_insights
        result = compute_insights(ROWS, COLUMNS, COL_TYPES, {"target": "nonexistent"})
        assert "error" not in result
        assert result["objective"]["target"] in ("x", "y", "z")

    def test_error_no_numeric_columns(self):
        from binomialhash.insights import compute_insights
        result = compute_insights(ROWS, ["cat"], {"cat": "string"}, {"target": "cat"})
        assert "error" in result
        assert "numeric" in result["error"].lower()

    def test_error_insufficient_data(self):
        from binomialhash.insights import compute_insights
        result = compute_insights(ROWS[:5], COLUMNS, COL_TYPES, {"target": "y"})
        assert "error" in result
        assert "30" in result["error"]

    def test_error_no_fittable_law(self):
        from binomialhash.insights import compute_insights
        rows = [{"a": float(i), "b": float(i)} for i in range(40)]
        result = compute_insights(rows, ["a"], {"a": "numeric"}, {"target": "a"})
        assert "error" in result

    def test_objective_passthrough(self):
        from binomialhash.insights import compute_insights
        result = compute_insights(ROWS, COLUMNS, COL_TYPES, {"target": "y", "goal": "minimize", "target_value": 42})
        assert result["objective"]["goal"] == "minimize"
        assert result["objective"]["target_value"] == 42


# ---------------------------------------------------------------------------
# 7. predicates.py — filter_rows_by_condition
# ---------------------------------------------------------------------------

class TestFilterRowsByCondition:
    ROWS = [
        {"price": 10, "name": "apple"},
        {"price": 20, "name": "banana"},
        {"price": 30, "name": "cherry"},
        {"price": None, "name": "date"},
        {"price": 40, "name": None},
    ]

    def test_greater_than(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", ">", 15)
        assert len(result) == 3
        assert all(float(r["price"]) > 15 for r in result)

    def test_greater_equal(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", ">=", 20)
        assert len(result) == 3

    def test_less_than(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", "<", 25)
        assert len(result) == 2

    def test_less_equal(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", "<=", 20)
        assert len(result) == 2

    def test_equal_numeric(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", "=", 20)
        assert len(result) == 1
        assert result[0]["name"] == "banana"

    def test_not_equal_numeric(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", "!=", 20)
        assert len(result) == 3

    def test_equal_string(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "name", "=", "apple")
        assert len(result) == 1
        assert result[0]["price"] == 10

    def test_not_equal_string(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "name", "!=", "apple")
        assert len(result) == 3

    def test_none_values_skipped(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "price", ">", 0)
        assert all(r["price"] is not None for r in result)

    def test_missing_column(self):
        from binomialhash.predicates import filter_rows_by_condition
        result = filter_rows_by_condition(self.ROWS, "missing_col", ">", 0)
        assert result == []

    def test_numeric_string_coercion(self):
        from binomialhash.predicates import filter_rows_by_condition
        rows = [{"val": "15.5"}, {"val": "25.5"}]
        result = filter_rows_by_condition(rows, "val", ">", "20")
        assert len(result) == 1
        assert result[0]["val"] == "25.5"


# ---------------------------------------------------------------------------
# 8. exporters/excel.py — export_excel_batch
# ---------------------------------------------------------------------------

class TestExportExcelBatch:
    ROWS = [{"a": i, "b": i * 10, "c": f"row_{i}"} for i in range(20)]
    COLS = ["a", "b", "c"]
    TYPES = {"a": "numeric", "b": "numeric", "c": "string"}

    def test_basic_export(self):
        from binomialhash.exporters.excel import export_excel_batch
        result = export_excel_batch(self.ROWS, self.COLS, self.TYPES, "k1", "Test", 20)
        assert result["key"] == "k1"
        assert result["label"] == "Test"
        assert result["headers"] == ["a", "b", "c"]
        assert result["total_exported"] == 20
        assert result["total_available"] == 20

    def test_column_projection(self):
        from binomialhash.exporters.excel import export_excel_batch
        result = export_excel_batch(self.ROWS, self.COLS, self.TYPES, "k1", "Test", 20, select_columns=["a", "c"])
        assert result["headers"] == ["a", "c"]
        assert all(len(row) == 2 for row in result["values"])

    def test_sort_desc(self):
        from binomialhash.exporters.excel import export_excel_batch
        result = export_excel_batch(self.ROWS, self.COLS, self.TYPES, "k1", "Test", 20, sort_by="a", sort_desc=True)
        assert result["values"][0][0] == 19
        assert result["values"][-1][0] == 0

    def test_sort_asc(self):
        from binomialhash.exporters.excel import export_excel_batch
        shuffled = list(reversed(self.ROWS))
        result = export_excel_batch(shuffled, self.COLS, self.TYPES, "k1", "Test", 20, sort_by="a", sort_desc=False)
        assert result["values"][0][0] == 0

    def test_max_rows_cap(self):
        from binomialhash.exporters.excel import export_excel_batch
        result = export_excel_batch(self.ROWS, self.COLS, self.TYPES, "k1", "Test", 20, max_rows=5)
        assert result["total_exported"] == 5

    def test_no_hidden_hard_cap(self):
        from binomialhash.exporters.excel import export_excel_batch
        big = [{"a": i} for i in range(600)]
        result = export_excel_batch(big, ["a"], {"a": "numeric"}, "k1", "Test", 600, max_rows=600)
        assert result["total_exported"] == 600

    def test_total_available_reflects_original(self):
        from binomialhash.exporters.excel import export_excel_batch
        result = export_excel_batch(self.ROWS, self.COLS, self.TYPES, "k1", "Test", 9999, max_rows=5)
        assert result["total_available"] == 9999


# ---------------------------------------------------------------------------
# 9. exporters/chunks.py — slot_to_chunk
# ---------------------------------------------------------------------------

class TestSlotToChunk:
    def test_basic_chunk(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        rows = [{"price": 100, "name": "AAPL"}, {"price": 200, "name": "MSFT"}]
        chunk = slot_to_chunk(
            "k1", "Stock Prices", "abc123def456", 2,
            rows, ["price", "name"], {"price": "numeric", "name": "string"},
            {"price": {"min": 100, "max": 200, "mean": 150}, "name": {"top_values": ["AAPL", "MSFT"], "unique_count": 2}},
        )
        assert chunk["chunk_type"] == "bh_dataset"
        assert chunk["sheet_name"] == "_bh"
        assert chunk["cell_range"] is None
        assert chunk["metadata"]["bh_key"] == "k1"
        assert chunk["metadata"]["label"] == "Stock Prices"
        assert chunk["metadata"]["row_count"] == 2

    def test_content_includes_schema(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        chunk = slot_to_chunk(
            "k1", "Test", "fp12345678901234", 5,
            [{"a": 1}], ["a"], {"a": "numeric"}, {"a": {"min": 0, "max": 10, "mean": 5}},
        )
        assert "Columns:" in chunk["content"]
        assert "a (numeric)" in chunk["content"]
        assert "range [0..10]" in chunk["content"]

    def test_content_truncation(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        long_cols = [f"col_{i}" for i in range(100)]
        col_types = {c: "string" for c in long_cols}
        col_stats = {c: {"top_values": ["a"], "unique_count": 1} for c in long_cols}
        chunk = slot_to_chunk("k1", "Big", "fp123456", 1, [{}], long_cols, col_types, col_stats, max_content_chars=200)
        assert len(chunk["content"]) <= 200

    def test_fingerprint_truncation(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        chunk = slot_to_chunk("k1", "T", "abcdefghijklmnopqrstuvwxyz", 1, [], ["a"], {"a": "numeric"}, {})
        assert chunk["metadata"]["bh_fingerprint"] == "abcdefghijklmnop"
        assert "abcdefghijkl" in chunk["content"]

    def test_date_column_format(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        chunk = slot_to_chunk(
            "k1", "Dates", "fp123", 2,
            [{"d": "2024-01-01"}], ["d"], {"d": "date"},
            {"d": {"min_date": "2024-01-01", "max_date": "2024-12-31"}},
        )
        assert "2024-01-01" in chunk["content"]
        assert "2024-12-31" in chunk["content"]

    def test_metadata_column_limit(self):
        from binomialhash.exporters.chunks import slot_to_chunk
        cols = [f"c{i}" for i in range(30)]
        chunk = slot_to_chunk("k1", "T", "fp", 1, [], cols, {c: "string" for c in cols}, {})
        assert len(chunk["metadata"]["columns"]) == 20


# ---------------------------------------------------------------------------
# 10. manifold/navigation.py — gridpoint_record
# ---------------------------------------------------------------------------

class TestGridpointRecord:
    def test_basic_record(self):
        from binomialhash.manifold.structures import ManifoldAxis, ManifoldSurface, GridPoint
        from binomialhash.manifold.navigation import gridpoint_record

        ax = ManifoldAxis(column="sector", values=["Tech", "Health"], ordered=False, axis_type="categorical", size=2)
        gp = GridPoint(
            index=0, axis_coords=("Tech",),
            field_values={"revenue": 100.0, "margin": 0.15},
            curvature=0.5, forman_ricci=-0.2, density=3,
        )
        surface = ManifoldSurface(
            axes=[ax], field_columns=["revenue", "margin"],
            grid={("Tech",): gp}, surface_name="test", genus=0,
            orientable=True, euler_characteristic=1, handles=0,
            crosscaps=0, vertex_count=1, edge_count=0,
        )

        rec = gridpoint_record(surface, gp)
        assert rec["sector"] == "Tech"
        assert rec["revenue"] == 100.0
        assert rec["margin"] == 0.15
        assert rec["density"] == 3
        assert rec["curvature"] == 0.5
        assert rec["forman_ricci"] == -0.2


# ---------------------------------------------------------------------------
# 11. manifold/navigation.py — find_frontier_edges
# ---------------------------------------------------------------------------

class TestFindFrontierEdges:
    def _build_linear_surface(self):
        """3-point linear grid: A(0) -- B(1) -- C(2). density=10,20,30."""
        from binomialhash.manifold.structures import ManifoldAxis, ManifoldSurface, GridPoint

        ax = ManifoldAxis(column="idx", values=["0", "1", "2"], ordered=True, axis_type="ordinal", size=3)
        gp0 = GridPoint(index=0, axis_coords=("0",), field_values={"val": 10.0}, density=10, neighbors=[1])
        gp1 = GridPoint(index=1, axis_coords=("1",), field_values={"val": 20.0}, density=20, neighbors=[0, 2])
        gp2 = GridPoint(index=2, axis_coords=("2",), field_values={"val": 30.0}, density=30, neighbors=[1])
        surface = ManifoldSurface(
            axes=[ax], field_columns=["val"],
            grid={("0",): gp0, ("1",): gp1, ("2",): gp2},
            surface_name="line", genus=0, orientable=True,
            euler_characteristic=1, handles=0, crosscaps=0,
            vertex_count=3, edge_count=2,
        )
        return surface

    def test_finds_crossing_edge(self):
        from binomialhash.manifold.navigation import find_frontier_edges
        surface = self._build_linear_surface()
        pred_a = lambda rec: int(rec["density"]) <= 10
        pred_b = lambda rec: int(rec["density"]) >= 20
        edges = find_frontier_edges(surface, pred_a, pred_b)
        assert len(edges) == 1
        assert set(edges[0]["labels"]) == {"A", "B"}

    def test_no_crossing_same_region(self):
        from binomialhash.manifold.navigation import find_frontier_edges
        surface = self._build_linear_surface()
        pred_a = lambda rec: True
        pred_b = lambda rec: False
        edges = find_frontier_edges(surface, pred_a, pred_b)
        assert len(edges) == 0

    def test_target_jump_included(self):
        from binomialhash.manifold.navigation import find_frontier_edges
        surface = self._build_linear_surface()
        pred_a = lambda rec: float(rec["density"]) <= 10
        pred_b = lambda rec: float(rec["density"]) >= 20
        edges = find_frontier_edges(surface, pred_a, pred_b, target_field="val")
        assert len(edges) == 1
        assert "target_jump" in edges[0]
        assert abs(edges[0]["target_jump"] - 10.0) < 0.01

    def test_limit_respected(self):
        from binomialhash.manifold.navigation import find_frontier_edges
        surface = self._build_linear_surface()
        pred_a = lambda rec: float(rec["density"]) <= 15
        pred_b = lambda rec: float(rec["density"]) > 15
        edges = find_frontier_edges(surface, pred_a, pred_b, limit=1)
        assert len(edges) <= 1

    def test_no_duplicate_edges(self):
        from binomialhash.manifold.navigation import find_frontier_edges
        surface = self._build_linear_surface()
        pred_a = lambda rec: float(rec["density"]) <= 15
        pred_b = lambda rec: float(rec["density"]) > 15
        edges = find_frontier_edges(surface, pred_a, pred_b, limit=100)
        edge_tuples = [tuple(sorted([str(e["from"]), str(e["to"])])) for e in edges]
        assert len(edge_tuples) == len(set(edge_tuples))


# ---------------------------------------------------------------------------
# 12. core.py promotion — backward-compat and identity
# ---------------------------------------------------------------------------

class TestCorePromotion:
    def test_binomial_hash_identity(self):
        from binomialhash.core import BinomialHash as BH_core
        from binomialhash._bootstrap_core import BinomialHash as BH_shim
        from binomialhash import BinomialHash as BH_top
        assert BH_core is BH_top
        assert BH_shim is BH_top

    def test_slot_identity(self):
        from binomialhash.core import BinomialHashSlot as S1
        from binomialhash._bootstrap_core import BinomialHashSlot as S2
        assert S1 is S2

    def test_policy_identity(self):
        from binomialhash.core import BinomialHashPolicy as P1
        from binomialhash._bootstrap_core import BinomialHashPolicy as P2
        assert P1 is P2

    def test_constants_from_shim(self):
        from binomialhash._bootstrap_core import (
            BUDGET_BYTES,
            INGEST_THRESHOLD_CHARS,
            MAX_PREVIEW_ROWS,
            MAX_RETRIEVE_ROWS,
            MAX_SLOTS,
        )
        assert INGEST_THRESHOLD_CHARS == 3000
        assert MAX_PREVIEW_ROWS == 3
        assert MAX_RETRIEVE_ROWS == 50
        assert MAX_SLOTS == 50
        assert BUDGET_BYTES == 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# 13. BinomialHash integration — end-to-end method delegation
# ---------------------------------------------------------------------------

def _make_bh_with_data(n: int = 60) -> "BinomialHash":
    """Create a BinomialHash instance and ingest a synthetic dataset.

    Adds extra string fields to push the JSON payload above the 3000-char
    ingest threshold so data is actually stored (not passed through).
    """
    from binomialhash import BinomialHash
    bh = BinomialHash()
    rows = _linear_dataset(n=n)
    for i, row in enumerate(rows):
        row["description"] = f"observation_{i}_with_padding_text_to_ensure_payload_exceeds_threshold"
    payload = json.dumps(rows)
    assert len(payload) > 3000, f"Payload only {len(payload)} chars, need >3000"
    summary = bh.ingest(payload, "linear_test")
    return bh


def _get_first_key(bh) -> str:
    """Extract the first stored key from BinomialHash.keys() which returns a list of dicts."""
    key_list = bh.keys()
    assert len(key_list) >= 1, "Expected at least one stored slot"
    return key_list[0]["key"]


class TestBHManifoldInsightsIntegration:
    def test_returns_discovered_law(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_insights(key, json.dumps({"target": "y"}))
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert "discovered_law" in result
        assert result["discovered_law"]["driver"] == "x"
        assert result["discovered_law"]["r2"] > 0.99

    def test_key_and_label_attached(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_insights(key, json.dumps({"target": "y"}))
        assert result.get("key") == key
        assert "label" in result

    def test_missing_key_error(self):
        bh = _make_bh_with_data()
        result = bh.manifold_insights("nonexistent_key", json.dumps({"target": "y"}))
        assert "error" in result

    def test_bad_json_error(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_insights(key, "not valid json {{{")
        assert "error" in result

    def test_custom_top_k(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_insights(key, json.dumps({"target": "y"}), top_k=2)
        assert len(result["insights"]["surprises"]) == 2

    def test_tracks_context_chars(self):
        bh = _make_bh_with_data()
        before = bh.context_stats()
        key = _get_first_key(bh)
        bh.manifold_insights(key, json.dumps({"target": "y"}))
        after = bh.context_stats()
        assert after["chars_out_to_llm"] > before["chars_out_to_llm"]


class TestBHToExcelBatchIntegration:
    def test_basic_export(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.to_excel_batch(key)
        assert "headers" in result
        assert "values" in result
        assert result["total_exported"] == 60

    def test_column_projection(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.to_excel_batch(key, columns=["x", "y"])
        assert result["headers"] == ["x", "y"]

    def test_missing_key(self):
        bh = _make_bh_with_data()
        result = bh.to_excel_batch("nope")
        assert "error" in result


class TestBHToChunksIntegration:
    def test_returns_one_chunk_per_slot(self):
        bh = _make_bh_with_data()
        chunks = bh.to_chunks()
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "bh_dataset"

    def test_chunk_metadata_matches_slot(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        chunks = bh.to_chunks()
        assert chunks[0]["metadata"]["bh_key"] == key
        assert chunks[0]["metadata"]["row_count"] == 60

    def test_multiple_slots(self):
        from binomialhash import BinomialHash
        bh = BinomialHash()
        rows1 = [{"a": i, "pad": f"padding_text_{i}_to_exceed_ingest_threshold_chars"} for i in range(80)]
        rows2 = [{"b": i, "pad": f"another_padding_string_{i}_for_second_dataset_slot"} for i in range(80)]
        bh.ingest(json.dumps(rows1), "set1")
        bh.ingest(json.dumps(rows2), "set2")
        chunks = bh.to_chunks()
        assert len(chunks) == 2


class TestBHManifoldSliceIntegration:
    def test_slice_with_condition(self):
        bh = _make_bh_with_data(n=80)
        key = _get_first_key(bh)
        result = bh.manifold_slice(key, json.dumps({"column": "x", "op": ">", "value": 40}))
        if "error" not in result:
            assert result["slice_rows"] < 80
            assert result["parent_rows"] == 80

    def test_too_few_rows_error(self):
        bh = _make_bh_with_data(n=60)
        key = _get_first_key(bh)
        result = bh.manifold_slice(key, json.dumps({"column": "x", "op": ">", "value": 999}))
        assert "error" in result

    def test_bad_condition_json(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_slice(key, "not json!!!")
        assert "error" in result

    def test_missing_condition_fields(self):
        bh = _make_bh_with_data()
        key = _get_first_key(bh)
        result = bh.manifold_slice(key, json.dumps({"column": "x"}))
        assert "error" in result


# ---------------------------------------------------------------------------
# 14. Import completeness — all extracted symbols accessible
# ---------------------------------------------------------------------------

class TestImportCompleteness:
    def test_insights_imports(self):
        from binomialhash.insights import (
            build_counterfactual,
            compute_branch_divergence,
            compute_insights,
            compute_regime_boundaries,
            compute_surprises,
            discover_best_driver,
        )

    def test_predicates_imports(self):
        from binomialhash.predicates import filter_rows_by_condition

    def test_exporters_imports(self):
        from binomialhash.exporters import export_excel_batch, slot_to_chunk
        from binomialhash.exporters.chunks import slot_to_chunk as s2c
        from binomialhash.exporters.excel import export_excel_batch as eeb
        assert s2c is slot_to_chunk
        assert eeb is export_excel_batch

    def test_navigation_imports(self):
        from binomialhash.manifold.navigation import find_frontier_edges, gridpoint_record
