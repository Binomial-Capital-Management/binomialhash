"""Manifold navigation, inspection, and spatial reasoning tool specs."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .base import ToolSpec, _prop

if TYPE_CHECKING:
    from ..core import BinomialHash


def _make_manifold_specs(bh: "BinomialHash") -> List[ToolSpec]:
    return [
        ToolSpec(
            name="bh_manifold_state",
            description=(
                "Build deterministic manifold state for a stored dataset.  "
                "Returns discovered dimensions, dependency graph preview, and "
                "categorical complexity.  Call before objective-driven insights."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                },
                "required": ["key"],
            },
            handler=lambda key: bh.manifold_state(key),
            group="manifold",
        ),
        ToolSpec(
            name="bh_manifold_insights",
            description=(
                "Objective-driven manifold insights.  Returns surprises "
                "(strongest contradictions to discovered law), regime boundaries "
                "(sharp transitions), branch divergence (same target, different "
                "context), and counterfactual direction suggestion."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "objective_json": _prop(
                        "string",
                        "JSON objective, e.g. '{\"target\":\"return_1d\",\"goal\":\"maximize\",\"target_value\":0.02}'.",
                    ),
                    "top_k": _prop("integer", "Max entries per insight block.", default=5),
                },
                "required": ["key", "objective_json"],
            },
            handler=lambda key, objective_json, top_k=5: (
                bh.manifold_insights(key, objective_json, top_k)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_navigate",
            description=(
                "Navigate the manifold at a specific coordinate.  Returns "
                "position (field values, curvature, density), topology summary, "
                "gradients toward neighbors, and steepest move direction."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "coord_json": _prop(
                        "string",
                        "JSON dict of axis values, e.g. '{\"strike\":\"200\",\"type\":\"call\"}'.",
                    ),
                    "target_field": _prop("string", "Field column to track gradients for."),
                },
                "required": ["key", "coord_json"],
            },
            handler=lambda key, coord_json, target_field=None: (
                bh.manifold_navigate(key, coord_json, target_field)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_manifold_slice",
            description=(
                "Build a conditional sub-manifold from filtered rows.  "
                "Answers: 'what does the shape look like WHEN this holds?'  "
                "Changes in topology or critical points indicate a structural "
                "regime shift."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "condition_json": _prop(
                        "string",
                        "JSON filter, e.g. '{\"column\":\"vol\",\"op\":\">\",\"value\":0.035}'.  "
                        "Ops: >, >=, <, <=, =, !=.",
                    ),
                    "target_field": _prop("string", "Optional field for persistence in the slice."),
                },
                "required": ["key", "condition_json"],
            },
            handler=lambda key, condition_json, target_field=None: (
                bh.manifold_slice(key, condition_json, target_field)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_geodesic",
            description=(
                "Shortest path between two manifold points.  Edge weight = "
                "|Δ target_field| if provided, else hop count.  Returns "
                "waypoints with field values and deltas at each step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "start_json": _prop("string", "Starting coordinate as JSON dict."),
                    "end_json": _prop("string", "Ending coordinate as JSON dict."),
                    "target_field": _prop("string", "Field to weight edges by (optional)."),
                },
                "required": ["key", "start_json", "end_json"],
            },
            handler=lambda key, start_json, end_json, target_field=None: (
                bh.geodesic(key, start_json, end_json, target_field)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_walk_axis",
            description=(
                "Sweep one manifold axis while averaging target across all "
                "others.  Returns target mean/min/max at each axis value plus "
                "total sensitivity.  Use to scan which axes matter most."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "walk_axis": _prop("string", "Axis name to walk along."),
                    "target_field": _prop("string", "Numeric field to track."),
                },
                "required": ["key", "walk_axis", "target_field"],
            },
            handler=lambda key, walk_axis, target_field: (
                bh.controlled_walk(key, walk_axis, target_field)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_orbit",
            description=(
                "Survey a neighborhood around a center point at a given "
                "graph-hop radius.  Like a satellite pass."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "center_json": _prop("string", "Center coordinate as JSON dict."),
                    "radius": _prop("integer", "Graph-hop radius."),
                    "target_field": _prop("string", "Field to track."),
                    "resolution": _prop("integer", "Max points per ring.", default=16),
                    "mode": _prop("string", "Survey mode.", default="ring", enum=["ring", "disk"]),
                },
                "required": ["key", "center_json", "radius"],
            },
            handler=lambda key, center_json, radius, target_field=None, resolution=16, mode="ring": (
                bh.orbit(key, center_json, radius, target_field, resolution, mode)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_multiscale_view",
            description=(
                "Inspect the same center at multiple neighborhood radii.  "
                "Reveals how local vs global structure differs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "center_json": _prop("string", "Center coordinate as JSON dict."),
                    "radii_json": _prop("string", "JSON list of radii, e.g. '[1,2,4]'."),
                    "target_field": _prop("string", "Field to track."),
                    "resolution": _prop("integer", "Max points per ring.", default=16),
                },
                "required": ["key", "center_json", "radii_json"],
            },
            handler=lambda key, center_json, radii_json, target_field=None, resolution=16: (
                bh.multiscale_view(key, center_json, radii_json, target_field, resolution)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_frontier",
            description=(
                "Trace the structural frontier between two predicate-defined "
                "regions on the manifold grid."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "condition_a_json": _prop("string", "JSON predicate for region A."),
                    "condition_b_json": _prop("string", "JSON predicate for region B."),
                    "target_field": _prop("string", "Field to measure jumps across frontier."),
                    "limit": _prop("integer", "Max frontier edges.", default=200),
                },
                "required": ["key", "condition_a_json", "condition_b_json"],
            },
            handler=lambda key, condition_a_json, condition_b_json, target_field=None, limit=200: (
                bh.frontier(key, condition_a_json, condition_b_json, target_field, limit)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_basin",
            description=(
                "Find the basin of attraction around a seed point by following "
                "gradient descent (or ascent) on a target field."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "seed_json": _prop("string", "Seed coordinate as JSON dict."),
                    "target_field": _prop("string", "Field to follow."),
                    "direction": _prop("string", "Descent or ascent.", default="descend", enum=["descend", "ascend"]),
                },
                "required": ["key", "seed_json", "target_field"],
            },
            handler=lambda key, seed_json, target_field, direction="descend": (
                bh.basin(key, seed_json, target_field, direction)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_ridge_trace",
            description="Trace a ridge (high-value corridor) on the target field from a seed point.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "seed_json": _prop("string", "Seed coordinate as JSON dict."),
                    "target_field": _prop("string", "Field to trace."),
                    "max_steps": _prop("integer", "Max trace steps.", default=25),
                },
                "required": ["key", "seed_json", "target_field"],
            },
            handler=lambda key, seed_json, target_field, max_steps=25: (
                bh.ridge_trace(key, seed_json, target_field, max_steps)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_valley_trace",
            description="Trace a valley (low-value channel) on the target field from a seed point.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "seed_json": _prop("string", "Seed coordinate as JSON dict."),
                    "target_field": _prop("string", "Field to trace."),
                    "max_steps": _prop("integer", "Max trace steps.", default=25),
                },
                "required": ["key", "seed_json", "target_field"],
            },
            handler=lambda key, seed_json, target_field, max_steps=25: (
                bh.valley_trace(key, seed_json, target_field, max_steps)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_coverage_audit",
            description=(
                "Audit graph/face support quality, sparsity, occupancy, and "
                "manifold classification confidence."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                },
                "required": ["key"],
            },
            handler=lambda key: bh.coverage_audit(key),
            group="manifold",
        ),
        ToolSpec(
            name="bh_wrap_audit",
            description=(
                "Explain boundary-wrap inference evidence and confidence "
                "for one axis or all axes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "axis": _prop("string", "Axis name (omit for all axes)."),
                },
                "required": ["key"],
            },
            handler=lambda key, axis=None: bh.wrap_audit(key, axis),
            group="manifold",
        ),
        ToolSpec(
            name="bh_heat_kernel",
            description=(
                "Multi-scale heat kernel signatures and geometric bottleneck "
                "detection.  Reveals where the manifold has narrow bridges "
                "connecting broader regions — bottlenecks invisible to local "
                "navigation.  Also clusters points by shape similarity."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target_field": _prop("string", "Optional field for bottleneck context."),
                    "n_eigen": _prop("integer", "Number of Laplacian eigenmodes (default 20).", default=20),
                    "top_k": _prop("integer", "Top bottleneck points to return.", default=10),
                },
                "required": ["key"],
            },
            handler=lambda key, target_field=None, n_eigen=20, top_k=10: (
                bh.heat_kernel(key, target_field, n_eigen, top_k)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_reeb_graph",
            description=(
                "Topological skeleton of a scalar field on the manifold.  "
                "Sweeps through the field's level sets and tracks how connected "
                "components birth, merge, split, and die.  Compresses the "
                "manifold into a graph that reveals qualitative regime structure: "
                "'at low revenue there's one regime, at medium it splits into two "
                "distinct strategies, at high they merge back.'"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target_field": _prop("string", "Scalar field to build skeleton for."),
                    "n_levels": _prop("integer", "Number of level-set slices.", default=20),
                },
                "required": ["key", "target_field"],
            },
            handler=lambda key, target_field, n_levels=20: (
                bh.reeb_graph(key, target_field, n_levels)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_vector_field",
            description=(
                "Gradient flow analysis: divergence (sources/sinks), curl "
                "(rotational structure), and fixed-point classification.  "
                "Sources are where the field is being 'created'; sinks where it "
                "is 'destroyed'; vortices indicate cyclical dynamics (e.g. "
                "returns -> volatility -> hedging cost -> returns)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target_field": _prop("string", "Field whose gradient to analyze."),
                    "top_k": _prop("integer", "Max points per category.", default=10),
                },
                "required": ["key", "target_field"],
            },
            handler=lambda key, target_field, top_k=10: (
                bh.vector_field(key, target_field, top_k)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_laplacian_spectrum",
            description=(
                "Graph Laplacian eigenvalues and manifold-native spectral "
                "clustering.  Unlike feature-space clustering, this respects "
                "the manifold's adjacency: two points with similar features "
                "but separated by a ridge end up in different clusters.  "
                "The spectral gap measures global connectivity; the Fiedler "
                "vector gives the optimal bisection."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "n_eigen": _prop("integer", "Number of eigenvalues.", default=15),
                    "n_clusters": _prop("integer", "Force cluster count (auto if omitted)."),
                },
                "required": ["key"],
            },
            handler=lambda key, n_eigen=15, n_clusters=None: (
                bh.laplacian_spectrum(key, n_eigen, n_clusters)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_scalar_harmonics",
            description=(
                "Decompose a scalar field into the manifold's natural modes "
                "(Laplacian eigenbasis).  Returns smooth global trend, "
                "structural residual (where data deviates from the trend), "
                "and energy spectrum (how much variance each mode captures).  "
                "Anomalies = points where the residual is largest."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target_field": _prop("string", "Field to decompose."),
                    "n_modes": _prop("integer", "Number of harmonic modes.", default=10),
                    "top_k": _prop("integer", "Top anomaly points.", default=10),
                },
                "required": ["key", "target_field"],
            },
            handler=lambda key, target_field, n_modes=10, top_k=10: (
                bh.scalar_harmonics(key, target_field, n_modes, top_k)
            ),
            group="manifold",
        ),
        ToolSpec(
            name="bh_diffusion_distance",
            description=(
                "Robust multi-path distance between landmarks on the manifold.  "
                "Unlike geodesic (shortest single path), diffusion distance "
                "averages over ALL paths: two points connected by many redundant "
                "routes are 'closer' than two connected by one narrow corridor.  "
                "Reveals connectivity robustness and true geometric proximity."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "landmarks_json": _prop(
                        "string",
                        "Optional JSON array of landmark coords, e.g. "
                        "'[{\"strike\":\"200\",\"dte\":\"30\"}, ...]'. "
                        "Auto-selected if omitted.",
                    ),
                    "time_param": _prop("number", "Diffusion time scale (default 1.0).", default=1.0),
                    "n_landmarks": _prop("integer", "Number of auto-landmarks.", default=8),
                },
                "required": ["key"],
            },
            handler=lambda key, landmarks_json=None, time_param=1.0, n_landmarks=8: (
                bh.diffusion_distance(key, landmarks_json, time_param, n_landmarks)
            ),
            group="manifold",
        ),
    ]
