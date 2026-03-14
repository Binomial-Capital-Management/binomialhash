"""
ManifoldSurface builds an operational structure over tabular data.

Axes are inferred from discovered columns and values, then projected into a
coordinate grid with adjacency, field summaries, and face-complex diagnostics
where the grid supports them. Boundary behavior is inferred heuristically and
used for product-topology labels plus gated surface naming; it is not a proof
of underlying manifold truth.

The module also computes graph and local-field diagnostics such as
Forman-Ricci curvature, Morse-like critical points, persistence-style features,
interaction curvature, and navigation helpers.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from .axes import identify_axes
from .diagnostics import (
    check_boundary_wrap,
    classify_morse_points,
    compute_betti_numbers,
    compute_field_curvature,
    compute_forman_ricci,
    compute_interaction_curvature,
    compute_persistence,
)
from .grid import build_adjacency, build_grid
from .navigation import (
    basin,
    basin_of,
    bfs_distances,
    controlled_walk,
    coverage_audit,
    geodesic_path,
    idx_to_point,
    multiscale_view,
    navigate,
    orbit,
    resolve_coord,
    summarize_points,
    trace_extremum,
    wrap_audit,
)
from .structures import CriticalPoint, GridPoint, ManifoldAxis, ManifoldSurface
from .topology import classify_product_topology, compute_face_topology_2d

logger = logging.getLogger(__name__)


def _at(self: ManifoldSurface, coord: Tuple) -> Optional[GridPoint]:
    return self.grid.get(coord)


def _at_index(self: ManifoldSurface, index: int) -> Optional[GridPoint]:
    for point in self.grid.values():
        if point.index == index:
            return point
    return None


def _highest_curvature(self: ManifoldSurface, top_k: int = 5) -> List[GridPoint]:
    return sorted(self.grid.values(), key=lambda point: point.curvature, reverse=True)[:top_k]


def _lowest_density(self: ManifoldSurface, top_k: int = 5) -> List[GridPoint]:
    return sorted(self.grid.values(), key=lambda point: point.density)[:top_k]


def _most_negative_ricci(self: ManifoldSurface, top_k: int = 5) -> List[GridPoint]:
    return sorted(self.grid.values(), key=lambda point: point.forman_ricci)[:top_k]


def _critical_points_for(self: ManifoldSurface, field_name: str) -> Dict[str, List[GridPoint]]:
    result: Dict[str, List[GridPoint]] = {"minimum": [], "saddle": [], "maximum": []}
    for point in self.grid.values():
        morse_type = point.morse_type.get(field_name)
        if morse_type and morse_type in result:
            result[morse_type].append(point)
    return result


def _to_summary(self: ManifoldSurface) -> Dict[str, Any]:
    peaks = self.highest_curvature(3)
    sparse = self.lowest_density(3)
    bridges = self.most_negative_ricci(3)

    morse_summary: Dict[str, Dict[str, int]] = {}
    for field_name in self.field_columns[:5]:
        counts = {"minimum": 0, "saddle": 0, "maximum": 0, "regular": 0}
        for point in self.grid.values():
            morse_type = point.morse_type.get(field_name)
            if morse_type and morse_type in counts:
                counts[morse_type] += 1
        if any(value > 0 for key, value in counts.items() if key != "regular"):
            morse_summary[field_name] = {key: value for key, value in counts.items() if value > 0}

    persistence_summary: Dict[str, Any] = {}
    for field_name, pairs in self.persistence_pairs.items():
        persistent = [point for point in pairs if point.persistence > 0]
        if persistent:
            persistence_summary[field_name] = {
                "total_features": len(persistent),
                "top_3": [
                    {"coord": point.coord, "value": point.value, "persistence": point.persistence}
                    for point in persistent[:3]
                ],
                "noise_threshold": (
                    round(persistent[len(persistent) // 2].persistence, 6)
                    if len(persistent) > 1
                    else 0
                ),
            }

    return {
        "surface": self.surface_name,
        "genus": self.genus,
        "orientable": self.orientable,
        "euler_characteristic": self.euler_characteristic,
        "classification_mode": self.classification_mode,
        "betti": {"β₀": self.betti_0, "β₁": self.betti_1},
        "axes": [
            {
                "column": axis.column,
                "type": axis.axis_type,
                "size": axis.size,
                "wraps": axis.wraps,
                "wrap_orientation": (
                    "preserving"
                    if axis.wrap_orientation == 1
                    else "reversing"
                    if axis.wrap_orientation == -1
                    else "none"
                ),
            }
            for axis in self.axes
        ],
        "field_columns": self.field_columns,
        "product_topology_label": self.product_topology_label,
        "graph": {"V": self.vertex_count, "E": self.edge_count, "β₀": self.betti_0, "β₁": self.betti_1},
        "face_complex": {
            "V2": self.face_vertex_count,
            "E2": self.face_edge_count,
            "F": self.face_count,
            "χ": self.face_euler_characteristic,
            "boundary_edges": self.boundary_edge_count,
            "boundary_loops": self.boundary_loop_count,
            "genus_estimate": self.face_genus_estimate if self.orientable else None,
            "crosscap_estimate": self.face_crosscap_estimate if not self.orientable else None,
            "is_manifold": self.is_manifold,
            "nonmanifold_edges": self.nonmanifold_edges,
            "face_coverage": self.face_coverage,
            "occupancy": self.occupancy,
        },
        "surface_confidence": self.surface_confidence_obj,
        "critical_points": morse_summary,
        "persistence": persistence_summary,
        "interactions": self.interactions[:5],
        "curvature_peaks": [
            {
                "coord": point.axis_coords,
                "curvature": round(point.curvature, 6),
                "forman_ricci": round(point.forman_ricci, 4),
                "density": point.density,
            }
            for point in peaks
        ],
        "bridges": [
            {
                "coord": point.axis_coords,
                "forman_ricci": round(point.forman_ricci, 4),
                "curvature": round(point.curvature, 6),
                "density": point.density,
            }
            for point in bridges
        ],
        "sparse_regions": [
            {"coord": point.axis_coords, "density": point.density, "curvature": round(point.curvature, 6)}
            for point in sparse
        ],
    }


# Monkey-patched to avoid circular imports: ManifoldSurface lives in structures.py but methods depend on navigation, spatial, etc.
ManifoldSurface.at = _at
ManifoldSurface.at_index = _at_index
ManifoldSurface.highest_curvature = _highest_curvature
ManifoldSurface.lowest_density = _lowest_density
ManifoldSurface.most_negative_ricci = _most_negative_ricci
ManifoldSurface.critical_points_for = _critical_points_for
ManifoldSurface._basin_of = basin_of
ManifoldSurface.to_summary = _to_summary
ManifoldSurface.geodesic_path = geodesic_path
ManifoldSurface.controlled_walk = controlled_walk
ManifoldSurface._idx_to_point = idx_to_point
ManifoldSurface._resolve_coord = resolve_coord
ManifoldSurface._bfs_distances = bfs_distances
ManifoldSurface._summarize_points = summarize_points
ManifoldSurface.orbit = orbit
ManifoldSurface.multiscale_view = multiscale_view
ManifoldSurface.basin = basin
ManifoldSurface.trace_extremum = trace_extremum
ManifoldSurface.coverage_audit = coverage_audit
ManifoldSurface.wrap_audit = wrap_audit
ManifoldSurface.navigate = navigate

from .spatial import (
    diffusion_distance as _diffusion_distance,
    heat_kernel as _heat_kernel,
    laplacian_spectrum as _laplacian_spectrum,
    reeb_graph as _reeb_graph,
    scalar_harmonics as _scalar_harmonics,
    vector_field_analysis as _vector_field_analysis,
)
ManifoldSurface.heat_kernel = lambda self, **kw: _heat_kernel(self, **kw)
ManifoldSurface.reeb_graph = lambda self, target_field, **kw: _reeb_graph(self, target_field, **kw)
ManifoldSurface.vector_field = lambda self, target_field, **kw: _vector_field_analysis(self, target_field, **kw)
ManifoldSurface.laplacian_spectrum = lambda self, **kw: _laplacian_spectrum(self, **kw)
ManifoldSurface.scalar_harmonics = lambda self, target_field, **kw: _scalar_harmonics(self, target_field, **kw)
ManifoldSurface.diffusion_distance = lambda self, **kw: _diffusion_distance(self, **kw)

# ═══════════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_manifold(
    rows: List[Dict[str, Any]],
    columns: List[str],
    col_types: Dict[str, str],
    col_stats: Dict[str, Dict[str, Any]],
) -> Optional[ManifoldSurface]:
    """Build the manifold surface from BH slot data."""
    if len(rows) < 10:
        return None

    axes, fields = identify_axes(rows, columns, col_types, col_stats)
    if len(axes) < 1 or not fields:
        logger.info("[Manifold] Not enough axes (%d) or fields (%d) to build surface", len(axes), len(fields))
        return None

    grid = build_grid(rows, axes, fields)
    if len(grid) < 4:
        logger.info("[Manifold] Grid too small (%d points)", len(grid))
        return None

    handles = 0
    crosscaps = 0
    for i, ax in enumerate(axes):
        if ax.ordered and ax.size >= 4:
            wraps, orientation = check_boundary_wrap(grid, axes, fields, i)
            ax.wraps = wraps
            ax.wrap_orientation = orientation
            # orientation=1: preserving (circle/torus); orientation=-1: reversing (Möbius/Klein).
            if wraps:
                if orientation == 1:
                    handles += 1
                elif orientation == -1:
                    crosscaps += 1

    build_adjacency(grid, axes)

    compute_field_curvature(grid, fields)
    compute_forman_ricci(grid)
    classify_morse_points(grid, fields)

    betti_0, betti_1 = compute_betti_numbers(grid)

    persistence_pairs: Dict[str, List[CriticalPoint]] = {}
    for f in fields[:5]:
        pairs = compute_persistence(grid, f)
        if pairs:
            persistence_pairs[f] = pairs[:20]

    interactions = compute_interaction_curvature(grid, fields)

    product_label, genus, orientable, chi = classify_product_topology(axes)
    face_stats = compute_face_topology_2d(grid, axes)

    classification_mode = "graph_heuristic"
    face_count = 0
    face_coverage = 0.0
    occupancy = 0.0
    face_vertex_count = 0
    face_edge_count = 0
    boundary_edge_count = 0
    boundary_loop_count = 0
    face_euler_characteristic: Optional[int] = None
    face_genus_estimate: Optional[float] = None
    face_crosscap_estimate: Optional[int] = None
    is_manifold = False
    nonmanifold_edges = 0
    confidence_val = 0.0

    if face_stats is not None:
        classification_mode = face_stats.get("mode", "graph_heuristic")
        confidence_val = float(face_stats.get("confidence", 0.0))
        face_count = int(face_stats.get("faces", 0))
        face_coverage = float(face_stats.get("face_coverage", 0.0))
        occupancy = float(face_stats.get("occupancy", 0.0))
        face_vertex_count = int(face_stats.get("face_vertex_count", 0))
        face_edge_count = int(face_stats.get("face_edge_count", 0))
        boundary_edge_count = int(face_stats.get("boundary_edges", 0))
        boundary_loop_count = int(face_stats.get("boundary_loops", 0))
        face_euler_characteristic = face_stats.get("face_euler_characteristic")
        if orientable:
            face_genus_estimate = face_stats.get("face_genus_estimate_orientable")
        else:
            face_crosscap_estimate = face_stats.get("face_crosscap_estimate_nonorientable")
        is_manifold = bool(face_stats.get("is_manifold", False))
        nonmanifold_edges = int(face_stats.get("nonmanifold_edges", 0))

    # Three-tier fallback: graph_only → nonmanifold_complex → canonical product name.
    if face_count == 0:
        surface_name = "graph_only"
    elif not is_manifold:
        surface_name = "nonmanifold_complex"
    else:
        surface_name = product_label

    surface_confidence_obj = {
        "has_faces": face_count > 0,
        "is_edge_manifold": is_manifold,
        "is_edge_manifold_note": "Edge incidence check only; vertex-link manifoldness not validated.",
        "nonmanifold_edges": nonmanifold_edges,
        "face_coverage": face_coverage,
        "occupancy": occupancy,
        "boundary_wrap_method": "mean_field_similarity",
        "classification_mode": classification_mode,
        "confidence": confidence_val,
    }

    edge_count = sum(len(gp.neighbors) for gp in grid.values()) // 2

    logger.info(
        "[Manifold] Built %s (product=%s): genus=%d orientable=%s χ=%d β₀=%d β₁=%d | "
        "%d axes, %d fields, %d vertices, %d graph_edges | "
        "face_complex: V2=%d E2=%d F=%d manifold=%s | cov=%.2f conf=%.2f mode=%s | "
        "%d interactions | wraps: %s",
        surface_name, product_label, genus, orientable, chi, betti_0, betti_1,
        len(axes), len(fields), len(grid), edge_count,
        face_vertex_count, face_edge_count, face_count, is_manifold,
        face_coverage, confidence_val, classification_mode,
        len(interactions),
        [(a.column, a.wraps, a.wrap_orientation) for a in axes],
    )

    return ManifoldSurface(
        axes=axes, field_columns=fields, grid=grid,
        surface_name=surface_name, genus=genus, orientable=orientable,
        euler_characteristic=chi, handles=handles, crosscaps=crosscaps,
        vertex_count=len(grid), edge_count=edge_count,
        betti_0=betti_0, betti_1=betti_1,
        persistence_pairs=persistence_pairs, interactions=interactions,
        classification_mode=classification_mode,
        product_topology_label=product_label,
        surface_confidence_obj=surface_confidence_obj,
        face_count=face_count,
        face_coverage=face_coverage,
        occupancy=occupancy,
        face_vertex_count=face_vertex_count,
        face_edge_count=face_edge_count,
        boundary_edge_count=boundary_edge_count,
        boundary_loop_count=boundary_loop_count,
        face_euler_characteristic=face_euler_characteristic,
        face_genus_estimate=face_genus_estimate,
        face_crosscap_estimate=face_crosscap_estimate,
        is_manifold=is_manifold,
        nonmanifold_edges=nonmanifold_edges,
    )
