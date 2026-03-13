from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .structures import GridPoint, ManifoldAxis


def classify_product_topology(axes: List[ManifoldAxis]) -> Tuple[str, int, bool, int]:
    """Classify surface from the product of per-axis boundary types."""
    types = []
    for axis in axes:
        if axis.wraps and axis.wrap_orientation == -1:
            types.append("X")
        elif axis.wraps and axis.wrap_orientation == 1:
            types.append("C")
        else:
            types.append("I")

    n_c = types.count("C")
    n_x = types.count("X")
    n_i = types.count("I")

    if n_x == 0 and n_c == 0:
        return "bounded_patch", 0, True, 2
    if n_x == 0 and n_c == 1:
        return "cylinder", 0, True, 0
    if n_x == 0 and n_c >= 2:
        return "torus", 1, True, 0
    if n_x >= 1 and n_c == 0 and n_i >= 1:
        return "mobius_band", 1, False, 0
    if n_x >= 1 and n_c >= 1:
        return "klein_bottle", 2, False, 0
    if n_x >= 2 and n_c == 0 and n_i == 0:
        return "klein_bottle", 2, False, 0
    return "bounded_patch", 0, True, 2


def count_edge_components(edges: List[Tuple[int, int]]) -> int:
    """Count connected components in an undirected edge set."""
    if not edges:
        return 0

    adjacency: Dict[int, List[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    seen = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return components


def compute_face_topology_2d(
    grid: Dict[Tuple, GridPoint],
    axes: List[ManifoldAxis],
) -> Optional[Dict[str, Any]]:
    """
    Build square faces on a 2D grid and compute face-based diagnostics.

    Works with any 2-axis grid. Categorical axes use their natural value
    ordering (alphabetical) for face construction; wrap detection still
    only applies to ordered (numeric/temporal) axes.
    """
    if len(axes) != 2:
        return None

    axis0, axis1 = axes
    m, n = axis0.size, axis1.size
    if m < 2 or n < 2:
        return None

    coord_to_idx = {coord: point.index for coord, point in grid.items()}

    cells_u = m if axis0.wraps else m - 1
    cells_v = n if axis1.wraps else n - 1
    possible_faces = max(cells_u, 0) * max(cells_v, 0)
    if possible_faces <= 0:
        return None

    face_count = 0
    edge_face_incidence: Dict[Tuple[int, int], int] = {}
    face_vertices = set()

    values0 = [str(value) for value in axis0.values]
    values1 = [str(value) for value in axis1.values]

    for i in range(cells_u):
        for j in range(cells_v):
            i2 = (i + 1) % m
            j2 = (j + 1) % n

            c00 = (values0[i], values1[j])
            c10 = (values0[i2], values1[j])
            c01 = (values0[i], values1[j2])
            c11 = (values0[i2], values1[j2])

            if (
                c00 not in coord_to_idx
                or c10 not in coord_to_idx
                or c01 not in coord_to_idx
                or c11 not in coord_to_idx
            ):
                continue

            v00 = coord_to_idx[c00]
            v10 = coord_to_idx[c10]
            v01 = coord_to_idx[c01]
            v11 = coord_to_idx[c11]

            face_count += 1
            face_vertices.update((v00, v10, v01, v11))

            square_edges = (
                tuple(sorted((v00, v10))),
                tuple(sorted((v10, v11))),
                tuple(sorted((v11, v01))),
                tuple(sorted((v01, v00))),
            )
            for edge in square_edges:
                edge_face_incidence[edge] = edge_face_incidence.get(edge, 0) + 1

    if face_count == 0:
        return {
            "mode": "graph_heuristic",
            "faces": 0,
            "possible_faces": possible_faces,
            "face_coverage": 0.0,
            "occupancy": len(grid) / max(m * n, 1),
            "face_vertex_count": 0,
            "face_edge_count": 0,
            "boundary_edges": 0,
            "boundary_loops": 0,
            "face_euler_characteristic": None,
            "face_genus_estimate": None,
            "confidence": 0.0,
            "torus_consistent": False,
            "is_manifold": False,
            "nonmanifold_edges": 0,
        }

    edge_count_faces = len(edge_face_incidence)
    vertex_count_faces = len(face_vertices)
    chi_face = vertex_count_faces - edge_count_faces + face_count

    boundary_edges = [edge for edge, count in edge_face_incidence.items() if count == 1]
    boundary_edge_count = len(boundary_edges)
    boundary_loops = count_edge_components(boundary_edges)

    nonmanifold_edges = sum(1 for count in edge_face_incidence.values() if count > 2)
    is_manifold = nonmanifold_edges == 0

    boundary_components = boundary_loops if boundary_edge_count > 0 else 0
    genus_est_orientable = (2 - boundary_components - chi_face) / 2
    crosscap_est_nonorientable = 2 - boundary_components - chi_face

    face_coverage = face_count / max(possible_faces, 1)
    occupancy = len(grid) / max(m * n, 1)
    confidence = max(0.0, min(1.0, 0.55 * face_coverage + 0.45 * occupancy))
    if not is_manifold:
        confidence *= 0.5
    torus_consistent = (
        boundary_edge_count == 0
        and chi_face == 0
        and abs(genus_est_orientable - 1.0) < 1e-9
    )

    return {
        "mode": "validated_2d_faces" if is_manifold else "nonmanifold_complex",
        "faces": face_count,
        "possible_faces": possible_faces,
        "face_coverage": round(face_coverage, 4),
        "occupancy": round(occupancy, 4),
        "face_vertex_count": vertex_count_faces,
        "face_edge_count": edge_count_faces,
        "boundary_edges": boundary_edge_count,
        "boundary_loops": boundary_loops,
        "face_euler_characteristic": chi_face,
        "face_genus_estimate_orientable": round(genus_est_orientable, 6),
        "face_crosscap_estimate_nonorientable": crosscap_est_nonorientable,
        "confidence": round(confidence, 4),
        "torus_consistent": torus_consistent,
        "is_manifold": is_manifold,
        "nonmanifold_edges": nonmanifold_edges,
    }
