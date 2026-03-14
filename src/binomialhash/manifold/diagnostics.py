from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

from .structures import CriticalPoint, GridPoint, ManifoldAxis


def boundary_wrap_diagnostics(
    grid: Dict[Tuple, GridPoint],
    axes: List[ManifoldAxis],
    fields: List[str],
    axis_index: int,
) -> Dict[str, Any]:
    """Return boundary-wrap diagnostics for one axis."""
    axis = axes[axis_index]
    if not axis.ordered or axis.size < 4:
        return {
            "axis": axis.column,
            "ordered": axis.ordered,
            "wraps": False,
            "orientation": 0,
            "orientation_label": "none",
            "avg_similarity": 0.0,
            "sign_matches": 0,
            "sign_flips": 0,
            "low_points": 0,
            "high_points": 0,
            "method": "mean_field_similarity",
            "reason": "axis not ordered or too small",
        }

    value_list = [str(value) for value in axis.values]
    low_values = set(value_list[: max(1, axis.size // 5)])
    high_values = set(value_list[-(max(1, axis.size // 5)) :])

    low_points: List[Dict[str, float]] = []
    high_points: List[Dict[str, float]] = []

    for coord, point in grid.items():
        axis_value = coord[axis_index]
        if axis_value in low_values:
            low_points.append(point.field_values)
        elif axis_value in high_values:
            high_points.append(point.field_values)

    if len(low_points) < 3 or len(high_points) < 3:
        return {
            "axis": axis.column,
            "ordered": axis.ordered,
            "wraps": False,
            "orientation": 0,
            "orientation_label": "none",
            "avg_similarity": 0.0,
            "sign_matches": 0,
            "sign_flips": 0,
            "low_points": len(low_points),
            "high_points": len(high_points),
            "method": "mean_field_similarity",
            "reason": "insufficient boundary support",
        }

    def mean_vector(points: List[Dict[str, float]]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for field_name in fields:
            values = [point.get(field_name, 0.0) for point in points]
            result[field_name] = sum(values) / len(values) if values else 0.0
        return result

    low_mean = mean_vector(low_points)
    high_mean = mean_vector(high_points)

    similarities = []
    sign_matches = 0
    sign_flips = 0
    for field_name in fields:
        low_value = low_mean.get(field_name, 0.0)
        high_value = high_mean.get(field_name, 0.0)
        magnitude_sum = abs(low_value) + abs(high_value)
        if magnitude_sum < 1e-9:
            continue
        magnitude_diff = abs(abs(low_value) - abs(high_value))
        similarities.append(1.0 - min(magnitude_diff / magnitude_sum, 1.0))
        if abs(low_value) > 1e-9 and abs(high_value) > 1e-9:
            if (low_value > 0) == (high_value > 0):
                sign_matches += 1
            else:
                sign_flips += 1

    if not similarities:
        return {
            "axis": axis.column,
            "ordered": axis.ordered,
            "wraps": False,
            "orientation": 0,
            "orientation_label": "none",
            "avg_similarity": 0.0,
            "sign_matches": sign_matches,
            "sign_flips": sign_flips,
            "low_points": len(low_points),
            "high_points": len(high_points),
            "method": "mean_field_similarity",
            "reason": "no informative fields",
        }

    avg_similarity = sum(similarities) / len(similarities)
    # Empirically tuned threshold for wrap detection.
    wraps = avg_similarity >= 0.6
    if not wraps:
        orientation = 0
    elif sign_flips > sign_matches:
        orientation = -1
    else:
        orientation = 1

    return {
        "axis": axis.column,
        "ordered": axis.ordered,
        "wraps": wraps,
        "orientation": orientation,
        "orientation_label": (
            "preserving" if orientation == 1 else "reversing" if orientation == -1 else "none"
        ),
        "avg_similarity": round(avg_similarity, 6),
        "sign_matches": sign_matches,
        "sign_flips": sign_flips,
        "low_points": len(low_points),
        "high_points": len(high_points),
        "method": "mean_field_similarity",
        "threshold": 0.6,
        "reason": (
            "avg similarity above threshold" if wraps else "avg similarity below threshold"
        ),
    }


def check_boundary_wrap(
    grid: Dict[Tuple, GridPoint],
    axes: List[ManifoldAxis],
    fields: List[str],
    axis_index: int,
) -> Tuple[bool, int]:
    """Check if an axis wraps using the boundary-wrap diagnostics."""
    diagnostics = boundary_wrap_diagnostics(grid, axes, fields, axis_index)
    return bool(diagnostics["wraps"]), int(diagnostics["orientation"])


def compute_field_curvature(grid: Dict[Tuple, GridPoint], fields: List[str]) -> None:
    """Approximate curvature at each point from field variation."""
    idx_to_point = {point.index: point for point in grid.values()}
    for point in grid.values():
        if not point.neighbors or not point.field_values:
            continue
        total_curvature = 0.0
        count = 0
        for field_name in fields:
            my_value = point.field_values.get(field_name)
            if my_value is None:
                continue
            neighbor_values = []
            for neighbor_idx in point.neighbors:
                neighbor = idx_to_point.get(neighbor_idx)
                if neighbor and field_name in neighbor.field_values:
                    neighbor_values.append(neighbor.field_values[field_name])
            if not neighbor_values:
                continue
            neighbor_mean = sum(neighbor_values) / len(neighbor_values)
            scale = max(abs(neighbor_mean), 1.0)
            # Discrete Laplacian-style: curvature = |value - neighbor_mean| / scale.
            total_curvature += abs(my_value - neighbor_mean) / scale
            count += 1
        point.curvature = total_curvature / count if count else 0.0


def compute_forman_ricci(grid: Dict[Tuple, GridPoint]) -> None:
    """Compute Forman-Ricci curvature at each vertex."""
    idx_to_point = {point.index: point for point in grid.values()}
    for point in grid.values():
        if not point.neighbors:
            point.forman_ricci = 0.0
            continue
        degree = len(point.neighbors)
        edge_curvatures = []
        for neighbor_idx in point.neighbors:
            neighbor = idx_to_point.get(neighbor_idx)
            if neighbor is None:
                continue
            # Forman-Ricci: 4 - deg(v) - deg(w) per edge (v,w).
            edge_curvatures.append(4.0 - degree - len(neighbor.neighbors))
        point.forman_ricci = (
            sum(edge_curvatures) / len(edge_curvatures) if edge_curvatures else 0.0
        )


def classify_morse_points(grid: Dict[Tuple, GridPoint], fields: List[str]) -> None:
    """Classify each grid point as minimum, saddle, maximum, or regular per field."""
    idx_to_point = {point.index: point for point in grid.values()}
    for point in grid.values():
        if not point.neighbors:
            continue
        for field_name in fields:
            my_value = point.field_values.get(field_name)
            if my_value is None:
                continue
            above = 0
            below = 0
            for neighbor_idx in point.neighbors:
                neighbor = idx_to_point.get(neighbor_idx)
                if neighbor is None:
                    continue
                neighbor_value = neighbor.field_values.get(field_name)
                if neighbor_value is None:
                    continue
                # Avoid floating-point equality issues when classifying critical points.
                if neighbor_value > my_value + 1e-12:
                    above += 1
                elif neighbor_value < my_value - 1e-12:
                    below += 1
            if above + below == 0:
                continue
            if below == 0:
                point.morse_type[field_name] = "minimum"
            elif above == 0:
                point.morse_type[field_name] = "maximum"
            elif above >= 2 and below >= 2:
                point.morse_type[field_name] = "saddle"
            else:
                point.morse_type[field_name] = "regular"


def compute_persistence(
    grid: Dict[Tuple, GridPoint],
    target_field: str,
) -> List[CriticalPoint]:
    """Compute 0-dimensional persistence via sublevel-set filtration."""
    vertices = []
    for point in grid.values():
        value = point.field_values.get(target_field)
        if value is not None:
            vertices.append((value, point.index, point.axis_coords))
    vertices.sort()

    if len(vertices) < 2:
        return []

    idx_to_point = {point.index: point for point in grid.values()}
    # 0-dimensional persistent homology: track births during sublevel-set filtration, record persistence when components merge.
    parent: Dict[int, int] = {}
    rank: Dict[int, int] = {}
    comp_birth: Dict[int, float] = {}
    comp_birth_coord: Dict[int, Tuple] = {}

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    pairs: List[CriticalPoint] = []

    for value, index, coord in vertices:
        parent[index] = index
        rank[index] = 0
        comp_birth[index] = value
        comp_birth_coord[index] = coord

        point = idx_to_point.get(index)
        if point is None:
            continue

        for neighbor_idx in point.neighbors:
            if neighbor_idx not in parent:
                continue
            root_i = find(index)
            root_n = find(neighbor_idx)
            if root_i == root_n:
                continue

            # Elder rule: older component absorbs younger; younger's birth-death gap is its persistence.
            if comp_birth[root_i] <= comp_birth[root_n]:
                elder, younger = root_i, root_n
            else:
                elder, younger = root_n, root_i

            persistence = value - comp_birth[younger]
            if persistence > 1e-12:
                pairs.append(
                    CriticalPoint(
                        vertex_index=younger,
                        coord=comp_birth_coord[younger],
                        morse_type="minimum",
                        field=target_field,
                        value=round(comp_birth[younger], 6),
                        persistence=round(persistence, 6),
                    )
                )

            if rank[elder] < rank[younger]:
                parent[elder] = younger
                comp_birth[younger] = comp_birth[elder]
                comp_birth_coord[younger] = comp_birth_coord[elder]
            else:
                parent[younger] = elder
                if rank[elder] == rank[younger]:
                    rank[elder] += 1

    pairs.sort(key=lambda point: point.persistence, reverse=True)
    return pairs


def compute_betti_numbers(grid: Dict[Tuple, GridPoint]) -> Tuple[int, int]:
    """Compute β₀ and β₁ for the grid graph."""
    if not grid:
        return 0, 0

    idx_to_point = {point.index: point for point in grid.values()}
    visited = set()
    components = 0

    for point in grid.values():
        if point.index in visited:
            continue
        components += 1
        queue = deque([point.index])
        visited.add(point.index)
        while queue:
            current = queue.popleft()
            current_point = idx_to_point.get(current)
            if current_point is None:
                continue
            for neighbor_idx in current_point.neighbors:
                if neighbor_idx not in visited and neighbor_idx in idx_to_point:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

    vertex_count = len(grid)
    edge_count = sum(len(point.neighbors) for point in grid.values()) // 2
    beta_0 = components
    # Euler relation χ = V - E + β₀ ⟹ β₁ = E - V + β₀.
    beta_1 = max(0, edge_count - vertex_count + beta_0)
    return beta_0, beta_1


def compute_interaction_curvature(
    grid: Dict[Tuple, GridPoint],
    fields: List[str],
) -> List[Dict[str, Any]]:
    """Measure gradient correlation between field pairs."""
    if len(fields) < 2:
        return []

    idx_to_point = {point.index: point for point in grid.values()}
    check_fields = fields[:10]
    interactions = []

    for i, field_one in enumerate(check_fields):
        for field_two in check_fields[i + 1 :]:
            grad_products = []
            for point in grid.values():
                value_one = point.field_values.get(field_one)
                value_two = point.field_values.get(field_two)
                if value_one is None or value_two is None:
                    continue
                for neighbor_idx in point.neighbors:
                    neighbor = idx_to_point.get(neighbor_idx)
                    if neighbor is None:
                        continue
                    next_one = neighbor.field_values.get(field_one)
                    next_two = neighbor.field_values.get(field_two)
                    if next_one is None or next_two is None:
                        continue
                    # Normalized gradient dot-product: do the two fields increase/decrease together across edges?
                    grad_products.append((next_one - value_one) * (next_two - value_two))

            if len(grad_products) < 10:
                continue

            mean_product = sum(grad_products) / len(grad_products)
            scale = sum(abs(product) for product in grad_products) / len(grad_products)
            correlation = mean_product / scale if scale > 1e-12 else 0.0
            if abs(correlation) < 0.05:
                continue

            interactions.append(
                {
                    "fields": [field_one, field_two],
                    "gradient_correlation": round(correlation, 4),
                    "strength": round(abs(correlation), 4),
                    "direction": (
                        "synergistic"
                        if correlation > 0.1
                        else "antagonistic"
                        if correlation < -0.1
                        else "weak"
                    ),
                    "samples": len(grad_products),
                }
            )

    interactions.sort(key=lambda item: item["strength"], reverse=True)
    return interactions[:15]
