"""Navigation and pathfinding on manifold grid surfaces."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .structures import GridPoint

if TYPE_CHECKING:
    from .structures import ManifoldSurface


def dijkstra(
    grid: Dict[Tuple, GridPoint],
    idx_to_point: Dict[int, GridPoint],
    start_idx: int,
    end_idx: int,
    target_field: Optional[str],
) -> Tuple[Optional[List[int]], Optional[float]]:
    """Weight = |Δ target| + ε per edge, or 1.0 if no target."""
    import heapq

    distances = {start_idx: 0.0}
    previous: Dict[int, int] = {}
    heap = [(0.0, start_idx)]
    visited = set()

    while heap:
        distance, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == end_idx:
            break

        point = idx_to_point.get(current)
        if point is None:
            continue

        for neighbor_idx in point.neighbors:
            if neighbor_idx in visited:
                continue
            neighbor = idx_to_point.get(neighbor_idx)
            if neighbor is None:
                continue
            if target_field:
                # |Δ target| + ε makes geodesics follow the smoothest path through the target field landscape.
                current_value = point.field_values.get(target_field, 0.0)
                neighbor_value = neighbor.field_values.get(target_field, 0.0)
                weight = abs(neighbor_value - current_value) + 1e-6
            else:
                weight = 1.0
            new_distance = distance + weight
            if new_distance < distances.get(neighbor_idx, float("inf")):
                distances[neighbor_idx] = new_distance
                previous[neighbor_idx] = current
                heapq.heappush(heap, (new_distance, neighbor_idx))

    if end_idx not in distances:
        return None, None

    path = []
    current = end_idx
    while current != start_idx:
        path.append(current)
        current = previous.get(current)
        if current is None:
            return None, None
    path.append(start_idx)
    path.reverse()
    return path, distances[end_idx]


def idx_to_point(surface: ManifoldSurface) -> Dict[int, GridPoint]:
    return {point.index: point for point in surface.grid.values()}


def resolve_coord(surface: ManifoldSurface, coord: Tuple) -> Optional[GridPoint]:
    return surface.grid.get(coord)


def basin_of(surface: ManifoldSurface, point: GridPoint, target_field: str) -> Optional[Tuple]:
    """Follow steepest descent to find which minimum this point drains to."""
    index_map = idx_to_point(surface)
    current = point
    visited = {current.index}
    # 100-step cap prevents infinite loops in degenerate graphs.
    for _ in range(100):
        my_value = current.field_values.get(target_field)
        if my_value is None:
            return None
        best_neighbor = None
        best_value = my_value
        for neighbor_idx in current.neighbors:
            neighbor = index_map.get(neighbor_idx)
            if neighbor is None:
                continue
            neighbor_value = neighbor.field_values.get(target_field)
            if neighbor_value is not None and neighbor_value < best_value:
                best_value = neighbor_value
                best_neighbor = neighbor
        if best_neighbor is None or best_neighbor.index in visited:
            return current.axis_coords
        visited.add(best_neighbor.index)
        current = best_neighbor
    return current.axis_coords


def bfs_distances(
    surface: ManifoldSurface,
    center: Tuple,
    max_hops: int,
) -> Tuple[Optional[GridPoint], Dict[int, int]]:
    point = surface.grid.get(center)
    if point is None:
        return None, {}

    index_map = idx_to_point(surface)
    distances = {point.index: 0}
    queue = deque([point.index])
    while queue:
        current = queue.popleft()
        if distances[current] >= max_hops:
            continue
        current_point = index_map.get(current)
        if current_point is None:
            continue
        for neighbor_idx in current_point.neighbors:
            if neighbor_idx in index_map and neighbor_idx not in distances:
                distances[neighbor_idx] = distances[current] + 1
                queue.append(neighbor_idx)
    return point, distances


def summarize_points(
    surface: ManifoldSurface,
    points: List[GridPoint],
    target_field: Optional[str] = None,
) -> Dict[str, Any]:
    if not points:
        return {"points": 0}

    values = []
    if target_field:
        values = [
            point.field_values.get(target_field)
            for point in points
            if point.field_values.get(target_field) is not None
        ]
    curvatures = [point.curvature for point in points]
    densities = [point.density for point in points]
    result: Dict[str, Any] = {
        "points": len(points),
        "mean_curvature": round(sum(curvatures) / len(curvatures), 6),
        "mean_density": round(sum(densities) / len(densities), 6),
    }
    if values:
        result.update(
            {
                "target_field": target_field,
                "mean_value": round(sum(values) / len(values), 6),
                "min_value": round(min(values), 6),
                "max_value": round(max(values), 6),
            }
        )
    return result


def geodesic_path(
    surface: ManifoldSurface,
    start: Tuple,
    end: Tuple,
    target_field: Optional[str] = None,
) -> Dict[str, Any]:
    if start not in surface.grid:
        return {"error": f"Start {start} not in grid."}
    if end not in surface.grid:
        return {"error": f"End {end} not in grid."}
    if start == end:
        point = surface.grid[start]
        return {
            "hops": 0,
            "total_cost": 0.0,
            "waypoints": [
                {"coord": start, "fields": {key: round(value, 6) for key, value in point.field_values.items()}}
            ],
        }

    index_map = idx_to_point(surface)
    start_idx = surface.grid[start].index
    end_idx = surface.grid[end].index
    path_indices, cost = dijkstra(surface.grid, index_map, start_idx, end_idx, target_field)
    if path_indices is None:
        return {"error": "No path found (disconnected components)."}

    waypoints = []
    for offset, point_idx in enumerate(path_indices):
        point = index_map[point_idx]
        waypoint: Dict[str, Any] = {
            "step": offset,
            "coord": point.axis_coords,
            "fields": {key: round(value, 6) for key, value in point.field_values.items()},
            "density": point.density,
        }
        if target_field:
            waypoint["morse_type"] = point.morse_type.get(target_field)
        if offset > 0:
            previous_point = index_map[path_indices[offset - 1]]
            deltas = {}
            for field_name in surface.field_columns:
                previous_value = previous_point.field_values.get(field_name)
                current_value = point.field_values.get(field_name)
                if previous_value is not None and current_value is not None:
                    delta = current_value - previous_value
                    if abs(delta) > 1e-9:
                        deltas[field_name] = round(delta, 6)
            waypoint["delta_from_prev"] = deltas
        waypoints.append(waypoint)

    return {
        "start": start,
        "end": end,
        "hops": len(waypoints) - 1,
        "total_cost": round(cost or 0.0, 6),
        "weight": target_field or "hop_count",
        "waypoints": waypoints,
    }


def controlled_walk(surface: ManifoldSurface, walk_axis: str, target_field: str) -> Dict[str, Any]:
    axis_idx = None
    for index, axis in enumerate(surface.axes):
        if axis.column == walk_axis:
            axis_idx = index
            break
    if axis_idx is None:
        return {"error": f"Axis '{walk_axis}' not found. Available: {[axis.column for axis in surface.axes]}"}

    axis = surface.axes[axis_idx]
    buckets: Dict[str, List[float]] = {}
    for point in surface.grid.values():
        axis_value = point.axis_coords[axis_idx]
        target_value = point.field_values.get(target_field)
        if target_value is not None:
            buckets.setdefault(axis_value, []).append(target_value)

    if not buckets:
        return {"error": f"No data for axis '{walk_axis}' with field '{target_field}'."}

    profile = []
    for value in [str(axis_value) for axis_value in axis.values]:
        if value not in buckets:
            continue
        values = buckets[value]
        profile.append(
            {
                "axis_value": value,
                "mean": round(sum(values) / len(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "points": len(values),
            }
        )

    sensitivity = max((item["mean"] for item in profile), default=0.0) - min(
        (item["mean"] for item in profile), default=0.0
    )
    return {
        "axis": walk_axis,
        "target_field": target_field,
        "steps": len(profile),
        "sensitivity": round(sensitivity, 6),
        "profile": profile,
    }


def orbit(
    surface: ManifoldSurface,
    center: Tuple,
    radius: int,
    target_field: Optional[str] = None,
    resolution: int = 16,
    mode: str = "ring",
) -> Dict[str, Any]:
    if radius < 1:
        return {"error": "radius must be >= 1"}

    center_point, distances = bfs_distances(surface, center, radius)
    if center_point is None:
        return {"error": f"Center {center} not in grid."}

    index_map = idx_to_point(surface)
    shells: Dict[int, List[GridPoint]] = {}
    for point_idx, distance in distances.items():
        shells.setdefault(distance, []).append(index_map[point_idx])

    if mode == "ring":
        selected = shells.get(radius, [])
    else:
        selected = [point for distance, points in shells.items() if distance <= radius for point in points]

    shell_profiles = [
        {"distance": distance, **summarize_points(surface, shells[distance], target_field)}
        for distance in sorted(shells.keys())
    ]

    if target_field:
        ranked = sorted(
            [point for point in selected if point.field_values.get(target_field) is not None],
            key=lambda point: abs(point.field_values.get(target_field, 0.0)),
            reverse=True,
        )[: max(3, min(resolution, 12))]
    else:
        ranked = sorted(selected, key=lambda point: abs(point.curvature), reverse=True)[
            : max(3, min(resolution, 12))
        ]

    return {
        "center": center,
        "mode": mode,
        "radius": radius,
        "target_field": target_field,
        "center_summary": summarize_points(surface, [center_point], target_field),
        "zone_summary": summarize_points(surface, selected, target_field),
        "shell_profiles": shell_profiles,
        "highlights": [
            {
                "coord": point.axis_coords,
                "value": (
                    round(point.field_values.get(target_field, 0.0), 6)
                    if target_field and point.field_values.get(target_field) is not None
                    else None
                ),
                "curvature": round(point.curvature, 6),
                "density": point.density,
            }
            for point in ranked
        ],
    }


def multiscale_view(
    surface: ManifoldSurface,
    center: Tuple,
    radii: List[int],
    target_field: Optional[str] = None,
    resolution: int = 16,
) -> Dict[str, Any]:
    # Runs orbit() at increasing radii to show how statistics change with scale.
    cleaned = sorted(set(radius for radius in radii if isinstance(radius, int) and radius >= 1))
    if not cleaned:
        return {"error": "radii must include at least one integer >= 1"}
    return {
        "center": center,
        "target_field": target_field,
        "scales": [
            orbit(surface, center, radius, target_field=target_field, resolution=resolution, mode="disk")
            for radius in cleaned
        ],
    }


def basin(
    surface: ManifoldSurface,
    seed: Tuple,
    target_field: str,
    direction: str = "descend",
) -> Dict[str, Any]:
    seed_point = surface.grid.get(seed)
    if seed_point is None:
        return {"error": f"Seed {seed} not in grid."}

    index_map = idx_to_point(surface)

    def flow(point: GridPoint) -> Optional[Tuple]:
        current = point
        visited = {current.index}
        for _ in range(100):
            my_value = current.field_values.get(target_field)
            if my_value is None:
                return None
            best_neighbor = None
            best_value = my_value
            for neighbor_idx in current.neighbors:
                neighbor = index_map.get(neighbor_idx)
                if neighbor is None:
                    continue
                neighbor_value = neighbor.field_values.get(target_field)
                if neighbor_value is None:
                    continue
                # Direction-aware: descend finds minima, ascend finds maxima.
                better = neighbor_value < best_value if direction != "ascend" else neighbor_value > best_value
                if better:
                    best_value = neighbor_value
                    best_neighbor = neighbor
            if best_neighbor is None or best_neighbor.index in visited:
                return current.axis_coords
            visited.add(best_neighbor.index)
            current = best_neighbor
        return current.axis_coords

    extremum = flow(seed_point)
    if extremum is None:
        return {"error": f"No target_field '{target_field}' at seed."}

    members = [point for point in surface.grid.values() if flow(point) == extremum]
    return {
        "seed": seed,
        "target_field": target_field,
        "direction": direction,
        "extremum_coord": extremum,
        "basin_summary": summarize_points(surface, members, target_field),
        "members_preview": [point.axis_coords for point in members[:20]],
    }


def trace_extremum(
    surface: ManifoldSurface,
    seed: Tuple,
    target_field: str,
    mode: str = "ridge",
    max_steps: int = 25,
) -> Dict[str, Any]:
    point = surface.grid.get(seed)
    if point is None:
        return {"error": f"Seed {seed} not in grid."}

    index_map = idx_to_point(surface)
    current = point
    path = [current]
    visited = {current.index}
    # Ridge traces follow steepest ascent; valley traces follow steepest descent.
    ascend = mode == "ridge"

    for _ in range(max_steps):
        my_value = current.field_values.get(target_field)
        if my_value is None:
            break
        best = None
        best_value = my_value
        for neighbor_idx in current.neighbors:
            neighbor = index_map.get(neighbor_idx)
            if neighbor is None or neighbor.index in visited:
                continue
            neighbor_value = neighbor.field_values.get(target_field)
            if neighbor_value is None:
                continue
            if (ascend and neighbor_value > best_value) or ((not ascend) and neighbor_value < best_value):
                best_value = neighbor_value
                best = neighbor
        if best is None:
            break
        path.append(best)
        visited.add(best.index)
        current = best

    return {
        "seed": seed,
        "mode": mode,
        "target_field": target_field,
        "steps": len(path) - 1,
        "path": [
            {
                "coord": point.axis_coords,
                "value": round(point.field_values.get(target_field, 0.0), 6),
                "curvature": round(point.curvature, 6),
            }
            for point in path
        ],
    }


def wrap_audit(surface: ManifoldSurface) -> Dict[str, Any]:
    return {
        "axes": [
            {
                "axis": axis.column,
                "wraps": axis.wraps,
                "wrap_orientation": axis.wrap_orientation,
                "wrap_orientation_label": (
                    "preserving"
                    if axis.wrap_orientation == 1
                    else "reversing"
                    if axis.wrap_orientation == -1
                    else "none"
                ),
            }
            for axis in surface.axes
        ],
        "surface_confidence": surface.surface_confidence_obj,
    }


def navigate(
    surface: ManifoldSurface,
    coord: Tuple,
    target_field: Optional[str] = None,
) -> Dict[str, Any]:
    point = surface.grid.get(coord)
    if point is None:
        return {"error": f"No grid point at {coord}"}

    index_map = idx_to_point(surface)
    neighbors = [index_map[neighbor_idx] for neighbor_idx in point.neighbors if neighbor_idx in index_map]

    gradients: Dict[str, Dict[str, Any]] = {}
    if target_field and target_field in point.field_values:
        my_value = point.field_values[target_field]
        for neighbor in neighbors:
            neighbor_value = neighbor.field_values.get(target_field)
            if neighbor_value is None:
                continue
            diff = neighbor_value - my_value
            direction = []
            for index, axis in enumerate(surface.axes):
                if point.axis_coords[index] != neighbor.axis_coords[index]:
                    direction.append(f"{axis.column}: {point.axis_coords[index]} -> {neighbor.axis_coords[index]}")
            direction_key = ", ".join(direction) if direction else "same"
            gradient_vector = {}
            for field_name in surface.field_columns:
                point_value = point.field_values.get(field_name)
                neighbor_field_value = neighbor.field_values.get(field_name)
                if point_value is not None and neighbor_field_value is not None:
                    gradient_vector[field_name] = round(neighbor_field_value - point_value, 6)
            gradients[direction_key] = {
                "delta": round(diff, 6),
                "neighbor_value": round(neighbor_value, 6),
                "neighbor_curvature": round(neighbor.curvature, 6),
                "neighbor_forman_ricci": round(neighbor.forman_ricci, 4),
                "gradient_vector": gradient_vector,
            }

    steepest = max(gradients.items(), key=lambda item: abs(item[1]["delta"])) if gradients else None
    basin_target = basin_of(surface, point, target_field) if target_field else None
    morse = point.morse_type.get(target_field) if target_field else None

    return {
        "position": {
            "coord": point.axis_coords,
            "field_values": {key: round(value, 6) for key, value in point.field_values.items()},
            "curvature": round(point.curvature, 6),
            "forman_ricci": round(point.forman_ricci, 4),
            "density": point.density,
            "neighbor_count": len(point.neighbors),
            "morse_type": morse,
        },
        "surface": surface.surface_name,
        "topology": (
            f"genus {surface.genus}, "
            f"{'orientable' if surface.orientable else 'non-orientable'}, "
            f"chi={surface.euler_characteristic}, beta_0={surface.betti_0}, beta_1={surface.betti_1}"
        ),
        "basin": {"drains_to": basin_target} if basin_target else None,
        "gradients": gradients,
        "steepest_move": (
            {"direction": steepest[0], "delta": steepest[1]["delta"]} if steepest else None
        ),
    }


def coverage_audit(surface: ManifoldSurface) -> Dict[str, Any]:
    densities = [point.density for point in surface.grid.values()]
    isolated = sum(1 for point in surface.grid.values() if not point.neighbors)
    return {
        "graph": {
            "vertices": surface.vertex_count,
            "edges": surface.edge_count,
            "components": surface.betti_0,
            "isolated_vertices": isolated,
        },
        "face_complex": {
            "V2": surface.face_vertex_count,
            "E2": surface.face_edge_count,
            "F": surface.face_count,
            "face_coverage": surface.face_coverage,
            "occupancy": surface.occupancy,
            "is_manifold": surface.is_manifold,
            "nonmanifold_edges": surface.nonmanifold_edges,
        },
        "density": {
            "mean": round(sum(densities) / len(densities), 6) if densities else 0.0,
            "min": min(densities) if densities else 0,
            "max": max(densities) if densities else 0,
        },
        "surface_confidence": surface.surface_confidence_obj,
    }


def gridpoint_record(surface: "ManifoldSurface", gp: GridPoint) -> Dict[str, Any]:
    """Flatten a GridPoint into a plain dict suitable for predicate evaluation."""
    rec = {ax.column: gp.axis_coords[i] for i, ax in enumerate(surface.axes)}
    rec.update(gp.field_values)
    rec["density"] = gp.density
    rec["curvature"] = gp.curvature
    rec["forman_ricci"] = gp.forman_ricci
    return rec


def find_frontier_edges(
    surface: "ManifoldSurface",
    pred_a: Callable[[Dict[str, Any]], bool],
    pred_b: Callable[[Dict[str, Any]], bool],
    target_field: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return edges in the manifold grid that cross between predicate regions A and B."""
    idx_to_pt = {gp.index: gp for gp in surface.grid.values()}
    frontier_edges: List[Dict[str, Any]] = []
    seen: set = set()

    for gp in surface.grid.values():
        rec_gp = gridpoint_record(surface, gp)
        label_gp = "A" if pred_a(rec_gp) else "B" if pred_b(rec_gp) else None
        if label_gp is None:
            continue
        for ni in gp.neighbors:
            if ni not in idx_to_pt:
                continue
            edge = tuple(sorted((gp.index, ni)))
            if edge in seen:
                continue
            seen.add(edge)
            np_ = idx_to_pt[ni]
            rec_n = gridpoint_record(surface, np_)
            label_n = "A" if pred_a(rec_n) else "B" if pred_b(rec_n) else None
            if label_n is None or label_n == label_gp:
                continue
            edge_info: Dict[str, Any] = {
                "from": gp.axis_coords,
                "to": np_.axis_coords,
                "labels": [label_gp, label_n],
            }
            if target_field:
                gv = gp.field_values.get(target_field)
                nv = np_.field_values.get(target_field)
                if gv is not None and nv is not None:
                    edge_info["target_jump"] = round(nv - gv, 6)
            frontier_edges.append(edge_info)
    return frontier_edges[: max(1, min(limit, 500))]
