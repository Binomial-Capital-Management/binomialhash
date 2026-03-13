from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .structures import GridPoint, ManifoldAxis


def build_grid(
    rows: List[Dict[str, Any]],
    axes: List[ManifoldAxis],
    fields: List[str],
) -> Dict[Tuple, GridPoint]:
    """Build the discrete mesh: map each row to a grid point defined by axis coordinates."""
    grid: Dict[Tuple, GridPoint] = {}
    accumulator: Dict[Tuple, Dict[str, List[float]]] = {}
    density: Dict[Tuple, int] = {}

    idx_counter = 0
    for row in rows:
        coord_parts = []
        valid = True
        for ax in axes:
            rv = row.get(ax.column)
            if rv is None:
                valid = False
                break
            coord_parts.append(str(rv))
        if not valid:
            continue
        coord = tuple(coord_parts)

        if coord not in accumulator:
            accumulator[coord] = {f: [] for f in fields}
            density[coord] = 0

        density[coord] += 1
        for f in fields:
            v = row.get(f)
            if v is not None:
                try:
                    accumulator[coord][f].append(float(v))
                except (ValueError, TypeError):
                    pass

    for coord, field_lists in accumulator.items():
        avg_fields: Dict[str, float] = {}
        for f, vals in field_lists.items():
            if vals:
                avg_fields[f] = sum(vals) / len(vals)
        grid[coord] = GridPoint(
            index=idx_counter,
            axis_coords=coord,
            field_values=avg_fields,
            density=density[coord],
        )
        idx_counter += 1

    return grid


def build_adjacency(grid: Dict[Tuple, GridPoint], axes: List[ManifoldAxis]) -> None:
    """Connect neighboring grid points along each axis."""
    coord_list = list(grid.keys())
    coord_to_idx = {c: grid[c].index for c in coord_list}

    for coord in coord_list:
        gp = grid[coord]
        for axis_i, ax in enumerate(axes):
            current_val = coord[axis_i]
            if not ax.ordered:
                for other_coord in coord_list:
                    if other_coord == coord:
                        continue
                    same_on_other_axes = all(
                        other_coord[j] == coord[j] for j in range(len(axes)) if j != axis_i
                    )
                    if same_on_other_axes and other_coord in coord_to_idx:
                        neighbor_idx = coord_to_idx[other_coord]
                        if neighbor_idx not in gp.neighbors:
                            gp.neighbors.append(neighbor_idx)
            else:
                val_list = [str(v) for v in ax.values]
                try:
                    pos = val_list.index(current_val)
                except ValueError:
                    continue
                for delta in [-1, 1]:
                    npos = pos + delta
                    if ax.wraps:
                        npos = npos % len(val_list)
                    if 0 <= npos < len(val_list):
                        new_coord = list(coord)
                        new_coord[axis_i] = val_list[npos]
                        new_coord_t = tuple(new_coord)
                        if new_coord_t in coord_to_idx:
                            neighbor_idx = coord_to_idx[new_coord_t]
                            if neighbor_idx not in gp.neighbors:
                                gp.neighbors.append(neighbor_idx)
