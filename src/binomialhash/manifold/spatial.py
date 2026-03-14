"""Spatial reasoning primitives for manifold geometry."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .structures import ManifoldSurface

from .structures import GridPoint

try:
    import numpy as np
except ImportError:
    np = None


def _idx_map(surface: "ManifoldSurface") -> Dict[int, GridPoint]:
    return {gp.index: gp for gp in surface.grid.values()}


def _build_graph_laplacian(
    surface: "ManifoldSurface",
) -> Tuple[Any, List[int], Dict[int, int]]:
    """L = D - A (unnormalized). Returns (L, ordered_ids, id_to_pos). Requires numpy."""
    # L = D - A (degree minus adjacency) captures the graph's connectivity structure.
    if np is None:
        raise RuntimeError("numpy required for spatial reasoning tools.")

    ordered = sorted(gp.index for gp in surface.grid.values())
    pos = {idx: i for i, idx in enumerate(ordered)}
    n = len(ordered)
    L = np.zeros((n, n), dtype=float)
    idx_to_gp = _idx_map(surface)

    for idx in ordered:
        gp = idx_to_gp[idx]
        i = pos[idx]
        deg = 0
        for ni in gp.neighbors:
            if ni in pos:
                j = pos[ni]
                L[i, j] = -1.0
                deg += 1
        L[i, i] = float(deg)

    return L, ordered, pos


def _eigen_decomposition(L: Any, k: int) -> Tuple[Any, Any]:
    """Smallest k eigenvalues and eigenvectors of L."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    k = min(k, len(eigenvalues))
    return eigenvalues[:k], eigenvectors[:, :k]


def heat_kernel(
    surface: "ManifoldSurface",
    target_field: Optional[str] = None,
    time_scales: Optional[List[float]] = None,
    n_eigen: int = 20,
    top_k_bottlenecks: int = 10,
) -> Dict[str, Any]:
    """HKS(x, t) = sum_i exp(-lambda_i * t) * phi_i(x)^2. Points where heat
    dissipates slowly (low HKS at small t) relative to neighbors are bottlenecks."""
    # HKS measures how heat dissipates from each vertex; bottlenecks are points where heat gets trapped.
    if np is None:
        return {"error": "numpy required for heat_kernel."}
    if surface.vertex_count < 4:
        return {"error": "Grid too small for heat kernel analysis."}

    L, ordered, pos = _build_graph_laplacian(surface)
    n_eig = min(n_eigen, len(ordered) - 1)
    if n_eig < 2:
        return {"error": "Not enough vertices for eigendecomposition."}

    evals, evecs = _eigen_decomposition(L, n_eig)

    # Scales chosen relative to spectral range: fast times capture local geometry, slow times capture global structure.
    if time_scales is None:
        lam_max = max(float(evals[-1]), 1e-6)
        lam_min = max(float(evals[1]) if len(evals) > 1 else 1e-6, 1e-6)
        time_scales = [
            0.1 / lam_max,
            1.0 / lam_max,
            1.0 / lam_min,
            10.0 / lam_min,
        ]

    idx_to_gp = _idx_map(surface)
    n = len(ordered)
    hks_matrix = np.zeros((n, len(time_scales)))

    for ti, t in enumerate(time_scales):
        heat_coeffs = np.exp(-evals * t)
        for i in range(n):
            hks_matrix[i, ti] = float(np.sum(heat_coeffs * evecs[i, :] ** 2))

    bottleneck_scores = np.zeros(n)
    for i in range(n):
        gp = idx_to_gp[ordered[i]]
        neighbor_hks = []
        for ni in gp.neighbors:
            if ni in pos:
                neighbor_hks.append(hks_matrix[pos[ni], 0])
        if neighbor_hks:
            my_hks = hks_matrix[i, 0]
            mean_neighbor = sum(neighbor_hks) / len(neighbor_hks)
            if mean_neighbor > 1e-12:
                bottleneck_scores[i] = max(0.0, 1.0 - my_hks / mean_neighbor)

    top_indices = np.argsort(bottleneck_scores)[::-1][:top_k_bottlenecks]
    bottlenecks = []
    for idx in top_indices:
        if bottleneck_scores[idx] < 1e-6:
            break
        gp = idx_to_gp[ordered[idx]]
        entry: Dict[str, Any] = {
            "coord": gp.axis_coords,
            "bottleneck_score": round(float(bottleneck_scores[idx]), 6),
            "density": gp.density,
        }
        if target_field and target_field in gp.field_values:
            entry["target_value"] = round(gp.field_values[target_field], 6)
        bottlenecks.append(entry)

    shape_clusters: Dict[str, List[Tuple]] = {}
    if n > 4:
        from_hks = hks_matrix
        mean_hks = from_hks.mean(axis=0)
        std_hks = from_hks.std(axis=0)
        std_hks[std_hks < 1e-12] = 1.0
        normed = (from_hks - mean_hks) / std_hks
        labels = ["low", "mid", "high"]
        for i in range(n):
            score = float(normed[i].mean())
            if score < -0.5:
                label = "low"
            elif score > 0.5:
                label = "high"
            else:
                label = "mid"
            shape_clusters.setdefault(label, []).append(idx_to_gp[ordered[i]].axis_coords)

    return {
        "vertices": n,
        "eigenvalues_used": n_eig,
        "time_scales": [round(t, 6) for t in time_scales],
        "bottlenecks": bottlenecks,
        "shape_cluster_sizes": {k: len(v) for k, v in shape_clusters.items()},
        "shape_cluster_preview": {
            k: v[:5] for k, v in shape_clusters.items()
        },
        "spectral_gap": round(float(evals[1]) if len(evals) > 1 else 0.0, 6),
    }


def reeb_graph(
    surface: "ManifoldSurface",
    target_field: str,
    n_levels: int = 20,
) -> Dict[str, Any]:
    """Topological skeleton via level-set tracking. Sweeps quantile-based level
    sets, tracks connected components; produces nodes (birth/merge/split/death)
    and arcs."""
    idx_to_gp = _idx_map(surface)
    valued = []
    for gp in surface.grid.values():
        v = gp.field_values.get(target_field)
        if v is not None:
            valued.append((v, gp.index))

    if len(valued) < 4:
        return {"error": f"Not enough valued points for Reeb graph (got {len(valued)})."}

    # Tracks how connected components of level sets evolve as the function value sweeps from min to max.
    valued.sort()
    vals = [v for v, _ in valued]
    n = len(vals)

    thresholds = []
    for i in range(n_levels + 1):
        q = i / n_levels
        idx = min(int(q * (n - 1)), n - 1)
        thresholds.append(vals[idx])
    thresholds = sorted(set(thresholds))
    if len(thresholds) < 2:
        return {"error": "All values identical; no Reeb structure."}

    def _components_at(lo: float, hi: float) -> List[set]:
        active = set()
        for val, gid in valued:
            if lo <= val < hi:
                active.add(gid)
        if not active:
            return []
        adj: Dict[int, set] = {g: set() for g in active}
        for gid in active:
            gp = idx_to_gp.get(gid)
            if gp is None:
                continue
            for ni in gp.neighbors:
                if ni in active:
                    adj[gid].add(ni)

        seen = set()
        comps = []
        for start in active:
            if start in seen:
                continue
            comp = set()
            queue = deque([start])
            seen.add(start)
            while queue:
                cur = queue.popleft()
                comp.add(cur)
                for nb in adj.get(cur, set()):
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    nodes: List[Dict[str, Any]] = []
    arcs: List[Dict[str, Any]] = []
    prev_comps: List[set] = []
    node_id = 0
    prev_node_map: Dict[int, int] = {}

    for li in range(len(thresholds) - 1):
        lo, hi = thresholds[li], thresholds[li + 1]
        cur_comps = _components_at(lo, hi)
        level_val = round((lo + hi) / 2, 6)

        if not prev_comps:
            for ci, comp in enumerate(cur_comps):
                nodes.append({
                    "id": node_id, "type": "birth",
                    "level": level_val, "size": len(comp),
                })
                for gid in comp:
                    prev_node_map[gid] = node_id
                node_id += 1
            prev_comps = cur_comps
            continue

        cur_node_ids = {}
        for ci, cur_comp in enumerate(cur_comps):
            parent_nodes = set()
            for gid in cur_comp:
                for prev_ci, prev_comp in enumerate(prev_comps):
                    overlap = cur_comp & prev_comp
                    if overlap:
                        for og in overlap:
                            if og in prev_node_map:
                                parent_nodes.add(prev_node_map[og])

            if not parent_nodes:
                nodes.append({
                    "id": node_id, "type": "birth",
                    "level": level_val, "size": len(cur_comp),
                })
                for gid in cur_comp:
                    cur_node_ids[gid] = node_id
                node_id += 1
            elif len(parent_nodes) > 1:
                nodes.append({
                    "id": node_id, "type": "merge",
                    "level": level_val, "size": len(cur_comp),
                    "parents": sorted(parent_nodes),
                })
                for pid in parent_nodes:
                    arcs.append({"from": pid, "to": node_id})
                for gid in cur_comp:
                    cur_node_ids[gid] = node_id
                node_id += 1
            else:
                pid = next(iter(parent_nodes))
                for gid in cur_comp:
                    cur_node_ids[gid] = pid

        child_map: Dict[int, List[int]] = {}
        for gid, nid in cur_node_ids.items():
            if gid in prev_node_map:
                old = prev_node_map[gid]
                if old != nid:
                    child_map.setdefault(old, []).append(nid)
        for parent_id, children in child_map.items():
            unique_children = list(set(children))
            if len(unique_children) > 1:
                nodes.append({
                    "id": node_id, "type": "split",
                    "level": level_val, "children": unique_children,
                })
                arcs.append({"from": parent_id, "to": node_id})
                node_id += 1

        prev_node_map = cur_node_ids
        prev_comps = cur_comps

    births = sum(1 for n in nodes if n["type"] == "birth")
    merges = sum(1 for n in nodes if n["type"] == "merge")
    splits = sum(1 for n in nodes if n["type"] == "split")

    if merges == 0 and splits == 0:
        complexity = "simple"
    elif merges + splits <= 3:
        complexity = "moderate"
    else:
        complexity = "complex"

    return {
        "target_field": target_field,
        "n_levels": len(thresholds) - 1,
        "nodes": nodes[:50],
        "arcs": arcs[:50],
        "summary": {
            "births": births, "merges": merges, "splits": splits,
            "total_nodes": len(nodes), "total_arcs": len(arcs),
            "complexity_label": complexity,
        },
        "value_range": [round(vals[0], 6), round(vals[-1], 6)],
    }


def vector_field_analysis(
    surface: "ManifoldSurface",
    target_field: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Divergence > 0 → source; < 0 → sink. High |curl| → rotational structure."""
    idx_to_gp = _idx_map(surface)
    # div > 0: field radiates outward (source); div < 0: converges inward (sink); high curl: rotational flow.
    divergences: Dict[int, float] = {}
    curl_scores: Dict[int, float] = {}

    for gp in surface.grid.values():
        my_val = gp.field_values.get(target_field)
        if my_val is None:
            continue
        if not gp.neighbors:
            continue

        outward_flux = 0.0
        neighbor_grads = []
        for ni in gp.neighbors:
            ngp = idx_to_gp.get(ni)
            if ngp is None:
                continue
            nval = ngp.field_values.get(target_field)
            if nval is None:
                continue
            grad = nval - my_val
            outward_flux += grad
            neighbor_grads.append(grad)

        div = outward_flux / max(len(gp.neighbors), 1)
        divergences[gp.index] = div

        if len(neighbor_grads) >= 3:
            sign_changes = 0
            for i in range(len(neighbor_grads)):
                g1 = neighbor_grads[i]
                g2 = neighbor_grads[(i + 1) % len(neighbor_grads)]
                if (g1 > 0) != (g2 > 0):
                    sign_changes += 1
            curl_scores[gp.index] = sign_changes / len(neighbor_grads)
        else:
            curl_scores[gp.index] = 0.0

    if not divergences:
        return {"error": f"No computable gradients for field '{target_field}'."}

    div_vals = list(divergences.values())
    mean_div = sum(div_vals) / len(div_vals)
    std_div = (sum((d - mean_div) ** 2 for d in div_vals) / len(div_vals)) ** 0.5

    sources = []
    sinks = []
    vortices = []
    saddles = []

    for gp in surface.grid.values():
        if gp.index not in divergences:
            continue
        d = divergences[gp.index]
        c = curl_scores.get(gp.index, 0.0)

        z_div = (d - mean_div) / std_div if std_div > 1e-12 else 0.0

        ftype = "regular"
        if z_div > 1.5:
            ftype = "source"
            sources.append(gp)
        elif z_div < -1.5:
            ftype = "sink"
            sinks.append(gp)

        if c > 0.6:
            if ftype == "regular":
                ftype = "vortex"
                vortices.append(gp)
            elif ftype in ("source", "sink"):
                ftype = "saddle"
                saddles.append(gp)

    def _fmt(gps: List[GridPoint], limit: int) -> List[Dict[str, Any]]:
        sorted_gps = sorted(
            gps,
            key=lambda g: abs(divergences.get(g.index, 0)),
            reverse=True,
        )[:limit]
        result = []
        for g in sorted_gps:
            entry: Dict[str, Any] = {
                "coord": g.axis_coords,
                "divergence": round(divergences.get(g.index, 0), 6),
                "curl_score": round(curl_scores.get(g.index, 0), 4),
            }
            if target_field in g.field_values:
                entry["value"] = round(g.field_values[target_field], 6)
            result.append(entry)
        return result

    return {
        "target_field": target_field,
        "vertices_analyzed": len(divergences),
        "mean_divergence": round(mean_div, 6),
        "std_divergence": round(std_div, 6),
        "sources": _fmt(sources, top_k),
        "sinks": _fmt(sinks, top_k),
        "vortices": _fmt(vortices, top_k),
        "saddles": _fmt(saddles, top_k),
        "flow_summary": {
            "n_sources": len(sources),
            "n_sinks": len(sinks),
            "n_vortices": len(vortices),
            "n_saddles": len(saddles),
            "has_rotational_structure": len(vortices) > 0,
            "flow_balance": round(
                len(sources) / max(len(sinks), 1), 3
            ),
        },
    }


def laplacian_spectrum(
    surface: "ManifoldSurface",
    n_eigen: int = 15,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Spectral gap (lambda_2) measures connectivity; Fiedler vector gives optimal
    bisection; higher eigenvectors give finer partitions."""
    if np is None:
        return {"error": "numpy required for laplacian_spectrum."}
    if surface.vertex_count < 4:
        return {"error": "Grid too small for spectral analysis."}

    L, ordered, pos = _build_graph_laplacian(surface)
    n_eig = min(n_eigen, len(ordered) - 1)
    evals, evecs = _eigen_decomposition(L, n_eig)

    # λ₂ (algebraic connectivity): large gap = tightly connected, small gap = bottleneck.
    spectral_gap = float(evals[1]) if len(evals) > 1 else 0.0

    if spectral_gap < 1e-6:
        connectivity_label = "disconnected"
    elif spectral_gap < 0.5:
        connectivity_label = "weakly_connected"
    elif spectral_gap < 2.0:
        connectivity_label = "moderately_connected"
    else:
        connectivity_label = "strongly_connected"

    idx_to_gp = _idx_map(surface)

    fiedler_partition = {"positive": [], "negative": []}
    if len(evals) > 1:
        fiedler = evecs[:, 1]
        for i, idx in enumerate(ordered):
            gp = idx_to_gp[idx]
            side = "positive" if fiedler[i] >= 0 else "negative"
            fiedler_partition[side].append(gp.axis_coords)

    auto_k = 2
    if n_clusters is None:
        gaps = []
        for i in range(1, min(len(evals) - 1, 10)):
            gaps.append((float(evals[i + 1] - evals[i]), i + 1))
        if gaps:
            gaps.sort(reverse=True)
            auto_k = gaps[0][1]
            auto_k = max(2, min(auto_k, 8))
    else:
        auto_k = n_clusters

    clusters: Dict[int, List[Tuple]] = {}
    if len(evals) >= auto_k and auto_k >= 2:
        embedding = evecs[:, 1:auto_k]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        embedding = embedding / norms

        centroids_idx = [0]
        for _ in range(auto_k - 1):
            dists = np.array([
                min(np.linalg.norm(embedding[i] - embedding[c]) for c in centroids_idx)
                for i in range(len(embedding))
            ])
            centroids_idx.append(int(np.argmax(dists)))

        centroids = embedding[centroids_idx]
        for i, idx in enumerate(ordered):
            dists_to_centroids = [
                float(np.linalg.norm(embedding[i] - c)) for c in centroids
            ]
            label = int(np.argmin(dists_to_centroids))
            gp = idx_to_gp[idx]
            clusters.setdefault(label, []).append(gp.axis_coords)

    return {
        "vertices": len(ordered),
        "eigenvalues": [round(float(e), 6) for e in evals],
        "spectral_gap": round(spectral_gap, 6),
        "connectivity_label": connectivity_label,
        "fiedler_partition_sizes": {
            k: len(v) for k, v in fiedler_partition.items()
        },
        "fiedler_partition_preview": {
            k: v[:5] for k, v in fiedler_partition.items()
        },
        "suggested_k": auto_k,
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "cluster_preview": {k: v[:5] for k, v in clusters.items()},
    }


def scalar_harmonics(
    surface: "ManifoldSurface",
    target_field: str,
    n_modes: int = 10,
    top_k_anomalies: int = 10,
) -> Dict[str, Any]:
    """Decompose field into Laplacian eigenbasis: smooth = first k modes,
    residual = structural deviations. Energy spectrum = variance per mode."""
    if np is None:
        return {"error": "numpy required for scalar_harmonics."}

    L, ordered, pos = _build_graph_laplacian(surface)
    idx_to_gp = _idx_map(surface)
    # Decomposing into Laplacian eigenbasis separates smooth trends (low-freq) from structural anomalies (high-freq residual).
    signal = np.zeros(len(ordered))
    valid = 0
    for i, idx in enumerate(ordered):
        v = idx_to_gp[idx].field_values.get(target_field)
        if v is not None:
            signal[i] = v
            valid += 1

    if valid < 4:
        return {"error": f"Not enough values for field '{target_field}'."}

    n_modes = min(n_modes, len(ordered) - 1)
    evals, evecs = _eigen_decomposition(L, n_modes)

    coefficients = evecs.T @ signal
    energies = coefficients ** 2
    total_energy = float(energies.sum())
    if total_energy < 1e-12:
        total_energy = 1.0

    smooth = evecs @ coefficients
    residual = signal - smooth

    residual_abs = np.abs(residual)
    top_anomaly_idx = np.argsort(residual_abs)[::-1][:top_k_anomalies]

    anomalies = []
    for ai in top_anomaly_idx:
        if residual_abs[ai] < 1e-9:
            break
        gp = idx_to_gp[ordered[ai]]
        anomalies.append({
            "coord": gp.axis_coords,
            "observed": round(float(signal[ai]), 6),
            "smooth_prediction": round(float(smooth[ai]), 6),
            "residual": round(float(residual[ai]), 6),
            "residual_magnitude": round(float(residual_abs[ai]), 6),
        })

    energy_spectrum = []
    cumulative = 0.0
    for mi in range(n_modes):
        frac = float(energies[mi]) / total_energy
        cumulative += frac
        energy_spectrum.append({
            "mode": mi,
            "eigenvalue": round(float(evals[mi]), 6),
            "energy_fraction": round(frac, 6),
            "cumulative": round(cumulative, 6),
        })

    modes_for_90pct = 0
    cum = 0.0
    for mi in range(n_modes):
        cum += float(energies[mi]) / total_energy
        modes_for_90pct = mi + 1
        if cum >= 0.9:
            break

    return {
        "target_field": target_field,
        "vertices": len(ordered),
        "modes_computed": n_modes,
        "energy_spectrum": energy_spectrum,
        "modes_for_90pct_variance": modes_for_90pct,
        "smoothness_ratio": round(cumulative, 4),
        "anomalies": anomalies,
        "interpretation": (
            "smooth_dominated" if modes_for_90pct <= 3
            else "moderately_complex" if modes_for_90pct <= 6
            else "highly_structured"
        ),
    }


def diffusion_distance(
    surface: "ManifoldSurface",
    landmark_coords: Optional[List[Tuple]] = None,
    time_param: float = 1.0,
    n_eigen: int = 15,
    n_landmarks: int = 8,
) -> Dict[str, Any]:
    """D_t(x,y)^2 = sum_i exp(-2*lambda_i*t) * (phi_i(x) - phi_i(y))^2. Averages
    over all paths (unlike geodesic); robust to noise and sparse regions."""
    if np is None:
        return {"error": "numpy required for diffusion_distance."}
    if surface.vertex_count < 4:
        return {"error": "Grid too small for diffusion distance."}

    # Averages over all paths (unlike geodesic shortest path), making it robust to noise.
    L, ordered, pos = _build_graph_laplacian(surface)
    n_eig = min(n_eigen, len(ordered) - 1)
    evals, evecs = _eigen_decomposition(L, n_eig)

    idx_to_gp = _idx_map(surface)

    if landmark_coords:
        landmark_positions = []
        for lc in landmark_coords:
            gp = surface.grid.get(lc)
            if gp and gp.index in pos:
                landmark_positions.append(pos[gp.index])
        if len(landmark_positions) < 2:
            return {"error": "Need at least 2 valid landmark coordinates."}
    else:
        n_lm = min(n_landmarks, len(ordered))
        step = max(1, len(ordered) // n_lm)
        landmark_positions = list(range(0, len(ordered), step))[:n_lm]

    diffusion_coords = np.zeros((len(ordered), n_eig))
    for k in range(n_eig):
        if evals[k] < 1e-12:
            continue
        scale = math.exp(-evals[k] * time_param)
        diffusion_coords[:, k] = scale * evecs[:, k]

    n_lm = len(landmark_positions)
    dist_matrix = np.zeros((n_lm, n_lm))
    for i in range(n_lm):
        for j in range(i + 1, n_lm):
            pi, pj = landmark_positions[i], landmark_positions[j]
            diff = diffusion_coords[pi] - diffusion_coords[pj]
            d = float(np.sqrt(np.sum(diff ** 2)))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    landmarks_info = []
    for li, lp in enumerate(landmark_positions):
        gp = idx_to_gp[ordered[lp]]
        landmarks_info.append({
            "index": li,
            "coord": gp.axis_coords,
            "density": gp.density,
        })

    distance_pairs = []
    for i in range(n_lm):
        for j in range(i + 1, n_lm):
            distance_pairs.append({
                "from": landmarks_info[i]["coord"],
                "to": landmarks_info[j]["coord"],
                "diffusion_distance": round(float(dist_matrix[i, j]), 6),
            })
    distance_pairs.sort(key=lambda x: x["diffusion_distance"])

    all_dists = [dp["diffusion_distance"] for dp in distance_pairs]
    mean_d = sum(all_dists) / len(all_dists) if all_dists else 0
    max_d = max(all_dists) if all_dists else 0
    min_d = min(all_dists) if all_dists else 0

    close_threshold = mean_d * 0.3 if mean_d > 0 else 0
    far_threshold = mean_d * 2.0
    close_pairs = [dp for dp in distance_pairs if dp["diffusion_distance"] <= close_threshold]
    far_pairs = [dp for dp in distance_pairs if dp["diffusion_distance"] >= far_threshold]

    return {
        "time_param": time_param,
        "eigenvalues_used": n_eig,
        "n_landmarks": n_lm,
        "landmarks": landmarks_info,
        "distance_pairs": distance_pairs[:20],
        "statistics": {
            "mean_distance": round(mean_d, 6),
            "min_distance": round(min_d, 6),
            "max_distance": round(max_d, 6),
            "diameter_ratio": round(max_d / mean_d, 3) if mean_d > 0 else 0,
        },
        "tightly_connected_pairs": close_pairs[:5],
        "loosely_connected_pairs": far_pairs[:5],
    }
