"""Structure and topology analysis."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

from ..schema import T_NUMERIC
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    extract_numeric_matrix,
    np,
    to_float_permissive,
)


def cluster_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    k: Optional[int] = None,
    max_k: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """K-means clustering with auto-k via silhouette scoring."""
    if np is None:
        return {"error": "numpy required for clustering."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 numeric fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < 10:
        return {"error": f"Not enough rows ({n})."}

    x = np.array(mat, dtype=float)
    mu = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    xn = (x - mu) / std

    def _kmeans(xn, k_val, max_iter, n_init):
        best_labels = None
        best_inertia = float("inf")
        n_pts = len(xn)
        for _ in range(n_init):
            idx = np.random.choice(n_pts, k_val, replace=False)
            centroids = xn[idx].copy()
            for _ in range(max_iter):
                dists = np.array([[np.sum((xn[i] - centroids[c]) ** 2) for c in range(k_val)] for i in range(n_pts)])
                labels = np.argmin(dists, axis=1)
                new_c = np.array([xn[labels == c].mean(axis=0) if np.sum(labels == c) > 0 else centroids[c] for c in range(k_val)])
                if np.allclose(new_c, centroids, atol=1e-6):
                    break
                centroids = new_c
            inertia = sum(np.sum((xn[i] - centroids[labels[i]]) ** 2) for i in range(n_pts))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
        return best_labels, best_centroids, best_inertia

    def _silhouette(xn, labels, k_val):
        n_pts = len(xn)
        if k_val < 2:
            return 0.0
        scores = []
        for i in range(min(n_pts, 500)):
            ci = labels[i]
            same = [j for j in range(n_pts) if labels[j] == ci and j != i]
            a = np.mean([np.sqrt(np.sum((xn[i] - xn[j]) ** 2)) for j in same]) if same else 0
            b = float("inf")
            for c in range(k_val):
                if c == ci:
                    continue
                others = [j for j in range(n_pts) if labels[j] == c]
                if others:
                    b = min(b, np.mean([np.sqrt(np.sum((xn[i] - xn[j]) ** 2)) for j in others]))
            s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0
            scores.append(s)
        return float(np.mean(scores)) if scores else 0.0

    max_k_val = max_k or policy.cluster_max_k
    if k is not None:
        best_k = min(max(k, 2), min(n - 1, max_k_val))
        labels, centroids, _ = _kmeans(xn, best_k, policy.cluster_max_iter, policy.cluster_n_init)
        sil = _silhouette(xn, labels, best_k)
    else:
        best_k = 2
        best_sil = -1
        best_result = None
        for kk in range(2, min(max_k_val + 1, n)):
            labels, centroids, _ = _kmeans(xn, kk, policy.cluster_max_iter, policy.cluster_n_init)
            sil = _silhouette(xn, labels, kk)
            if sil > best_sil:
                best_sil = sil
                best_k = kk
                best_result = (labels, centroids)
        labels, centroids = best_result
        sil = best_sil

    cluster_info = []
    for c in range(best_k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        centroid_orig = centroids[c] * std.flatten() + mu.flatten()
        info = {"cluster": c, "size": int(len(members))}
        for j, f in enumerate(fields):
            info[f"{f}_centroid"] = round(float(centroid_orig[j]), 4)
        cluster_info.append(info)

    return {
        "fields": fields, "samples": n, "k": best_k,
        "silhouette_score": round(sil, 4),
        "clusters": cluster_info,
        "assignments": [int(l) for l in labels[:min(n, 500)]],
    }


def spectral_decomposition_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Graph Laplacian eigenspectrum of k-NN similarity graph."""
    if np is None:
        return {"error": "numpy required for spectral decomposition."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    nn = n_neighbors or policy.spectral_default_neighbors
    nc = n_components or policy.spectral_default_components
    if n < nn + 2:
        return {"error": f"Not enough rows ({n}) for {nn} neighbors."}

    x = np.array(mat, dtype=float)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((x[i] - x[j]) ** 2)))
            dists[i, j] = d
            dists[j, i] = d

    adj = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:nn + 1]
        for j in neighbors:
            adj[i, j] = 1
            adj[j, i] = 1

    degree = np.diag(adj.sum(axis=1))
    laplacian = degree - adj
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(adj.sum(axis=1), 1e-12)))
    L_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    nc = min(nc, n - 1)
    evals = [round(float(eigenvalues[i]), 6) for i in range(min(nc + 2, n))]

    nonzero = [e for e in evals if e > 1e-6]
    spectral_gap = nonzero[1] / nonzero[0] if len(nonzero) >= 2 and nonzero[0] > 1e-12 else 0

    n_near_zero = sum(1 for e in evals if e < 0.1)

    # Fiedler partition
    fiedler = eigenvectors[:, 1] if n > 1 else np.zeros(n)
    partition = [0 if fiedler[i] < 0 else 1 for i in range(n)]

    return {
        "fields": fields, "samples": n, "n_neighbors": nn,
        "eigenvalues": evals,
        "spectral_gap": round(spectral_gap, 4),
        "cluster_suggestion": max(n_near_zero, 1),
        "fiedler_partition_sizes": [partition.count(0), partition.count(1)],
    }


def latent_sources_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    n_sources: Optional[int] = None,
    max_iter: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Independent Component Analysis (FastICA)."""
    if np is None:
        return {"error": "numpy required for ICA."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:15]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < 20:
        return {"error": f"Not enough rows ({n})."}

    max_it = max_iter or policy.ica_max_iter
    x = np.array(mat, dtype=float).T  # p x n
    p = x.shape[0]
    x = x - x.mean(axis=1, keepdims=True)

    # Whitening via PCA
    cov = np.cov(x)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    ns = n_sources or min(p, max(2, int(sum(evals > 0.01 * evals[0]))))
    ns = min(ns, p, n - 1)
    d = np.diag(1.0 / np.sqrt(np.maximum(evals[:ns], 1e-12)))
    white = d @ evecs[:, :ns].T @ x  # ns x n

    # FastICA with deflation
    w_all = np.zeros((ns, ns))
    for ic in range(ns):
        w = np.random.randn(ns)
        w /= np.linalg.norm(w)
        for _ in range(max_it):
            s = w @ white
            g = np.tanh(s)
            gp = 1.0 - g ** 2
            w_new = (white * g).mean(axis=1) - gp.mean() * w
            for prev in range(ic):
                w_new -= (w_new @ w_all[prev]) * w_all[prev]
            norm = np.linalg.norm(w_new)
            if norm < 1e-12:
                break
            w_new /= norm
            if abs(abs(float(w_new @ w)) - 1.0) < policy.ica_tol:
                w = w_new
                break
            w = w_new
        w_all[ic] = w

    unmixing = w_all @ d @ evecs[:, :ns].T
    sources = unmixing @ x  # ns x n

    source_info = []
    for ic in range(ns):
        s = sources[ic]
        s_std = float(s.std())
        if s_std > 1e-12:
            s_norm = (s - s.mean()) / s_std
            kurt = float((s_norm ** 4).mean()) - 3.0
        else:
            kurt = 0.0
        contributions = [(fields[j], round(float(unmixing[ic, j]), 4)) for j in range(p)]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        source_info.append({
            "source": ic, "kurtosis": round(kurt, 4),
            "top_contributions": [{"field": f, "weight": w} for f, w in contributions[:5]],
        })

    return {
        "fields": fields, "samples": n, "n_sources": ns,
        "sources": source_info,
    }


def graphical_model_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    method: str = "threshold",
    alpha: Optional[float] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Sparse precision matrix -> conditional dependency graph."""
    if np is None:
        return {"error": "numpy required for graphical model."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:20]
    fields = [f for f in fields if f in numeric_cols]
    p = len(fields)
    if p < 3:
        return {"error": "Need at least 3 fields."}

    mat = extract_numeric_matrix(rows, fields)
    n = len(mat)
    if n < p + 5:
        return {"error": f"Not enough rows ({n}) for {p} fields."}

    x = np.array(mat, dtype=float)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    cov = np.cov(x, rowvar=False)
    alpha_val = alpha or policy.graphical_default_alpha

    try:
        precision = np.linalg.inv(cov + alpha_val * np.eye(p))
    except np.linalg.LinAlgError:
        return {"error": "Covariance matrix is singular."}

    diag = np.sqrt(np.diag(precision))
    partial = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if diag[i] > 1e-12 and diag[j] > 1e-12:
                partial[i, j] = -precision[i, j] / (diag[i] * diag[j])

    threshold = 2.0 / math.sqrt(n)
    edges = []
    degree = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if abs(partial[i, j]) > threshold:
                edges.append({
                    "field_a": fields[i], "field_b": fields[j],
                    "partial_corr": round(float(partial[i, j]), 4),
                    "strength": round(abs(float(partial[i, j])), 4),
                })
                degree[i] += 1
                degree[j] += 1
    edges.sort(key=lambda x: x["strength"], reverse=True)

    nodes = [{"field": fields[i], "degree": degree[i]} for i in range(p)]
    nodes.sort(key=lambda x: x["degree"], reverse=True)
    hubs = [nd["field"] for nd in nodes if nd["degree"] >= 3]

    return {
        "fields": fields, "samples": n, "method": method,
        "edges": edges, "nodes": nodes, "hub_variables": hubs,
        "edge_count": len(edges), "density": round(2 * len(edges) / (p * (p - 1)), 4) if p > 1 else 0,
    }


def persistent_topology_dataset(
    rows: List[Dict[str, Any]],
    col_types: Dict[str, str],
    fields_json: str,
    max_dimension: int = 1,
    max_points: Optional[int] = None,
    policy: StatsPolicy = DEFAULT_STATS_POLICY,
) -> Dict[str, Any]:
    """Persistent homology via Vietoris-Rips filtration."""
    if np is None:
        return {"error": "numpy required for persistent topology."}
    try:
        fields = json.loads(fields_json) if fields_json else []
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid fields_json."}
    numeric_cols = [c for c, t in col_types.items() if t == T_NUMERIC]
    if not fields:
        fields = numeric_cols[:10]
    fields = [f for f in fields if f in numeric_cols]
    if len(fields) < 2:
        return {"error": "Need at least 2 fields."}

    mat = extract_numeric_matrix(rows, fields)
    max_pts = max_points or policy.topology_max_points
    if len(mat) > max_pts:
        indices = sorted(random.sample(range(len(mat)), max_pts))
        mat = [mat[i] for i in indices]
    n = len(mat)
    if n < 5:
        return {"error": f"Not enough rows ({n})."}

    x = np.array(mat, dtype=float)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sqrt(np.sum((x[i] - x[j]) ** 2)))
            dists[i, j] = d
            dists[j, i] = d

    # H0: connected components via union-find over edge filtration
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    h0_births = [0.0] * n
    h0_deaths = [float("inf")] * n
    h0_diagram = []
    components = n

    for dist_val, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if union(i, j):
                dying = rj if rank[find(i)] >= rank[rj] else ri
                h0_diagram.append({"birth": 0.0, "death": round(dist_val, 6)})
                components -= 1

    # H1: track beta_1 = edges_in_complex - vertices + components at each threshold
    n_thresholds = policy.topology_n_thresholds
    max_dist = edges[-1][0] if edges else 1.0
    thresholds = [max_dist * (i + 1) / n_thresholds for i in range(n_thresholds)]
    betti_trace = []
    for thresh in thresholds:
        parent_t = list(range(n))
        rank_t = [0] * n

        def find_t(x):
            while parent_t[x] != x:
                parent_t[x] = parent_t[parent_t[x]]
                x = parent_t[x]
            return x

        n_edges = 0
        comps = n
        for d, i, j in edges:
            if d > thresh:
                break
            n_edges += 1
            ri, rj = find_t(i), find_t(j)
            if ri != rj:
                if rank_t[ri] < rank_t[rj]:
                    ri, rj = rj, ri
                parent_t[rj] = ri
                if rank_t[ri] == rank_t[rj]:
                    rank_t[ri] += 1
                comps -= 1
        b0 = comps
        b1 = n_edges - n + comps
        betti_trace.append({"threshold": round(thresh, 4), "b0": b0, "b1": max(b1, 0)})

    h0_diagram.sort(key=lambda x: x["death"] - x["birth"], reverse=True)
    total_persistence = sum(d["death"] - d["birth"] for d in h0_diagram if d["death"] < float("inf"))

    return {
        "fields": fields, "samples": n, "max_dimension": max_dimension,
        "h0_persistence": h0_diagram[:20],
        "betti_trace": betti_trace,
        "total_persistence_h0": round(total_persistence, 4),
        "final_components": betti_trace[-1]["b0"] if betti_trace else 1,
        "max_b1": max((bt["b1"] for bt in betti_trace), default=0),
    }
