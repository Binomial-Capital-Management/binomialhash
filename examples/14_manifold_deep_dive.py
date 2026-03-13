"""Manifold deep dive — full surface exploration.

The manifold is BH's most unique feature. It builds a discrete geometric
surface from tabular data, enabling operations you can't do with SQL:
  - Navigate: inspect a point and its neighbors on the data surface
  - Geodesic: find the shortest path between two points
  - Walk: sweep an axis and measure target sensitivity
  - Orbit: survey a neighborhood at a given radius
  - Frontier: find structural boundaries between regimes
  - Basin: trace downhill to a local minimum
  - Ridge/Valley: trace structural corridors
  - Insights: objective-driven analysis (surprises, regime shifts, counterfactuals)

This example uses options data — IV surface navigation is a natural fit.
"""

import json
import random
import math

from binomialhash import BinomialHash, init_binomial_hash


def generate_iv_surface():
    """Generate a synthetic implied volatility surface."""
    random.seed(42)
    rows = []
    spot = 200.0
    for strike in range(150, 260, 5):
        for days_to_expiry in [7, 14, 30, 60, 90, 180, 365]:
            moneyness = strike / spot
            base_iv = 0.20
            skew = 0.15 * max(0, 1.0 - moneyness) ** 2
            term = 0.05 * math.log(max(days_to_expiry, 1) / 30)
            smile = 0.08 * (moneyness - 1.0) ** 2
            iv = base_iv + skew + term + smile + random.gauss(0, 0.01)
            iv = max(iv, 0.05)

            rows.append({
                "strike": strike,
                "days_to_expiry": days_to_expiry,
                "moneyness": round(moneyness, 4),
                "implied_vol": round(iv, 6),
                "delta": round(max(-1, min(1, 0.5 + (spot - strike) / (spot * iv * math.sqrt(days_to_expiry / 365)))), 4),
                "gamma": round(random.uniform(0.001, 0.05), 6),
                "theta": round(-iv * spot * 0.01 / math.sqrt(max(days_to_expiry, 1)), 4),
                "vega": round(spot * math.sqrt(days_to_expiry / 365) * 0.01, 4),
                "open_interest": random.randint(100, 50000),
                "volume": random.randint(10, 5000),
                "bid_ask_spread": round(random.uniform(0.01, 0.15), 4),
            })
    return rows


def run_demo():
    bh = init_binomial_hash()

    print("=== Manifold Deep Dive: IV Surface ===\n")

    rows = generate_iv_surface()
    raw = json.dumps(rows)
    summary = bh.ingest(raw, "iv_surface")
    key = bh.keys()[0]["key"]
    print(f"Ingested {len(rows)} grid points ({len(raw):,} chars)\n")

    # 1. Manifold state — what did BH discover?
    print("[1] Manifold State")
    state = bh.manifold_state(key)
    m = state.get("manifold", {})
    axes = m.get("axes", [])
    for ax in axes:
        col = ax.get("column", ax.get("name", "?"))
        print(f"  Axis: {col:20s} size={ax.get('size', '?')}, "
              f"type={ax.get('type', '?')}, wraps={ax.get('wraps', '?')}")
    graph = m.get("graph", {})
    print(f"  Grid points: {graph.get('nodes', m.get('grid_point_count', '?'))}")
    print(f"  Faces: {graph.get('faces', m.get('face_count', '?'))}")
    print(f"  Surface: {m.get('surface', m.get('surface_label', '?'))}")

    # 2. Navigate to ATM 30-day options
    print(f"\n[2] Navigate: strike=200, days_to_expiry=30")
    nav = bh.manifold_navigate(
        key, json.dumps({"strike": "200", "days_to_expiry": "30"}), "implied_vol"
    )
    pos = nav.get("position", {})
    fv = pos.get("field_values", {})
    print(f"  implied_vol: {fv.get('implied_vol', '?')}")
    print(f"  delta: {fv.get('delta', '?')}")
    print(f"  coord: {pos.get('coord', '?')}")
    print(f"  Surface: {nav.get('surface', '?')}")
    gradients = nav.get("gradients", {})
    if isinstance(gradients, dict) and gradients:
        print(f"  Gradients ({len(gradients)} neighbors):")
        for direction, ginfo in list(gradients.items())[:4]:
            delta = ginfo.get("delta", 0)
            print(f"    → {direction:35s} Δ={delta:+.6f}")
    steepest = nav.get("steepest_move", {})
    if steepest:
        print(f"  Steepest: {steepest.get('direction', '?')} (Δ={steepest.get('delta', 0):+.6f})")

    # 3. Walk the strike axis — volatility smile
    print(f"\n[3] Walk: strike axis (volatility smile)")
    walk = bh.controlled_walk(key, "strike", "implied_vol")
    profile = walk.get("profile", [])
    for step in profile[:8]:
        mean = step.get("mean", 0)
        bar = "█" * int(mean * 100)
        print(f"  strike={str(step.get('axis_value', '?')):>5s}  IV={mean:.4f}  {bar}")
    if len(profile) > 8:
        print(f"  ... ({len(profile)} total steps)")
    print(f"  Sensitivity: {walk.get('sensitivity', 0):.6f}")

    # 4. Walk the term axis — term structure
    print(f"\n[4] Walk: days_to_expiry axis (term structure)")
    walk_term = bh.controlled_walk(key, "days_to_expiry", "implied_vol")
    for step in walk_term.get("profile", []):
        mean = step.get("mean", 0)
        bar = "█" * int(mean * 100)
        print(f"  DTE={str(step.get('axis_value', '?')):>5s}  IV={mean:.4f}  {bar}")

    # 5. Geodesic — shortest path from deep ITM to deep OTM
    print(f"\n[5] Geodesic: strike=160 → strike=240 (weighted by IV)")
    geo = bh.geodesic(
        key,
        json.dumps({"strike": "160", "days_to_expiry": "30"}),
        json.dumps({"strike": "240", "days_to_expiry": "30"}),
        "implied_vol",
    )
    waypoints = geo.get("waypoints", [])
    print(f"  Hops: {geo.get('hops', '?')}, total cost: {geo.get('total_cost', 0):.4f}")
    for wp in waypoints[:6]:
        coord = wp.get("coord", ())
        fields = wp.get("fields", {})
        iv = fields.get("implied_vol", 0)
        delta_from = wp.get("delta_from_prev", {})
        d_iv = delta_from.get("implied_vol", 0) if isinstance(delta_from, dict) else 0
        print(f"    coord={coord}  IV={iv:.4f}  Δiv={d_iv:+.6f}")
    if len(waypoints) > 6:
        print(f"    ... ({len(waypoints)} total)")

    # 6. Orbit — survey neighborhood around ATM
    print(f"\n[6] Orbit: radius=2 around strike=200, DTE=30")
    orbit = bh.orbit(
        key, json.dumps({"strike": "200", "days_to_expiry": "30"}),
        radius=2, target_field="implied_vol",
    )
    center = orbit.get("center_summary", {})
    zone = orbit.get("zone_summary", {})
    print(f"  Center: IV mean={center.get('mean_value', '?')}, density={center.get('mean_density', '?')}")
    print(f"  Zone:   IV mean={zone.get('mean_value', '?')}, points={zone.get('points', '?')}")
    shells = orbit.get("shell_profiles", [])
    for shell in shells:
        print(f"    distance={shell.get('distance', '?')}: IV={shell.get('mean_value', 0):.4f}, "
              f"points={shell.get('points', 0)}, curvature={shell.get('mean_curvature', 0):.4f}")

    # 7. Frontier — boundary between high-IV and low-IV regimes
    print(f"\n[7] Frontier: implied_vol > 0.25 vs. implied_vol <= 0.25")
    frontier = bh.frontier(
        key,
        json.dumps({"column": "implied_vol", "op": ">", "value": 0.25}),
        json.dumps({"column": "implied_vol", "op": "<=", "value": 0.25}),
        "implied_vol", 10,
    )
    edges = frontier.get("frontier_edges", [])
    print(f"  Frontier edges: {len(edges)}")
    for e in edges[:4]:
        fr = e.get("from", [])
        to = e.get("to", [])
        jump = e.get("target_jump", 0)
        labels = e.get("labels", [])
        print(f"    {fr} → {to}  ΔIV={jump:+.6f}  labels={labels}")

    # 8. Conditional slice — high DTE regime
    print(f"\n[8] Conditional slice: days_to_expiry > 90")
    slice_r = bh.manifold_slice(
        key,
        json.dumps({"column": "days_to_expiry", "op": ">", "value": 90}),
        "implied_vol",
    )
    slice_graph = slice_r.get("graph", {})
    print(f"  Surface: {slice_r.get('surface', '?')}")
    print(f"  Graph: V={slice_graph.get('V', '?')}, E={slice_graph.get('E', '?')}")
    print(f"  Topology: {slice_r.get('product_topology_label', '?')}")

    # 9. Coverage audit
    print(f"\n[9] Coverage audit")
    audit = bh.coverage_audit(key)
    graph_info = audit.get("graph", {})
    density = audit.get("density", {})
    confidence = audit.get("surface_confidence", {})
    print(f"  Graph: {graph_info.get('vertices', '?')} vertices, {graph_info.get('edges', '?')} edges, "
          f"{graph_info.get('components', '?')} components")
    print(f"  Density: mean={density.get('mean', '?')}, min={density.get('min', '?')}, max={density.get('max', '?')}")
    print(f"  Has faces: {confidence.get('has_faces', '?')}")

    # 10. Insights
    print(f"\n[10] Objective insights: minimize implied_vol")
    insights = bh.manifold_insights(
        key,
        json.dumps({"target": "implied_vol", "goal": "minimize", "target_value": 0.15}),
        top_k=3,
    )
    for section in ["best_driver", "surprises", "regime_boundaries", "counterfactual"]:
        val = insights.get(section)
        if val:
            print(f"  {section}: {json.dumps(val, default=str)[:150]}")

    stats = bh.context_stats()
    print(f"\n=== {stats['tool_calls']} tool calls | {stats['compression_ratio']:.1f}x compression | "
          f"{stats['mem_bytes'] / 1024:.0f}KB memory ===")


if __name__ == "__main__":
    run_demo()
