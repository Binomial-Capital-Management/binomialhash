"""Financial analysis pipeline — realistic quant workflow.

Shows BH used the way Sentry's RiskSpecialistAgent uses it:
  1. Ingest market data from multiple sources
  2. Schema inspection — understand what columns are available
  3. Regression — find what drives returns
  4. Manifold navigation — explore the data surface
  5. Insights — objective-driven analysis
  6. Frontier detection — find regime boundaries
"""

import json
import random
import math

from binomialhash import BinomialHash, init_binomial_hash


def generate_market_data():
    """Generate realistic-ish multi-factor equity data."""
    random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
               "JPM", "GS", "BAC", "JNJ", "PFE", "UNH", "XOM", "CVX"]
    sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
               "META": "Tech", "NVDA": "Tech", "TSLA": "Tech", "JPM": "Finance",
               "GS": "Finance", "BAC": "Finance", "JNJ": "Health", "PFE": "Health",
               "UNH": "Health", "XOM": "Energy", "CVX": "Energy"}
    rows = []
    for ticker in tickers:
        base_price = random.uniform(50, 500)
        base_vol = random.uniform(0.15, 0.45)
        for month in range(1, 25):
            drift = random.gauss(0.005, 0.03)
            price = base_price * math.exp(drift * month)
            realized_vol = base_vol * random.uniform(0.7, 1.3)
            rows.append({
                "ticker": ticker,
                "sector": sectors[ticker],
                "month": month,
                "price": round(price, 2),
                "return_1m": round(random.gauss(0.008, realized_vol / math.sqrt(12)), 6),
                "realized_vol_1m": round(realized_vol, 4),
                "implied_vol": round(realized_vol * random.uniform(0.9, 1.3), 4),
                "volume_avg_20d": random.randint(5_000_000, 100_000_000),
                "pe_ratio": round(random.uniform(8, 60), 2),
                "momentum_12m": round(random.gauss(0.1, 0.3), 4),
                "beta": round(random.uniform(0.5, 2.0), 3),
                "sharpe_12m": round(random.gauss(0.5, 1.0), 4),
                "max_drawdown": round(random.uniform(-0.5, -0.02), 4),
                "market_cap_b": round(random.uniform(30, 3000), 1),
            })
    return rows


def run_pipeline():
    bh = init_binomial_hash()

    print("=== Financial Analysis Pipeline ===\n")

    # Step 1: Ingest
    rows = generate_market_data()
    raw = json.dumps(rows)
    summary = bh.ingest(raw, "equity_factors")
    key = bh.keys()[0]["key"]
    print(f"[1] Ingested {len(rows)} rows ({len(raw):,} chars → {len(summary)} char summary)\n")

    # Step 2: Schema inspection
    schema = bh.schema(key)
    print(f"[2] Schema: {schema['row_count']} rows, {len(schema['columns'])} columns")
    numeric_cols = [c for c, info in schema["col_stats"].items()
                    if schema["col_types"].get(c) == "numeric" and "mean" in info]
    print(f"    Numeric columns: {numeric_cols[:8]}...")
    for col in ["return_1m", "realized_vol_1m", "sharpe_12m"]:
        s = schema["col_stats"].get(col, {})
        print(f"    {col:20s} mean={s.get('mean', '?'):>10}  std={s.get('std', '?'):>10}  "
              f"min={s.get('min', '?'):>10}  max={s.get('max', '?'):>10}")

    # Step 3: Regression — what drives returns?
    print(f"\n[3] Regression: what drives return_1m?")
    regression = bh.regress(
        key, "return_1m",
        json.dumps(["realized_vol_1m", "momentum_12m", "beta", "pe_ratio", "market_cap_b"]),
    )
    print(f"    R² = {regression.get('r2', regression.get('r_squared', 'N/A'))}")
    for coef in regression.get("drivers", regression.get("coefficients", [])):
        print(f"    {coef['driver']:20s}  β={coef['coefficient']:+.6f}  corr={coef.get('individual_correlation', 0):+.4f}")

    # Step 4: Dependency screen — broader search
    print(f"\n[4] Dependency screen: rank all drivers of sharpe_12m")
    screen = bh.dependency_screen(
        key, "sharpe_12m",
        json.dumps(["return_1m", "realized_vol_1m", "implied_vol", "momentum_12m",
                     "beta", "pe_ratio", "max_drawdown", "market_cap_b"]),
        "[]", 5,
    )
    for rank in screen.get("ranked_candidates", screen.get("rankings", [])):
        name = rank.get("candidate", rank.get("field", "?"))
        print(f"    {name:20s} raw_corr={rank.get('raw_correlation', 0):+.4f}  "
              f"partial={rank.get('partial_correlation', 0):+.4f}  coef={rank.get('coefficient', 0):+.6f}")

    # Step 5: Manifold navigation
    print(f"\n[5] Manifold state")
    state = bh.manifold_state(key)
    manifold = state.get("manifold", {})
    axes = manifold.get("axes", [])
    print(f"    Axes: {[a.get('column', a.get('name')) for a in axes[:5]]}")
    print(f"    Grid points: {manifold.get('grid_point_count', '?')}")
    print(f"    Faces: {manifold.get('face_count', '?')}")

    # Navigate to a specific point
    if axes:
        first_axis = axes[0]
        axis_name = first_axis.get("column", first_axis.get("name", ""))
        coord = {axis_name: str(first_axis.get("size", 1))}
        print(f"\n    Navigate to {coord}")
        nav = bh.manifold_navigate(key, json.dumps(coord), "return_1m")
        pos = nav.get("position", {})
        print(f"    Field values: {list(pos.get('field_values', {}).keys())[:6]}")
        print(f"    Surface: {nav.get('surface', 'N/A')}")
        steepest = nav.get("steepest_move", {})
        if steepest:
            print(f"    Steepest: {steepest.get('direction', '?')} (Δ={steepest.get('delta', 0):+.6f})")

    # Step 6: Walk an axis
    print(f"\n[6] Walk axis: how does return_1m change across sectors?")
    walk = bh.controlled_walk(key, "sector", "return_1m")
    for step in walk.get("profile", []):
        val = step.get("axis_value", "?")
        mean = step.get("mean", 0)
        n = step.get("points", 0)
        print(f"    {str(val):12s} mean_return={mean:+.6f}  (n={n})")
    sensitivity = walk.get("sensitivity", 0)
    print(f"    Sensitivity: {sensitivity:.6f}")

    # Step 7: Objective-driven insights
    print(f"\n[7] Manifold insights: maximize return_1m")
    insights = bh.manifold_insights(
        key,
        json.dumps({"target": "return_1m", "goal": "maximize", "target_value": 0.02}),
        top_k=3,
    )
    if "surprises" in insights:
        print(f"    Surprises (contradictions to the discovered law):")
        for s in insights["surprises"][:3]:
            print(f"      {s}")
    if "counterfactual" in insights:
        cf = insights["counterfactual"]
        print(f"    Counterfactual: {cf}")

    # Step 8: Conditional slice — how does the manifold differ in high-vol regime?
    print(f"\n[8] Conditional slice: realized_vol > 0.30")
    slice_result = bh.manifold_slice(
        key,
        json.dumps({"column": "realized_vol_1m", "op": ">", "value": 0.30}),
        "return_1m",
    )
    sg = slice_result.get("graph", {})
    print(f"    Surface: {slice_result.get('surface', '?')}")
    print(f"    Graph: V={sg.get('V', '?')}, E={sg.get('E', '?')}")
    print(f"    Topology: {slice_result.get('product_topology_label', '?')}")

    # Budget summary
    stats = bh.context_stats()
    print(f"\n=== Pipeline budget ===")
    print(f"  Tool calls: {stats['tool_calls']}")
    print(f"  Raw input:  {stats['chars_in_raw']:,} chars")
    print(f"  LLM output: {stats['chars_out_to_llm']:,} chars")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Memory: {stats['mem_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    run_pipeline()
