"""Anthropic Claude tool-use loop with BinomialHash.

Shows the full cycle:
  1. Register BH tools as Anthropic tool definitions
  2. Claude returns tool_use content blocks
  3. Dispatch through the Anthropic adapter
  4. Return tool_result blocks back to Claude

Requires: pip install anthropic binomialhash
Set ANTHROPIC_API_KEY or pass --mock.
"""

import argparse
import json
import os

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools, get_tools_by_group
from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use


# ---------------------------------------------------------------------------
# Simulated data source
# ---------------------------------------------------------------------------

def fetch_earnings(ticker: str) -> str:
    """Simulate an earnings API returning structured results."""
    import random
    random.seed(hash(ticker) % 2**31)
    quarters = []
    for year in range(2020, 2026):
        for q in range(1, 5):
            quarters.append({
                "ticker": ticker,
                "quarter": f"Q{q} {year}",
                "revenue_m": round(random.uniform(20_000, 120_000), 1),
                "eps": round(random.uniform(0.5, 8.0), 2),
                "eps_estimate": round(random.uniform(0.5, 8.0), 2),
                "gross_margin": round(random.uniform(0.3, 0.7), 4),
                "operating_margin": round(random.uniform(0.1, 0.4), 4),
                "guidance_revenue_m": round(random.uniform(20_000, 130_000), 1),
            })
    return json.dumps(quarters)


# ---------------------------------------------------------------------------
# Build tools
# ---------------------------------------------------------------------------

def build_tools(bh: BinomialHash):
    """BH retrieval + stats tools formatted for Anthropic."""
    retrieval = get_tools_by_group(bh, "retrieval")
    stats = get_tools_by_group(bh, "stats")
    specs = retrieval + stats

    anthropic_tools = get_anthropic_tools(specs)

    anthropic_tools.insert(0, {
        "name": "fetch_earnings",
        "description": "Fetch quarterly earnings data for a company.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    })
    return anthropic_tools, specs


def handle_tool(specs, name, input_dict):
    if name == "fetch_earnings":
        raw = fetch_earnings(input_dict["ticker"])
        return get_binomial_hash().ingest(raw, f"earnings_{input_dict['ticker']}")
    return json.dumps(handle_anthropic_tool_use(specs, name, input_dict), default=str)


# ---------------------------------------------------------------------------
# Mock loop
# ---------------------------------------------------------------------------

def run_mock():
    bh = init_binomial_hash()
    tools, specs = build_tools(bh)

    print("=== Anthropic mock loop ===\n")
    print(f"Registered {len(tools)} tools ({len(specs)} BH + 1 custom)\n")

    # Simulate: Claude calls fetch_earnings
    print("[Claude] tool_use: fetch_earnings({ticker: 'AAPL'})")
    result = handle_tool(specs, "fetch_earnings", {"ticker": "AAPL"})
    print(f"[BH] Ingested → {len(result)} chars (raw was ~5KB)\n")

    # Simulate: Claude calls bh_regress to find EPS drivers
    key = bh.keys()[0]["key"]
    print(f"[Claude] tool_use: bh_regress(target=eps, drivers=[revenue_m, gross_margin])")
    regression = handle_tool(specs, "bh_regress", {
        "key": key,
        "target": "eps",
        "drivers_json": '["revenue_m", "gross_margin", "operating_margin"]',
    })
    parsed = json.loads(regression)
    print(f"[Result] R²={parsed.get('r2', parsed.get('r_squared', 'N/A'))}")
    for coef in parsed.get("drivers", parsed.get("coefficients", [])):
        print(f"  {coef['driver']:20s}  β={coef['coefficient']:+.6f}  corr={coef.get('individual_correlation', 0):+.4f}")

    print(f"\n[Claude] tool_use: bh_aggregate(column=eps, func=mean)")
    agg = handle_tool(specs, "bh_aggregate", {"key": key, "column": "eps", "func": "mean"})
    print(f"[Result] {agg}")

    stats = bh.context_stats()
    print(f"\n=== {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")

    # Show what the Anthropic tool schema looks like
    print("\n--- Sample Anthropic tool definition ---")
    print(json.dumps(tools[1], indent=2)[:500])


def run_live():
    from anthropic import Anthropic

    client = Anthropic()
    bh = init_binomial_hash()
    tools, specs = build_tools(bh)

    messages = [{"role": "user", "content": "Analyze AAPL earnings trends. What drives EPS?"}]

    print("=== Live Anthropic loop ===\n")
    for turn in range(8):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                print(f"[Turn {turn}] {block.name}({json.dumps(block.input)[:80]}...)")
                result = handle_tool(specs, block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                print(f"  → {len(result)} chars")
            elif block.type == "text":
                print(f"\n[Claude] {block.text[:500]}")

        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    stats = bh.context_stats()
    print(f"\n=== {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or not os.environ.get("ANTHROPIC_API_KEY"):
        run_mock()
    else:
        run_live()
