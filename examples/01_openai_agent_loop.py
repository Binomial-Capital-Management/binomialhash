"""OpenAI Chat Completions tool-use loop with BinomialHash.

Shows a realistic multi-turn conversation where:
  1. User asks about market data
  2. A simulated MCP tool returns 500+ rows of JSON
  3. BH ingests it → LLM sees a compact summary (~300 chars instead of ~50KB)
  4. LLM calls bh_retrieve / bh_aggregate to answer the question
  5. Follow-up turns query the same stored data without re-fetching

Requires: pip install openai binomialhash
Set OPENAI_API_KEY or pass --mock to run with simulated responses.
"""

import argparse
import json
import os
import sys

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call


# ---------------------------------------------------------------------------
# Simulated MCP data source (replace with real API call in production)
# ---------------------------------------------------------------------------

def fetch_market_data(ticker: str) -> str:
    """Simulate an MCP server returning a large JSON payload."""
    import random
    random.seed(42)
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials",
               "GS": "Financials", "JNJ": "Healthcare", "PFE": "Healthcare",
               "XOM": "Energy", "CVX": "Energy", "AMZN": "Technology", "GOOGL": "Technology"}
    rows = []
    for t, sector in sectors.items():
        for day in range(1, 51):
            rows.append({
                "ticker": t,
                "date": f"2025-{(day // 30) + 1:02d}-{(day % 28) + 1:02d}",
                "close": round(random.uniform(100, 500), 2),
                "volume": random.randint(5_000_000, 80_000_000),
                "change_pct": round(random.uniform(-5, 5), 4),
                "sector": sector,
                "market_cap_b": round(random.uniform(50, 3000), 1),
            })
    return json.dumps(rows)


# ---------------------------------------------------------------------------
# BH-aware tool that ingests raw data before returning to the LLM
# ---------------------------------------------------------------------------

def market_data_tool(ticker: str) -> str:
    """Fetch market data and ingest through BH."""
    raw = fetch_market_data(ticker)
    bh = get_binomial_hash()
    return bh.ingest(raw, f"market_{ticker}")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM = """You are a financial analyst assistant. You have access to market data
tools and BinomialHash tools for querying stored datasets.

When you receive a [BH] summary, use bh_retrieve, bh_aggregate, bh_query, or
bh_group_by to answer the user's question from the stored data."""


def build_tools(bh: BinomialHash):
    """Combine the custom data tool with all BH tools."""
    specs = get_all_tools(bh)
    openai_tools = get_openai_tools(specs, format="chat_completions")

    fetch_tool = {
        "type": "function",
        "function": {
            "name": "fetch_market_data",
            "description": "Fetch historical market data for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
                "required": ["ticker"],
            },
        },
    }
    return [fetch_tool] + openai_tools, specs


def handle_tool(specs, name, arguments):
    """Route a tool call to the right handler."""
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    if name == "fetch_market_data":
        return market_data_tool(args["ticker"])
    return json.dumps(handle_openai_tool_call(specs, name, arguments), default=str)


def run_mock_loop():
    """Demonstrate the flow with mock LLM responses."""
    bh = init_binomial_hash()
    tools, specs = build_tools(bh)

    print("=== Mock agent loop (no API key needed) ===\n")

    # Turn 1: user asks, we simulate fetching data
    print("[User] Show me the top tech stocks by volume")
    print("[System] Calling fetch_market_data('AAPL')...")
    result = handle_tool(specs, "fetch_market_data", '{"ticker": "AAPL"}')
    print(f"[BH] Ingested → {len(result)} chars to LLM (raw was ~50KB)\n")

    # Turn 2: simulate LLM calling bh_group_by
    key = bh.keys()[0]["key"]
    print(f"[LLM] Calling bh_group_by(key={key}, group_cols='[\"sector\"]', ...)")
    group_result = handle_tool(
        specs, "bh_group_by",
        json.dumps({
            "key": key,
            "group_cols": '["sector"]',
            "agg_json": '[{"column": "volume", "func": "sum", "alias": "total_volume"}]',
            "sort_by": "total_volume",
            "sort_desc": True,
            "limit": 5,
        }),
    )
    print(f"[Result] {group_result[:300]}...\n")

    # Turn 3: drill into Technology sector
    print("[LLM] Calling bh_query(where: sector = Technology, sort_by: volume desc)...")
    query_result = handle_tool(
        specs, "bh_query",
        json.dumps({
            "key": key,
            "where_json": '{"column": "sector", "op": "=", "value": "Technology"}',
            "sort_by": "volume",
            "sort_desc": True,
            "limit": 5,
            "columns": '["ticker", "date", "close", "volume"]',
        }),
    )
    print(f"[Result] {query_result[:400]}...\n")

    stats = bh.context_stats()
    print("=== Context budget ===")
    print(f"  Raw input:  {stats['chars_in_raw']:,} chars")
    print(f"  LLM output: {stats['chars_out_to_llm']:,} chars")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Est tokens:  ~{stats['est_tokens_out']:,}")


def run_live_loop():
    """Run with real OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    bh = init_binomial_hash()
    tools, specs = build_tools(bh)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What sectors have the highest trading volume? Fetch market data first."},
    ]

    print("=== Live OpenAI agent loop ===\n")
    for turn in range(6):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=messages,
            tools=tools,
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                print(f"[Turn {turn}] Tool call: {tc.function.name}({tc.function.arguments[:80]}...)")
                result = handle_tool(specs, tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                print(f"  → {len(result)} chars returned to LLM")
        else:
            print(f"\n[Assistant] {choice.message.content}")
            break

    stats = bh.context_stats()
    print(f"\n=== Budget: {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with simulated LLM responses")
    args = parser.parse_args()

    if args.mock or not os.environ.get("OPENAI_API_KEY"):
        run_mock_loop()
    else:
        run_live_loop()
