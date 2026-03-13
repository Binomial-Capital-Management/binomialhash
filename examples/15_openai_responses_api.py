"""OpenAI Responses API — the current recommended tool-use flow.

Unlike Chat Completions (legacy), the Responses API:
  - Uses `client.responses.create()` with `input=` (not `messages=`)
  - Returns `output` items with `type: "function_call"` (not tool_calls)
  - Continues via `function_call_output` items with `call_id`
  - Supports namespaces, tool search, and strict mode natively

This example runs a real multi-turn conversation with GPT where BH
ingests market data and the model queries it through BH tools.

Requires: pip install openai binomialhash
Set OPENAI_API_KEY to run.
"""

import json
import os
import random
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call


# ---------------------------------------------------------------------------
# Simulated data source
# ---------------------------------------------------------------------------

def fetch_portfolio_data() -> str:
    """Return a large JSON payload of portfolio holdings and metrics."""
    random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
               "JPM", "GS", "BAC", "JNJ", "PFE", "UNH", "XOM", "CVX"]
    sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
               "META": "Tech", "NVDA": "Tech", "TSLA": "Tech", "JPM": "Finance",
               "GS": "Finance", "BAC": "Finance", "JNJ": "Health", "PFE": "Health",
               "UNH": "Health", "XOM": "Energy", "CVX": "Energy"}
    rows = []
    for ticker in tickers:
        for day in range(1, 31):
            rows.append({
                "ticker": ticker, "sector": sectors[ticker],
                "date": f"2025-03-{day:02d}",
                "price": round(random.uniform(80, 900), 2),
                "volume": random.randint(3_000_000, 80_000_000),
                "pe_ratio": round(random.uniform(10, 70), 2),
                "dividend_yield": round(random.uniform(0, 4), 3),
                "return_1d": round(random.gauss(0.001, 0.02), 6),
                "implied_vol": round(random.uniform(0.15, 0.55), 4),
                "beta": round(random.uniform(0.5, 2.0), 3),
            })
    return json.dumps(rows)


def ingest_market_data() -> str:
    """Fetch and ingest through BH."""
    raw = fetch_portfolio_data()
    bh = get_binomial_hash()
    return bh.ingest(raw, "portfolio_holdings")


# ---------------------------------------------------------------------------
# Build Responses API tool definitions
# ---------------------------------------------------------------------------

def build_tools(bh: BinomialHash):
    specs = get_all_tools(bh)
    openai_tools = get_openai_tools(specs, format="responses")

    data_tool = {
        "type": "function",
        "name": "fetch_portfolio_data",
        "description": "Fetch current portfolio holdings with daily price, volume, and risk metrics.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    return [data_tool] + openai_tools, specs


def dispatch_tool(specs, name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "fetch_portfolio_data":
        return ingest_market_data()
    result = handle_openai_tool_call(specs, name, arguments)
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Responses API agent loop
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You are a portfolio analyst. You have access to a portfolio data tool
and BinomialHash tools for querying stored data.

When you receive a [BH] summary, use bh_retrieve, bh_aggregate, bh_query,
bh_group_by, bh_regress, or bh_dependency_screen to answer analytically.

Give concise, data-backed answers. Show key numbers."""


def run_responses_loop(model: str = "gpt-5.4"):
    """Run a full Responses API loop with tool calling."""

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — cannot run live test.")
        print("Set it and re-run, or use 01_openai_agent_loop.py --mock for offline demo.")
        return

    client = OpenAI()
    bh = init_binomial_hash()
    tools, specs = build_tools(bh)

    print(f"=== OpenAI Responses API — Live ({model}) ===\n")
    print(f"Registered {len(tools)} tools ({len(specs)} BH + 1 custom)\n")

    user_messages = [
        "Fetch my portfolio data and tell me which sectors have the highest average implied volatility.",
        "Now run a regression: what drives return_1d? Use beta, implied_vol, pe_ratio, and volume as drivers.",
        "Export the top 10 stocks by volume as a markdown table with ticker, price, volume, and sector.",
    ]

    input_list = []

    for turn, user_msg in enumerate(user_messages):
        print(f"[Turn {turn + 1}] User: {user_msg}\n")
        input_list.append({"role": "user", "content": user_msg})

        for step in range(8):
            response = client.responses.create(
                model=model,
                instructions=INSTRUCTIONS,
                tools=tools,
                input=input_list,
            )

            input_list += response.output

            has_tool_calls = False
            for item in response.output:
                if item.type == "function_call":
                    has_tool_calls = True
                    print(f"  [Tool] {item.name}({item.arguments[:80]}{'...' if len(item.arguments) > 80 else ''})")
                    result = dispatch_tool(specs, item.name, item.arguments)
                    print(f"         → {len(result)} chars")
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    })

            if not has_tool_calls:
                print(f"  [Assistant] {response.output_text[:500]}")
                if len(response.output_text) > 500:
                    print(f"  ... ({len(response.output_text)} total chars)")
                print()
                break

    stats = bh.context_stats()
    print(f"=== Budget: {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")
    print(f"=== Tool calls: {stats['tool_calls']} | Datasets: {stats['slots']} | Est tokens: ~{stats['est_tokens_out']:,} ===")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4"
    run_responses_loop(model)
