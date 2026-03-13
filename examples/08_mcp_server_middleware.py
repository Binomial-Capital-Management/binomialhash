"""MCP server middleware — auto-intercept tool outputs with zero code changes.

Production pattern: you have existing MCP server tools that return large JSON.
Instead of modifying every tool function, wrap them with @bh_intercept or
wrap_tool_with_bh. BH automatically ingests large outputs and returns compact
summaries.

This is how Sentry's 11 MCP servers integrate with BH:
  @bh_intercept(label="fmp_market_data")
  async def get_stock_data(ticker: str) -> str:
      return await fmp_client.fetch(...)  # 50KB JSON → 300 char summary
"""

import asyncio
import json
import random

from binomialhash import init_binomial_hash, get_binomial_hash
from binomialhash.middleware import bh_intercept, wrap_tool_with_bh, raw_mode
from binomialhash.context import bh_raw_mode


# ---------------------------------------------------------------------------
# Existing MCP server tools (unchanged — no BH awareness)
# ---------------------------------------------------------------------------

def fmp_get_financial_statements(ticker: str, period: str = "annual") -> str:
    """Simulates FMP API: returns income statements, balance sheets, cash flows."""
    random.seed(hash(ticker))
    rows = []
    for year in range(2015, 2026):
        rows.append({
            "ticker": ticker, "year": year, "period": period,
            "revenue": round(random.uniform(50_000, 400_000), 1),
            "cost_of_revenue": round(random.uniform(25_000, 200_000), 1),
            "gross_profit": round(random.uniform(20_000, 200_000), 1),
            "operating_income": round(random.uniform(5_000, 100_000), 1),
            "net_income": round(random.uniform(3_000, 80_000), 1),
            "total_assets": round(random.uniform(100_000, 500_000), 1),
            "total_liabilities": round(random.uniform(50_000, 300_000), 1),
            "total_equity": round(random.uniform(30_000, 200_000), 1),
            "operating_cash_flow": round(random.uniform(5_000, 120_000), 1),
            "capex": round(random.uniform(1_000, 30_000), 1),
            "free_cash_flow": round(random.uniform(2_000, 90_000), 1),
            "shares_outstanding": random.randint(1_000, 20_000),
            "eps": round(random.uniform(1, 25), 2),
            "dividend_per_share": round(random.uniform(0, 5), 2),
        })
    return json.dumps(rows)


async def yfinance_get_historical(ticker: str, days: int = 252) -> str:
    """Simulates Yahoo Finance: returns daily OHLCV data (async)."""
    random.seed(hash(ticker) + 1)
    rows = []
    price = random.uniform(100, 500)
    for d in range(days):
        change = random.gauss(0, 2)
        price = max(price + change, 10)
        rows.append({
            "ticker": ticker, "day": d,
            "open": round(price + random.uniform(-1, 1), 2),
            "high": round(price + random.uniform(0, 3), 2),
            "low": round(price - random.uniform(0, 3), 2),
            "close": round(price, 2),
            "volume": random.randint(5_000_000, 80_000_000),
            "adj_close": round(price * 0.99, 2),
        })
    return json.dumps(rows)


def sec_get_filing_chunks(ticker: str, filing_type: str = "10-K") -> str:
    """Simulates SEC filing retrieval: returns text chunks."""
    sections = ["Risk Factors", "Business Description", "MD&A",
                 "Financial Statements", "Notes to Financial Statements"]
    chunks = []
    for i, section in enumerate(sections):
        chunks.append({
            "ticker": ticker, "filing_type": filing_type,
            "section": section, "chunk_index": i,
            "text": f"[{section}] " + " ".join(
                random.choice(["revenue", "growth", "risk", "market", "operating",
                              "the", "company", "increased", "decreased", "fiscal"])
                for _ in range(200)
            ),
            "token_count": random.randint(150, 300),
        })
    return json.dumps(chunks)


# ---------------------------------------------------------------------------
# Method 1: @bh_intercept decorator (recommended)
# ---------------------------------------------------------------------------

@bh_intercept(label="fmp_financials")
def get_financials_intercepted(ticker: str) -> str:
    """Same function, decorated. Output auto-compacted if > 3000 chars."""
    return fmp_get_financial_statements(ticker)


@bh_intercept(label="yfinance_hist")
async def get_historical_intercepted(ticker: str) -> str:
    """Async function — decorator handles both sync and async."""
    return await yfinance_get_historical(ticker)


@bh_intercept(label="sec_filing")
def get_filing_intercepted(ticker: str) -> str:
    return sec_get_filing_chunks(ticker)


# ---------------------------------------------------------------------------
# Method 2: wrap_tool_with_bh (for third-party functions you can't decorate)
# ---------------------------------------------------------------------------

get_financials_wrapped = wrap_tool_with_bh(
    fmp_get_financial_statements, label="fmp_financials_wrapped"
)

get_historical_wrapped = wrap_tool_with_bh(
    yfinance_get_historical, label="yfinance_hist_wrapped"
)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    bh = init_binomial_hash()

    print("=== MCP Server Middleware ===\n")

    # Method 1: Decorated functions
    print("--- @bh_intercept decorator ---")
    result = get_financials_intercepted("AAPL")
    print(f"fmp_financials(AAPL):    {len(result):>6} chars to LLM")

    result = asyncio.get_event_loop().run_until_complete(
        get_historical_intercepted("AAPL")
    )
    print(f"yfinance_hist(AAPL):     {len(result):>6} chars to LLM")

    result = get_filing_intercepted("AAPL")
    print(f"sec_filing(AAPL):        {len(result):>6} chars to LLM")

    # Method 2: Wrapped third-party functions
    print("\n--- wrap_tool_with_bh ---")
    result = get_financials_wrapped("MSFT")
    print(f"fmp_wrapped(MSFT):       {len(result):>6} chars to LLM")

    # raw_mode bypass — when an internal tool needs the full payload
    print("\n--- raw_mode bypass ---")
    with raw_mode():
        raw_result = get_financials_intercepted("GOOGL")
        raw_parsed = json.loads(raw_result)
        print(f"raw_mode(GOOGL):         {len(raw_result):>6} chars (full JSON, {len(raw_parsed)} rows)")

    # Also works with bh_raw_mode from context module
    with bh_raw_mode():
        raw_result2 = get_financials_intercepted("AMZN")
        print(f"bh_raw_mode(AMZN):       {len(raw_result2):>6} chars (full JSON)")

    # Summary
    print(f"\n--- BH state ---")
    print(f"Datasets stored: {len(bh.keys())}")
    for k in bh.keys():
        print(f"  {k['key']:30s} {k['row_count']:>4} rows, {len(k.get('columns', []))} cols")

    stats = bh.context_stats()
    print(f"\n=== {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ({stats['compression_ratio']:.1f}x) ===")


if __name__ == "__main__":
    run_demo()
