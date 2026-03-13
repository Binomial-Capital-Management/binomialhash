"""Multi-agent handoff with shared BinomialHash state.

Production pattern: two agents share the same BH instance via contextvars.
  - Agent A (Data Fetcher): calls external APIs, BH ingests the raw responses
  - Agent B (Analyst): queries the stored data via BH tools

The key insight: both agents see the same datasets because they run in the
same request context. Agent B can query data that Agent A stored, without
passing raw payloads between agents.

This is exactly how Sentry's LangGraph orchestrator works:
  Supervisor → GeneralistAgent (fetches) → RiskSpecialist (analyzes)
"""

import json
import random

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_tools_by_group
from binomialhash.adapters.common import handle_tool_call


# ---------------------------------------------------------------------------
# Simulated MCP data sources
# ---------------------------------------------------------------------------

def fetch_options_chain(ticker: str) -> str:
    random.seed(hash(ticker))
    rows = []
    for strike in range(150, 250, 5):
        for opt_type in ["call", "put"]:
            rows.append({
                "ticker": ticker, "strike": strike, "type": opt_type,
                "expiry": "2025-06-20",
                "bid": round(random.uniform(0.5, 30), 2),
                "ask": round(random.uniform(0.6, 31), 2),
                "volume": random.randint(10, 5000),
                "open_interest": random.randint(100, 50000),
                "implied_vol": round(random.uniform(0.15, 0.60), 4),
                "delta": round(random.uniform(-1, 1), 4),
                "gamma": round(random.uniform(0, 0.05), 6),
                "theta": round(random.uniform(-0.5, 0), 4),
                "vega": round(random.uniform(0, 0.5), 4),
            })
    return json.dumps(rows)


def fetch_fundamentals(ticker: str) -> str:
    random.seed(hash(ticker) + 1)
    return json.dumps([{
        "ticker": ticker,
        "metric": m,
        "value": round(random.uniform(0.5, 50), 4),
        "sector_avg": round(random.uniform(0.5, 50), 4),
        "percentile": random.randint(10, 99),
    } for m in [
        "pe_ratio", "pb_ratio", "ev_ebitda", "debt_equity", "roe", "roa",
        "current_ratio", "quick_ratio", "gross_margin", "operating_margin",
        "net_margin", "dividend_yield", "payout_ratio", "revenue_growth",
        "earnings_growth", "free_cash_flow_yield",
    ]])


# ---------------------------------------------------------------------------
# Agent A: Data Fetcher
# ---------------------------------------------------------------------------

class DataFetcherAgent:
    """Fetches raw data from external sources, ingests into BH."""

    def run(self, tickers: list[str]):
        bh = get_binomial_hash()
        results = {}
        for ticker in tickers:
            options_raw = fetch_options_chain(ticker)
            options_summary = bh.ingest(options_raw, f"options_{ticker}")
            results[f"options_{ticker}"] = {
                "raw_chars": len(options_raw),
                "summary_chars": len(options_summary),
            }

            fundamentals_raw = fetch_fundamentals(ticker)
            fundamentals_summary = bh.ingest(fundamentals_raw, f"fundamentals_{ticker}")
            results[f"fundamentals_{ticker}"] = {
                "raw_chars": len(fundamentals_raw),
                "summary_chars": len(fundamentals_summary),
            }
        return results


# ---------------------------------------------------------------------------
# Agent B: Analyst (uses BH tools to query Agent A's data)
# ---------------------------------------------------------------------------

class AnalystAgent:
    """Queries stored BH data to produce analysis. Never sees raw payloads."""

    def __init__(self):
        bh = get_binomial_hash()
        self.retrieval_specs = get_tools_by_group(bh, "retrieval")
        self.stats_specs = get_tools_by_group(bh, "stats")
        self.all_specs = self.retrieval_specs + self.stats_specs

    def analyze_options(self, key: str) -> dict:
        schema = handle_tool_call(self.all_specs, "bh_schema", {"key": key})
        agg_vol = handle_tool_call(self.all_specs, "bh_aggregate", {
            "key": key, "column": "implied_vol", "func": "mean",
        })
        agg_oi = handle_tool_call(self.all_specs, "bh_aggregate", {
            "key": key, "column": "open_interest", "func": "sum",
        })
        return {
            "columns": len(schema.get("columns", [])),
            "rows": schema.get("row_count", 0),
            "avg_implied_vol": agg_vol.get("result"),
            "total_open_interest": agg_oi.get("result"),
        }

    def rank_drivers(self, key: str, target: str) -> dict:
        result = handle_tool_call(self.all_specs, "bh_dependency_screen", {
            "key": key,
            "target": target,
            "candidates_json": json.dumps(["strike", "volume", "open_interest", "delta", "gamma", "theta", "vega"]),
            "top_k": 5,
        })
        candidates = result.get("ranked_candidates", result.get("rankings", []))
        return {"rankings": [
            {"field": c.get("candidate", c.get("field", "?")),
             "raw_corr": c.get("raw_correlation", 0),
             "partial_corr": c.get("partial_correlation", 0)}
            for c in candidates
        ]}


# ---------------------------------------------------------------------------
# Orchestrator: runs both agents in the same context
# ---------------------------------------------------------------------------

def run_demo():
    bh = init_binomial_hash()

    print("=== Multi-Agent Handoff with Shared BH State ===\n")

    # Agent A: fetch data
    fetcher = DataFetcherAgent()
    tickers = ["AAPL", "MSFT", "GOOGL"]
    print(f"[DataFetcher] Fetching options + fundamentals for {tickers}")
    fetch_results = fetcher.run(tickers)
    for dataset, info in fetch_results.items():
        print(f"  {dataset}: {info['raw_chars']:,} → {info['summary_chars']} chars")

    print(f"\n[BH] {len(bh.keys())} datasets stored:")
    for k in bh.keys():
        print(f"  {k['key']}: {k['row_count']} rows, {len(k.get('columns', []))} cols")

    # Agent B: analyze (sees the same BH instance via contextvars)
    analyst = AnalystAgent()
    options_key = next(k["key"] for k in bh.keys() if "options_aapl" in k["key"].lower())

    print(f"\n[Analyst] Analyzing {options_key}")
    analysis = analyst.analyze_options(options_key)
    print(f"  Columns: {analysis['columns']}")
    print(f"  Rows: {analysis['rows']}")
    print(f"  Avg IV: {analysis['avg_implied_vol']:.4f}")
    print(f"  Total OI: {analysis['total_open_interest']:,.0f}")

    print(f"\n[Analyst] Ranking drivers of implied_vol...")
    drivers = analyst.rank_drivers(options_key, "implied_vol")
    for rank in drivers.get("rankings", [])[:5]:
        print(f"  {rank['field']:20s} corr={rank.get('raw_corr', 0):+.4f}  partial={rank.get('partial_corr', 0):+.4f}")

    # Context budget across both agents
    stats = bh.context_stats()
    print(f"\n=== Cross-agent budget ===")
    print(f"  Tool calls: {stats['tool_calls']}")
    print(f"  Raw input:  {stats['chars_in_raw']:,} chars")
    print(f"  LLM output: {stats['chars_out_to_llm']:,} chars")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Datasets:   {stats['slots']}")


if __name__ == "__main__":
    run_demo()
