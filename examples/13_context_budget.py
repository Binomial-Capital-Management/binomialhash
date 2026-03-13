"""Context budget management — monitor and control token usage.

Production pattern: in a multi-turn agent conversation, track how much
context BH is saving. Make budget-aware decisions about when to fetch
more data vs. query existing data.

Shows:
  - context_stats() for real-time budget monitoring
  - Token counting per provider
  - Compression ratio tracking across tools
  - Budget-aware tool selection logic
"""

import json
import random

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tokenizers import count_tokens, is_exact


# ---------------------------------------------------------------------------
# Simulated multi-tool conversation
# ---------------------------------------------------------------------------

def simulate_tool_call(bh: BinomialHash, label: str, row_count: int) -> dict:
    """Simulate a tool returning data of varying sizes."""
    random.seed(hash(label))
    rows = [
        {f"col_{j}": round(random.uniform(0, 1000), 2) for j in range(12)}
        | {"label": label, "idx": i}
        for i in range(row_count)
    ]
    raw = json.dumps(rows)
    summary = bh.ingest(raw, label)
    return {"raw_chars": len(raw), "summary_chars": len(summary), "rows": row_count}


def run_demo():
    bh = init_binomial_hash()

    print("=== Context Budget Management ===\n")

    # Simulate a multi-tool conversation
    tools_called = [
        ("market_data", 500),
        ("options_chain", 300),
        ("earnings_history", 200),
        ("sec_filings", 150),
        ("analyst_estimates", 100),
        ("insider_trades", 80),
    ]

    print("--- Tool calls and compression ---")
    cumulative_raw = 0
    cumulative_llm = 0

    for label, row_count in tools_called:
        result = simulate_tool_call(bh, label, row_count)
        cumulative_raw += result["raw_chars"]
        cumulative_llm += result["summary_chars"]
        ratio = result["raw_chars"] / max(result["summary_chars"], 1)
        print(f"  {label:25s} {result['rows']:>4} rows | "
              f"{result['raw_chars']:>8,} → {result['summary_chars']:>5} chars ({ratio:>5.1f}x) | "
              f"cumulative: {cumulative_llm:,} chars to LLM")

    # Real-time stats
    stats = bh.context_stats()
    print(f"\n--- context_stats() ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.2f}")
        elif isinstance(v, int):
            print(f"  {k:25s} {v:,}")
        else:
            print(f"  {k:25s} {v}")

    # Token counting per provider
    print(f"\n--- Token estimates by provider ---")
    sample_text = json.dumps(bh.schema(bh.keys()[0]["key"]), default=str)

    for provider in ["openai", "anthropic", "gemini", "xai"]:
        tokens = count_tokens(sample_text, provider=provider)
        exact = is_exact(provider)
        method = "exact (tiktoken)" if exact else "heuristic (chars/4)"
        print(f"  {provider:12s} {tokens:>6,} tokens  [{method}]")

    # Budget-aware decisions
    print(f"\n--- Budget-aware tool selection ---")
    TOKEN_BUDGET = 8000
    current_tokens = stats["est_tokens_out"]

    print(f"  Budget: {TOKEN_BUDGET:,} tokens")
    print(f"  Used:   {current_tokens:,} tokens ({current_tokens / TOKEN_BUDGET * 100:.0f}%)")
    print(f"  Remaining: {TOKEN_BUDGET - current_tokens:,} tokens")

    remaining = TOKEN_BUDGET - current_tokens
    if remaining > 4000:
        print(f"  → Plenty of room. Can fetch more data.")
    elif remaining > 1500:
        print(f"  → Moderate budget. Query existing BH data instead of fetching new.")
        print(f"    Available datasets:")
        for k in bh.keys():
            print(f"      {k['key']:30s} {k['row_count']} rows")
    else:
        print(f"  → Tight budget. Use bh_aggregate for scalar answers only.")

    # Compression history by dataset
    print(f"\n--- Per-dataset efficiency ---")
    for k in bh.keys():
        slot = bh._get_slot(k["key"])
        if slot:
            raw_est = len(json.dumps(slot.rows[:5], default=str)) * (slot.row_count / 5)
            print(f"  {k['key']:30s} {slot.row_count:>4} rows, "
                  f"~{raw_est / 1024:.0f}KB raw, {slot.access_count} accesses")


if __name__ == "__main__":
    run_demo()
