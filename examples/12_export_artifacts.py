"""Export artifacts for chat frontend consumption.

Production pattern: after an agent analyzes data via BH, the user wants to
download results as CSV, view as a Markdown table, or get a structured
artifact that the frontend can render as a download button.

This example shows all export formats:
  - CSV (for spreadsheet download)
  - Markdown (for inline chat rendering)
  - Excel batch (for Office.js integration)
  - Structured artifact (generic frontend download payload)

The export tools are registered as ToolSpecs so the LLM can call them directly.
"""

import json
import random

from binomialhash import BinomialHash, init_binomial_hash
from binomialhash.tools import get_tools_by_group
from binomialhash.adapters.common import handle_tool_call
from binomialhash.exporters import export_csv, export_markdown, export_rows, build_artifact


# ---------------------------------------------------------------------------
# Ingest sample data
# ---------------------------------------------------------------------------

def setup():
    bh = init_binomial_hash()
    random.seed(42)

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
                "GS", "BAC", "JNJ", "PFE", "UNH", "XOM", "CVX"]
    rows = []
    for ticker in tickers:
        for day in range(1, 11):
            rows.append({
                "ticker": ticker,
                "date": f"2025-03-{day:02d}",
                "price": round(random.uniform(100, 900), 2),
                "volume": random.randint(5_000_000, 80_000_000),
                "market_cap_b": round(random.uniform(200, 3500), 1),
                "pe_ratio": round(random.uniform(12, 80), 2),
                "dividend_yield": round(random.uniform(0, 3.5), 2),
                "sector": random.choice(["Tech", "Finance", "Health", "Energy"]),
                "52w_high": round(random.uniform(200, 1000), 2),
                "52w_low": round(random.uniform(50, 400), 2),
                "analyst_rating": random.choice(["Buy", "Hold", "Sell", "Strong Buy"]),
            })

    raw = json.dumps(rows)
    summary = bh.ingest(raw, "portfolio_holdings")
    key = bh.keys()[0]["key"]
    return bh, key


# ---------------------------------------------------------------------------
# Export demos
# ---------------------------------------------------------------------------

def run_demo():
    bh, key = setup()
    slot = bh._get_slot(key)

    print("=== Export Artifacts ===\n")

    # 1. CSV export
    print("--- [1] CSV ---")
    csv_output = export_csv(
        slot.rows, slot.columns, slot.col_types,
        select_columns=["ticker", "price", "volume", "pe_ratio"],
        sort_by="price", sort_desc=True, max_rows=5,
    )
    print(csv_output)

    # 2. Markdown export
    print("--- [2] Markdown ---")
    md_output = export_markdown(
        slot.rows, slot.columns, slot.col_types,
        select_columns=["ticker", "price", "market_cap_b", "pe_ratio", "analyst_rating"],
        sort_by="market_cap_b", sort_desc=True, max_rows=8,
    )
    print(md_output)

    # 3. Excel batch (headers + values matrix for Office.js)
    print("--- [3] Excel Batch ---")
    batch = bh.to_excel_batch(key, ["ticker", "price", "volume"], "price", True, 5)
    print(f"Headers: {batch['headers']}")
    print(f"Values ({batch['total_exported']} rows):")
    for row in batch["values"][:3]:
        print(f"  {row}")
    print(f"  ... ({batch['total_available']} available)\n")

    # 4. Structured artifact (for frontend download buttons)
    print("--- [4] Artifacts ---")
    for fmt in ["csv", "markdown", "json"]:
        artifact = build_artifact(
            slot.rows, slot.columns, slot.col_types,
            format=fmt,
            label="portfolio_export",
            select_columns=["ticker", "price", "volume", "pe_ratio", "sector"],
            sort_by="price", sort_desc=True, max_rows=8,
        )
        print(f"  {fmt:10s} → type={artifact['type']}, mime={artifact['mime_type']}, "
              f"filename={artifact['filename']}, content_size={len(artifact['content'])} chars")

    # 5. Using export tools via ToolSpec (what the LLM calls)
    print("\n--- [5] Via ToolSpec (LLM tool calls) ---")
    export_specs = get_tools_by_group(bh, "export")
    print(f"Available export tools: {[s.name for s in export_specs]}")

    csv_result = handle_tool_call(export_specs, "bh_to_csv", {
        "key": key,
        "columns": "ticker,price,pe_ratio",
        "sort_by": "price",
        "sort_desc": True,
        "max_rows": 5,
    })
    print(f"\n  bh_to_csv result (artifact):")
    print(f"    type: {csv_result.get('type')}")
    print(f"    filename: {csv_result.get('filename')}")
    print(f"    content preview: {csv_result.get('content', '')[:150]}...")

    md_result = handle_tool_call(export_specs, "bh_to_markdown", {
        "key": key,
        "columns": "ticker,price,market_cap_b,analyst_rating",
        "sort_by": "market_cap_b",
        "sort_desc": True,
        "max_rows": 5,
    })
    md_text = md_result if isinstance(md_result, str) else md_result.get("content", "")
    print(f"\n  bh_to_markdown result (inline markdown):")
    print(f"    {md_text[:300]}...")

    # 6. Rows export (clean dicts for frontend tables)
    print("\n--- [6] Raw rows (for React table components) ---")
    clean_rows = export_rows(
        slot.rows, slot.columns, slot.col_types,
        select_columns=["ticker", "price", "volume"],
        sort_by="volume", sort_desc=True, limit=5,
    )
    for r in clean_rows:
        print(f"  {r}")

    print(f"\n=== All export formats demonstrated ===")


if __name__ == "__main__":
    run_demo()
