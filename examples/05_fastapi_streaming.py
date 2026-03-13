"""FastAPI SSE streaming endpoint with BinomialHash.

Production pattern: a chat endpoint that streams LLM responses while BH
compresses tool outputs mid-conversation. The client sees incremental
markdown chunks; the LLM never sees the full 50KB JSON — only the BH summary.

Run: uvicorn examples.05_fastapi_streaming:app --reload
Then: curl -N http://localhost:8000/chat -d '{"message":"analyze AAPL"}'
"""

import asyncio
import json
import time

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call

# ---------------------------------------------------------------------------
# FastAPI app (only created if fastapi is available)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app():
    app = FastAPI(title="BinomialHash Chat API")

    def _simulate_market_data():
        """In production this calls an MCP server or external API."""
        import random
        random.seed(int(time.time()) % 1000)
        return json.dumps([
            {"ticker": t, "date": f"2025-03-{d:02d}", "close": round(random.uniform(100, 500), 2),
             "volume": random.randint(5_000_000, 80_000_000), "sector": s,
             "pe_ratio": round(random.uniform(10, 50), 2), "dividend_yield": round(random.uniform(0, 4), 2)}
            for t, s in [("AAPL", "Tech"), ("MSFT", "Tech"), ("JPM", "Finance"), ("JNJ", "Health")]
            for d in range(1, 61)
        ])

    async def _stream_response(message: str):
        """Simulate a streaming agent response with BH tool use."""
        bh = init_binomial_hash()
        specs = get_all_tools(bh)

        yield f"data: {json.dumps({'type': 'status', 'content': 'Fetching market data...'})}\n\n"
        await asyncio.sleep(0.1)

        raw = _simulate_market_data()
        summary = bh.ingest(raw, "market_data")

        yield f"data: {json.dumps({'type': 'bh_summary', 'content': summary, 'raw_size': len(raw), 'summary_size': len(summary)})}\n\n"
        await asyncio.sleep(0.1)

        key = bh.keys()[0]["key"]
        schema = bh.schema(key)
        row_count = schema["row_count"]
        col_count = len(schema["columns"])
        yield f"data: {json.dumps({'type': 'status', 'content': f'Analyzing {row_count} rows across {col_count} columns...'})}\n\n"

        agg = bh.aggregate(key, "close", "mean")
        avg_price = agg["result"]
        yield f"data: {json.dumps({'type': 'chunk', 'content': f'Average closing price: ${avg_price:.2f}'})}\n\n"

        group = bh.group_by(
            key, ["sector"],
            json.dumps([{"column": "volume", "func": "sum", "alias": "total_vol"}]),
            "total_vol", True, 10,
        )
        for g in group.get("rows", []):
            line = f"  - {g.get('sector', '?')}: {g.get('total_vol', 0):,.0f} shares"
            yield f"data: {json.dumps({'type': 'chunk', 'content': line})}\n\n"
            await asyncio.sleep(0.05)

        stats = bh.context_stats()
        yield f"data: {json.dumps({'type': 'stats', 'compression': stats['compression_ratio'], 'tokens_saved': stats['chars_in_raw'] - stats['chars_out_to_llm']})}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", "")
        return StreamingResponse(
            _stream_response(message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "bh_version": "0.1.0"}

    return app


# ---------------------------------------------------------------------------
# Standalone demo (no server needed)
# ---------------------------------------------------------------------------

def run_standalone():
    """Run the streaming logic without FastAPI."""
    print("=== FastAPI streaming demo (standalone) ===\n")

    bh = init_binomial_hash()

    import random
    random.seed(42)
    raw = json.dumps([
        {"ticker": t, "date": f"2025-03-{d:02d}", "close": round(random.uniform(100, 500), 2),
         "volume": random.randint(5_000_000, 80_000_000), "sector": s,
         "pe_ratio": round(random.uniform(10, 50), 2)}
        for t, s in [("AAPL", "Tech"), ("MSFT", "Tech"), ("JPM", "Finance"),
                     ("GS", "Finance"), ("JNJ", "Health"), ("PFE", "Health")]
        for d in range(1, 61)
    ])

    print(f"Raw data: {len(raw):,} chars")
    summary = bh.ingest(raw, "market_data")
    print(f"BH summary: {len(summary)} chars ({len(raw) / len(summary):.0f}x compression)\n")

    key = bh.keys()[0]["key"]

    print("--- SSE events that would stream to frontend ---")
    print(f'event: bh_summary | {len(summary)} chars')
    print(f'event: chunk | "Analyzing {bh.schema(key)["row_count"]} rows..."')

    agg = bh.aggregate(key, "close", "mean")
    print(f'event: chunk | "Average close: ${agg["result"]:.2f}"')

    group = bh.group_by(
        key, ["sector"],
        json.dumps([{"column": "volume", "func": "sum", "alias": "total_vol"}]),
        "total_vol", True, 10,
    )
    for g in group.get("rows", []):
        print(f'event: chunk | "  {g["sector"]}: {g["total_vol"]:,.0f} shares"')

    stats = bh.context_stats()
    print(f'event: stats | compression={stats["compression_ratio"]:.1f}x, tokens_saved={stats["chars_in_raw"] - stats["chars_out_to_llm"]:,}')
    print('event: [DONE]')

    print(f"\n=== Total: {stats['chars_in_raw']:,} raw → {stats['chars_out_to_llm']:,} to LLM ===")


if HAS_FASTAPI:
    app = create_app()


if __name__ == "__main__":
    run_standalone()
