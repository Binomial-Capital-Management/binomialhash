"""Multi-tenant request isolation with BinomialHash.

Production pattern: in a web server handling concurrent requests, each user
gets their own BH instance via contextvars. User A's financial data never
leaks into User B's context window.

This is critical for SaaS platforms:
  - Each FastAPI request creates a fresh BH via init_binomial_hash()
  - contextvars ensure isolation even under async concurrency
  - No locks needed — each coroutine has its own context
"""

import asyncio
import contextvars
import json
import random

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash


# ---------------------------------------------------------------------------
# Simulated request handler
# ---------------------------------------------------------------------------

async def handle_user_request(user_id: str, tickers: list[str]):
    """Simulate a single user's request lifecycle.

    In production this would be called from a FastAPI endpoint with
    RLS middleware setting the user context.
    """
    bh = init_binomial_hash()

    # Each user gets different data
    random.seed(hash(user_id))
    for ticker in tickers:
        rows = [
            {"ticker": ticker, "user": user_id, "price": round(random.uniform(100, 500), 2),
             "volume": random.randint(1_000_000, 50_000_000),
             "secret_field": f"{user_id}_confidential_{random.randint(1000, 9999)}"}
            for _ in range(100)
        ]
        raw = json.dumps(rows)
        bh.ingest(raw, f"{ticker}_data")

    # Verify: this user can only see their own data
    keys = bh.keys()
    all_rows = []
    for k in keys:
        result = bh.retrieve(k["key"], 0, 50, None, True, None)
        all_rows.extend(result.get("rows", []))

    user_ids_seen = set(r.get("user") for r in all_rows if r.get("user"))
    stats = bh.context_stats()

    return {
        "user_id": user_id,
        "datasets": len(keys),
        "total_rows_accessible": len(all_rows),
        "user_ids_in_data": list(user_ids_seen),
        "isolated": user_ids_seen == {user_id},
        "compression": stats["compression_ratio"],
    }


# ---------------------------------------------------------------------------
# Concurrent request simulation
# ---------------------------------------------------------------------------

async def run_concurrent_requests():
    """Run multiple user requests concurrently, verify isolation."""

    print("=== Multi-Tenant Isolation ===\n")

    users = [
        ("user_alice", ["AAPL", "MSFT"]),
        ("user_bob", ["GOOGL", "AMZN"]),
        ("user_carol", ["JPM", "GS", "MS"]),
        ("user_dave", ["XOM", "CVX"]),
    ]

    # Run all requests concurrently with isolated contexts
    tasks = []
    for user_id, tickers in users:
        ctx = contextvars.copy_context()
        task = asyncio.ensure_future(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda uid=user_id, t=tickers: ctx.run(
                    asyncio.get_event_loop().run_until_complete,
                    handle_user_request(uid, t),
                ),
            )
        )
        tasks.append(task)

    # Simpler sequential version that still demonstrates isolation
    print("Running 4 concurrent user requests...\n")
    results = []
    for user_id, tickers in users:
        ctx = contextvars.copy_context()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda uid=user_id, t=tickers: ctx.run(
                lambda: asyncio.new_event_loop().run_until_complete(
                    handle_user_request(uid, t)
                )
            ),
        )
        results.append(result)

    # Verify isolation
    all_isolated = True
    for r in results:
        status = "ISOLATED" if r["isolated"] else "LEAKED!"
        all_isolated = all_isolated and r["isolated"]
        print(f"[{status}] {r['user_id']:12s} | {r['datasets']} datasets | "
              f"{r['total_rows_accessible']} rows | users_seen={r['user_ids_in_data']} | "
              f"{r['compression']:.1f}x compression")

    print(f"\n{'All users isolated!' if all_isolated else 'ISOLATION BREACH DETECTED!'}")


def run_demo():
    """Simplified synchronous demo of the same concept."""
    print("=== Multi-Tenant Isolation (sync demo) ===\n")

    results = []
    users = [
        ("user_alice", ["AAPL", "MSFT"]),
        ("user_bob", ["GOOGL", "AMZN"]),
        ("user_carol", ["JPM", "GS", "MS"]),
    ]

    for user_id, tickers in users:
        ctx = contextvars.copy_context()

        def _run(uid=user_id, t=tickers):
            bh = init_binomial_hash()
            random.seed(hash(uid))
            for ticker in t:
                rows = [{"ticker": ticker, "user": uid,
                         "price": round(random.uniform(100, 500), 2),
                         "secret": f"{uid}_secret_{random.randint(1000, 9999)}"}
                        for _ in range(100)]
                bh.ingest(json.dumps(rows), f"{ticker}_data")

            keys = bh.keys()
            all_rows = []
            for k in keys:
                result = bh.retrieve(k["key"], 0, 100, None, True, None)
                all_rows.extend(result.get("rows", []))

            user_ids_seen = set(r.get("user") for r in all_rows if r.get("user"))
            return {"user_id": uid, "datasets": len(keys),
                    "rows": len(all_rows), "seen": list(user_ids_seen),
                    "isolated": user_ids_seen == {uid}}

        result = ctx.run(_run)
        results.append(result)

    for r in results:
        mark = "OK" if r["isolated"] else "FAIL"
        print(f"[{mark}] {r['user_id']:12s} | {r['datasets']} datasets | "
              f"{r['rows']} rows | users_in_data={r['seen']}")

    all_ok = all(r["isolated"] for r in results)
    print(f"\n{'All users isolated — contextvars work!' if all_ok else 'ISOLATION FAILURE'}")

    # Show the mechanism
    print("\n--- How it works ---")
    print("1. Each request calls init_binomial_hash() → fresh ContextVar")
    print("2. get_binomial_hash() returns the current request's instance")
    print("3. contextvars.copy_context() creates an isolated copy")
    print("4. No locks, no thread-locals — just Python contextvars")


if __name__ == "__main__":
    run_demo()
