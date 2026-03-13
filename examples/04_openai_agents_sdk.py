"""OpenAI Agents SDK integration — bridge ToolSpec to @function_tool.

This is the pattern used in production at Sentry. The OpenAI Agents SDK
uses @function_tool decorators that inspect function signatures. This example
shows how to dynamically generate those from BinomialHash ToolSpec objects,
giving you the best of both worlds:

  - ToolSpec: provider-neutral, testable, defined in the package
  - @function_tool: SDK-native, works with Agent/Runner/handoffs

Requires: pip install openai-agents binomialhash
"""

import json
import inspect
from typing import Optional

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools, get_tools_by_group
from binomialhash.tools.base import ToolSpec


# ---------------------------------------------------------------------------
# Bridge: ToolSpec → @function_tool compatible async functions
# ---------------------------------------------------------------------------

def toolspec_to_function_tool(spec: ToolSpec):
    """Convert a ToolSpec into an OpenAI Agents SDK FunctionTool.

    The Agents SDK's @function_tool inspects the function signature to build
    the JSON schema. Since we already have the schema in the ToolSpec, we use
    the lower-level FunctionTool class directly.
    """
    from agents import FunctionTool

    async def _handler(ctx, args_json: str) -> str:
        args = json.loads(args_json) if isinstance(args_json, str) else args_json
        result = spec.handler(**args)
        return json.dumps(result, default=str)

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=spec.input_schema,
        on_invoke_tool=_handler,
    )


def toolspecs_to_function_tools(specs: list[ToolSpec]) -> list:
    """Batch-convert ToolSpecs to FunctionTools."""
    return [toolspec_to_function_tool(s) for s in specs]


# ---------------------------------------------------------------------------
# Alternative: keep hand-written @function_tool but delegate to ToolSpec handlers
# ---------------------------------------------------------------------------

def demo_manual_bridge():
    """Show the manual approach used in bh_tools.py.

    Sometimes you want hand-written functions for IDE autocomplete and docstrings,
    but still delegate the actual work to ToolSpec handlers.
    """
    from agents import function_tool

    @function_tool
    async def bh_retrieve(
        key: str,
        offset: int = 0,
        limit: int = 25,
        sort_by: Optional[str] = None,
        sort_desc: bool = True,
        columns: Optional[str] = None,
    ) -> str:
        """Retrieve rows from a BinomialHash dataset."""
        bh = get_binomial_hash()
        col_list = None
        if columns:
            try:
                parsed = json.loads(columns)
                col_list = parsed if isinstance(parsed, list) else None
            except (json.JSONDecodeError, TypeError):
                col_list = [c.strip() for c in columns.split(",") if c.strip()]
        result = bh.retrieve(key, offset, limit, sort_by, sort_desc, col_list)
        return json.dumps(result, default=str)

    return bh_retrieve


# ---------------------------------------------------------------------------
# Full agent example
# ---------------------------------------------------------------------------

def run_demo():
    """Demonstrate the ToolSpec → FunctionTool bridge."""
    bh = init_binomial_hash()
    specs = get_all_tools(bh)

    print("=== OpenAI Agents SDK Bridge ===\n")

    # Method 1: Dynamic bridge
    try:
        from agents import FunctionTool
        tools = toolspecs_to_function_tools(specs)
        print(f"[Dynamic] Created {len(tools)} FunctionTool instances from ToolSpecs")
        for t in tools[:5]:
            print(f"  - {t.name}: {t.description[:60]}...")
        print(f"  ... and {len(tools) - 5} more\n")
    except ImportError:
        print("[Dynamic] openai-agents not installed — showing schema only\n")
        for s in specs[:5]:
            print(f"  - {s.name} ({s.group}): {s.description[:60]}...")
        print(f"  ... and {len(specs) - 5} more\n")

    # Method 2: Manual bridge (works without agents SDK)
    print("[Manual] Hand-written @function_tool with BH delegation:")
    manual = demo_manual_bridge()
    print(f"  - {manual.name}: {type(manual).__name__}\n")

    # Show the schema that would be sent to the Agents SDK
    print("--- ToolSpec JSON Schema (what the SDK sees) ---")
    sample = specs[0]
    print(f"Name: {sample.name}")
    print(f"Group: {sample.group}")
    print(f"Schema: {json.dumps(sample.input_schema, indent=2)[:400]}")

    # Demonstrate actual data flow
    print("\n--- Data flow demo ---")
    import random
    random.seed(42)
    rows = [{"ticker": t, "price": round(random.uniform(100, 500), 2), "volume": random.randint(1_000_000, 50_000_000)}
            for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] for _ in range(100)]
    raw = json.dumps(rows)

    summary = bh.ingest(raw, "portfolio_data")
    key = bh.keys()[0]["key"]
    print(f"Ingested {len(raw):,} chars → {len(summary)} char summary")

    # Call through the ToolSpec handler directly
    result = specs[0].handler(key=key, offset=0, limit=3)
    print(f"bh_retrieve(limit=3): {json.dumps(result, default=str)[:200]}...")

    result = specs[1].handler(key=key, column="price", func="mean")
    print(f"bh_aggregate(price, mean): {json.dumps(result, default=str)}")


if __name__ == "__main__":
    run_demo()
