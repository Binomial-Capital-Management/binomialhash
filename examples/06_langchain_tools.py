"""LangChain / LangGraph integration with BinomialHash.

Shows how to convert BH ToolSpecs into LangChain StructuredTool objects
so they work natively with LangGraph agents, chains, and tool nodes.

Pattern:
  ToolSpec (provider-neutral) → StructuredTool (LangChain) → LangGraph node

Requires: pip install langchain-core langgraph binomialhash
"""

import json
from typing import Any, Dict

from binomialhash import BinomialHash, init_binomial_hash, get_binomial_hash
from binomialhash.tools import get_all_tools, get_tools_by_group
from binomialhash.tools.base import ToolSpec


# ---------------------------------------------------------------------------
# ToolSpec → LangChain StructuredTool bridge
# ---------------------------------------------------------------------------

def toolspec_to_langchain(spec: ToolSpec):
    """Convert a BinomialHash ToolSpec to a LangChain StructuredTool.

    LangChain tools need:
      - name, description
      - args_schema (Pydantic model) OR just a function with annotations
      - a callable that takes kwargs and returns a string

    We use the raw-schema approach since ToolSpec already has JSON Schema.
    """
    from langchain_core.tools import StructuredTool

    def _invoke(**kwargs: Any) -> str:
        result = spec.handler(**kwargs)
        return json.dumps(result, default=str)

    return StructuredTool(
        name=spec.name,
        description=spec.description,
        func=_invoke,
        args_schema=_json_schema_to_pydantic(spec.name, spec.input_schema),
    )


def _json_schema_to_pydantic(name: str, schema: Dict[str, Any]):
    """Dynamically build a Pydantic model from a JSON Schema.

    This lets LangChain validate tool inputs before calling the handler.
    """
    from pydantic import create_model, Field
    from typing import Optional

    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}

    for prop_name, prop_def in properties.items():
        py_type = type_map.get(prop_def.get("type", "string"), str)
        desc = prop_def.get("description", "")
        default = prop_def.get("default")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=desc))
        else:
            fields[prop_name] = (Optional[py_type], Field(default=default, description=desc))

    return create_model(f"{name}_Args", **fields)


def toolspecs_to_langchain(specs: list[ToolSpec]) -> list:
    """Batch-convert all ToolSpecs to LangChain StructuredTools."""
    return [toolspec_to_langchain(s) for s in specs]


# ---------------------------------------------------------------------------
# LangGraph agent with BH tools
# ---------------------------------------------------------------------------

def build_langgraph_agent(bh: BinomialHash):
    """Build a minimal LangGraph agent with BH tools.

    This is the architecture used in production:
      Supervisor → Tool Node (BH tools) → Agent Node → Response
    """
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError:
        return None

    retrieval_tools = toolspecs_to_langchain(get_tools_by_group(bh, "retrieval"))
    stats_tools = toolspecs_to_langchain(get_tools_by_group(bh, "stats"))
    all_tools = retrieval_tools + stats_tools

    tool_map = {t.name: t for t in all_tools}

    def tool_node(state):
        """Execute a tool call and return the result."""
        tool_call = state.get("pending_tool_call")
        if not tool_call:
            return state
        tool = tool_map.get(tool_call["name"])
        if tool:
            result = tool.invoke(tool_call["args"])
            state["tool_results"].append({"name": tool_call["name"], "result": result})
        state["pending_tool_call"] = None
        return state

    return all_tools, tool_node


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    bh = init_binomial_hash()
    specs = get_all_tools(bh)

    print("=== LangChain / LangGraph integration ===\n")

    # Convert to LangChain tools
    try:
        lc_tools = toolspecs_to_langchain(specs)
        print(f"Created {len(lc_tools)} LangChain StructuredTools:\n")
        for t in lc_tools[:8]:
            print(f"  {t.name:30s} ({t.description[:50]}...)")
        print(f"  ... and {len(lc_tools) - 8} more\n")

        # Show the auto-generated Pydantic model
        sample = lc_tools[0]
        print(f"--- {sample.name} args schema ---")
        print(f"  Model: {sample.args_schema.__name__}")
        for field_name, field_info in sample.args_schema.model_fields.items():
            print(f"  {field_name}: {field_info.annotation} = {field_info.default}")

    except ImportError as e:
        print(f"langchain not installed ({e}) — showing ToolSpec info only\n")
        for s in specs[:8]:
            print(f"  {s.name:30s} [{s.group}] {s.description[:50]}...")
        print(f"  ... and {len(specs) - 8} more")
        lc_tools = None

    # Demonstrate data flow
    print("\n--- Data flow demo ---")
    import random
    random.seed(42)
    raw = json.dumps([
        {"region": r, "product": p, "revenue": round(random.uniform(10_000, 500_000), 2),
         "units": random.randint(100, 10_000), "margin": round(random.uniform(0.1, 0.6), 4)}
        for r in ["US", "EU", "APAC", "LATAM"]
        for p in ["Widget A", "Widget B", "Service X", "Service Y", "Platform"]
        for _ in range(50)
    ])

    summary = bh.ingest(raw, "sales_data")
    key = bh.keys()[0]["key"]
    print(f"Ingested {len(raw):,} chars → {len(summary)} char summary\n")

    if lc_tools:
        # Use LangChain tools directly
        retrieve_tool = next(t for t in lc_tools if t.name == "bh_retrieve")
        result = retrieve_tool.invoke({"key": key, "limit": 3})
        print(f"LangChain bh_retrieve: {result[:200]}...\n")

        agg_tool = next(t for t in lc_tools if t.name == "bh_aggregate")
        result = agg_tool.invoke({"key": key, "column": "revenue", "func": "sum"})
        print(f"LangChain bh_aggregate: {result}")
    else:
        result = specs[0].handler(key=key, offset=0, limit=3)
        print(f"Direct ToolSpec call: {json.dumps(result, default=str)[:200]}...")


if __name__ == "__main__":
    run_demo()
