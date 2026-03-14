# Quickstart

This guide walks through the core BinomialHash workflow: ingesting data, querying it, and connecting it to an LLM via provider adapters.

## 1. Ingest Data

Create a `BinomialHash` instance and feed it a JSON payload:

```python
import json
from binomialhash import BinomialHash

bh = BinomialHash()

data = [
    {"ticker": "AAPL", "price": 189.50, "volume": 54_000_000, "sector": "Technology"},
    {"ticker": "MSFT", "price": 378.20, "volume": 28_000_000, "sector": "Technology"},
    {"ticker": "JPM",  "price": 195.30, "volume": 12_000_000, "sector": "Financials"},
    # ... imagine hundreds more rows
]

raw = json.dumps(data)
summary = bh.ingest(raw, "market_data")
print(summary)
```

If `len(raw) > 3000` characters, BinomialHash:

1. Parses the JSON and extracts the tabular rows
2. Infers column types and computes statistics
3. Builds a manifold surface (if enough axes and fields exist)
4. Stores everything in a content-addressed slot
5. Returns a compact summary with the storage key

If the payload is small, it passes through unchanged.

## 2. Retrieve and Query

Use the storage key from the summary to interact with the data:

```python
# Retrieve rows
rows = bh.retrieve("market_data_abc123", offset=0, limit=10)

# Aggregate
mean_price = bh.aggregate("market_data_abc123", "price", "mean")

# Query with a predicate
expensive = bh.query(
    "market_data_abc123",
    where={"column": "price", "op": ">", "value": 200},
    limit=20,
)

# Group by
by_sector = bh.group_by(
    "market_data_abc123",
    group_column="sector",
    agg_column="volume",
    agg_func="sum",
)

# Schema inspection
schema = bh.schema("market_data_abc123")
```

## 3. Connect to an LLM

BinomialHash provides 68 provider-neutral tool definitions. Use an adapter to register them with your LLM provider:

=== "OpenAI"

    ```python
    from binomialhash.tools import get_all_tools
    from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call

    specs = get_all_tools(bh)
    tools = get_openai_tools(specs)

    response = client.responses.create(model="gpt-4o", tools=tools, input=messages)

    for item in response.output:
        if item.type == "function_call":
            result = handle_openai_tool_call(specs, item.name, item.arguments)
    ```

=== "Anthropic"

    ```python
    from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use

    tools = get_anthropic_tools(specs)
    # Pass to client.messages.create(tools=tools, ...)

    result = handle_anthropic_tool_use(specs, block.name, block.input)
    ```

=== "Google Gemini"

    ```python
    from google.genai import types
    from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call

    decls = get_gemini_tools(specs)
    gemini_tools = types.Tool(function_declarations=decls)

    result = handle_gemini_tool_call(specs, fc.name, fc.args)
    ```

=== "xAI / Grok"

    ```python
    from binomialhash.adapters.xai import get_xai_tools, handle_xai_tool_call

    tools = get_xai_tools(specs)  # OpenAI-compatible format
    ```

## 4. Monitor Context Usage

Track how much context budget BinomialHash is saving:

```python
stats = bh.context_stats()
print(stats)
# {
#     "tool_calls": 5,
#     "chars_in_raw": 120000,
#     "chars_out_to_llm": 8000,
#     "compression_ratio": 15.0,
#     "est_tokens_out": 2000,
#     "slots": 2,
#     "mem_bytes": 1048576,
# }
```

## Next Steps

- [Architecture](../guides/architecture.md) -- understand the ingest pipeline and slot model
- [Provider Adapters](../guides/provider-adapters.md) -- deeper dive into tool registration per provider
- [Statistical Analysis](../guides/statistical-analysis.md) -- the 39 analysis tools across 7 stages
- [Manifold Navigation](../guides/manifold-navigation.md) -- spatial reasoning over your data
