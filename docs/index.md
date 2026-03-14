# BinomialHash

**Content-addressed, schema-aware structured data compaction for LLM tool outputs.**

BinomialHash intercepts large JSON payloads from tool calls, infers schema and statistics, deduplicates by content fingerprint, and returns compact summaries that fit in LLM context windows. A suite of 68 provider-neutral tools lets the model retrieve, aggregate, query, group, analyse, and export data on demand -- without blowing the token budget.

## Key Features

- **Automatic compaction** -- Large JSON tool outputs are ingested, schema-inferred, and compressed into concise summaries. Small payloads pass through unchanged.
- **Content-addressed dedup** -- SHA-256 fingerprinting ensures the same data is never stored twice.
- **68 LLM tools** -- Retrieval, statistical analysis (39 methods across 7 stages), manifold navigation (14 tools), spatial reasoning (6 tools), and export.
- **Provider-neutral** -- Adapters for OpenAI, Anthropic, Google Gemini, and xAI translate tool definitions into each provider's wire format.
- **Middleware** -- Decorator and wrapper patterns auto-intercept tool outputs with zero changes to existing tool code.
- **Thread-safe and async-ready** -- All shared state is protected by an `RLock`; async wrappers (`aingest`, `aretrieve`, etc.) offload work via `asyncio.to_thread`.

## Quick Example

```python
import json
from binomialhash import BinomialHash

bh = BinomialHash()

data = [
    {"ticker": "AAPL", "price": 189.50, "volume": 54_000_000, "sector": "Technology"},
    {"ticker": "MSFT", "price": 378.20, "volume": 28_000_000, "sector": "Technology"},
    {"ticker": "JPM",  "price": 195.30, "volume": 12_000_000, "sector": "Financials"},
    # ... hundreds more rows ...
]

summary = bh.ingest(json.dumps(data), "market_data")
# Returns a compact schema + stats summary when the payload is large

rows = bh.retrieve("market_data_abc123", offset=0, limit=10)
agg  = bh.aggregate("market_data_abc123", "price", "mean")
```

## Package Structure

```
binomialhash/
  core.py              # BinomialHash class -- ingest, retrieve, aggregate, query
  schema.py            # Schema inference and column typing
  extract.py           # Row extraction from nested JSON
  predicates.py        # Predicate building and row filtering
  context.py           # Request-scoped contextvar helpers
  insights.py          # Objective-driven insight extraction
  middleware.py         # Auto-interception decorator and raw-mode bypass
  stats/               # 39 statistical tools across 7 stages
  manifold/            # Manifold surface construction and navigation
  tools/               # 68 provider-neutral ToolSpec definitions
  adapters/            # OpenAI, Anthropic, Gemini, xAI schema translators
  exporters/           # Markdown, CSV, Excel, chunked artifacts
  tokenizers/          # Provider-aware token counting
```

## Next Steps

<div class="grid cards" markdown>

-   **Installation**

    Set up BinomialHash with pip, including optional extras for token counting and Excel export.

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

-   **Quickstart**

    Ingest your first dataset, query it, and see compression in action.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   **Architecture**

    Understand how the ingest pipeline, slot model, and manifold construction work under the hood.

    [:octicons-arrow-right-24: Architecture](guides/architecture.md)

-   **API Reference**

    Full reference for every class, function, and module.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>
