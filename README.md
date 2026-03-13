# BinomialHash

Content-addressed, schema-aware structured data compaction for LLM tool outputs.

BinomialHash intercepts large JSON payloads from tool calls, infers schema and statistics, deduplicates by content fingerprint, and returns compact summaries that fit in LLM context windows. Agent tools let the model retrieve, aggregate, query, group, and export data on demand without blowing the token budget.

## Install

```bash
pip install binomialhash
```

With exact token counting (OpenAI / xAI):

```bash
pip install binomialhash[openai]
```

All optional dependencies:

```bash
pip install binomialhash[all]
```

## Quickstart

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

raw = json.dumps(data)
summary = bh.ingest(raw, "market_data")
# If len(raw) > 3000 chars: returns a compact schema + stats summary
# If small: passes through unchanged

# Query stored data
rows = bh.retrieve("market_data_abc123", offset=0, limit=10)
agg  = bh.aggregate("market_data_abc123", "price", "mean")
```

## Provider Adapters

BinomialHash ships with 25 provider-neutral tool definitions that expose its full API (retrieve, aggregate, query, group, regress, manifold navigation, etc.) to any LLM. Adapters translate these into provider-specific formats.

### OpenAI

```python
from binomialhash import BinomialHash
from binomialhash.tools import get_all_tools
from binomialhash.adapters.openai import get_openai_tools, handle_openai_tool_call

bh = BinomialHash()
specs = get_all_tools(bh)
tools = get_openai_tools(specs)  # Responses API format (default)

# Pass to the API
response = client.responses.create(model="gpt-4o", tools=tools, input=messages)

# Handle function calls
for item in response.output:
    if item.type == "function_call":
        result = handle_openai_tool_call(specs, item.name, item.arguments)
```

For Chat Completions (legacy):

```python
tools = get_openai_tools(specs, format="chat_completions")
```

### Anthropic

```python
from binomialhash.adapters.anthropic import get_anthropic_tools, handle_anthropic_tool_use

tools = get_anthropic_tools(specs)
# Pass to client.messages.create(tools=tools, ...)

# Handle tool_use blocks
result = handle_anthropic_tool_use(specs, block.name, block.input)
```

### Google Gemini

```python
from google.genai import types
from binomialhash.adapters.gemini import get_gemini_tools, handle_gemini_tool_call

decls = get_gemini_tools(specs)
gemini_tools = types.Tool(function_declarations=decls)

# Handle function_call parts
result = handle_gemini_tool_call(specs, fc.name, fc.args)
```

### xAI / Grok

```python
from binomialhash.adapters.xai import get_xai_tools, handle_xai_tool_call

tools = get_xai_tools(specs)
# Uses OpenAI-compatible format
```

### Provider Router

```python
from binomialhash.adapters import get_tools_for_provider

tools = get_tools_for_provider(specs, provider="openai")
tools = get_tools_for_provider(specs, provider="anthropic")
```

## Middleware

Auto-intercept large tool outputs without modifying tool functions:

```python
from binomialhash.middleware import bh_intercept, raw_mode

@bh_intercept(label="market_data")
def fetch_data(ticker: str) -> dict:
    return huge_json_response  # auto-compacted if > 3000 chars

# Bypass interception when you need the raw payload
with raw_mode():
    native = fetch_data("AAPL")  # returns original dict
```

Wrapper form for third-party functions:

```python
from binomialhash.middleware import wrap_tool_with_bh

wrapped = wrap_tool_with_bh(third_party_fetch, label="external_data")
```

Both sync and async functions are supported.

## Token Counting

```python
from binomialhash.tokenizers import count_tokens, is_exact

n = count_tokens("Hello world", provider="openai")   # exact with tiktoken
n = count_tokens("Hello world", provider="anthropic") # heuristic (chars/4)

if is_exact("openai"):
    print("Using tiktoken")
```

Built-in context stats on every BinomialHash instance:

```python
stats = bh.context_stats()
# {"tool_calls": 5, "chars_in_raw": 120000, "chars_out_to_llm": 8000,
#  "compression_ratio": 15.0, "est_tokens_out": 2000, ...}
```

## Package Structure

```
binomialhash/
  core.py              # BinomialHash class — ingest, retrieve, aggregate, query
  schema.py            # Schema inference and column typing
  stats.py             # Statistical functions (regression, PCA, correlations)
  extract.py           # Row extraction from nested JSON
  predicates.py        # Predicate building and row filtering
  context.py           # Request-scoped contextvar helpers
  insights.py          # Objective-driven insight extraction
  middleware.py         # Auto-interception decorator and raw-mode bypass
  manifold/            # Manifold surface construction and navigation
  tools/               # 25 provider-neutral ToolSpec definitions
  adapters/            # OpenAI, Anthropic, Gemini, xAI schema translators
  exporters/           # Excel batch export, embedding chunks
  tokenizers/          # Provider-aware token counting
```

## Scope and Limitations

BinomialHash is a structured data compaction and analytics engine. Its manifold topology outputs are operational structural diagnostics — not proofs of true underlying manifold topology. The edge-incidence manifoldness gate is implemented; vertex-link validation and combinatorial orientability are on the roadmap.

## Development

```bash
cd binomialhash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT
