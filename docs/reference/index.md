# API Reference

Full reference documentation for all BinomialHash modules, auto-generated from source docstrings.

## Core

| Module | Description |
|--------|-------------|
| [Core](core.md) | `BinomialHash` class, `BinomialHashSlot`, `BinomialHashPolicy` |
| [Context](context.md) | Request-scoped helpers: `init_binomial_hash`, `get_binomial_hash`, `bh_ingest` |
| [Middleware](middleware.md) | Auto-interception: `bh_intercept`, `wrap_tool_with_bh`, `raw_mode` |

## Data Processing

| Module | Description |
|--------|-------------|
| [Schema](schema.md) | Schema inference, column typing, `SchemaFeatureProfile`, `SchemaDecision` |
| [Extract](extract.md) | Row extraction, flattening, embedded table explosion |
| [Predicates](predicates.md) | Predicate building, sorting, filtering |
| [Insights](insights.md) | Objective-driven insight extraction |

## Tools & Adapters

| Module | Description |
|--------|-------------|
| [Tools](tools.md) | `ToolSpec`, `get_all_tools`, `get_tools_by_group` |
| [Adapters](adapters.md) | OpenAI, Anthropic, Gemini, xAI adapters |

## Analysis & Navigation

| Module | Description |
|--------|-------------|
| [Stats](stats.md) | `StatsPolicy` and all 39 analysis functions |
| [Manifold](manifold.md) | `ManifoldSurface`, grid construction, spatial reasoning |

## Output

| Module | Description |
|--------|-------------|
| [Exporters](exporters.md) | CSV, Markdown, Excel, rows, chunks, artifact exporters |
| [Tokenizers](tokenizers.md) | Provider-aware token counting |
