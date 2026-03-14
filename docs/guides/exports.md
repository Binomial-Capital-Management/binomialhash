# Exports

BinomialHash can export stored data in multiple formats for download, display, or further processing.

## Export Formats

| Format | Method / Tool | Description |
|--------|---------------|-------------|
| CSV | `bh_export_csv` | Standard CSV string via Python's `csv` module |
| Markdown | `bh_export_markdown` | GitHub Flavored Markdown table |
| Excel | `bh_export_excel` | Header + values matrix (requires `openpyxl` for file writing) |
| Rows | `bh_export_rows` | Raw list of dicts (JSON-serializable) |
| Chunks | `bh.to_chunks()` | Split data into sequential chunks for streaming |
| Artifact | `bh_export_artifact` | Wrapped format with metadata for chat frontend consumption |

## CSV Export

```python
from binomialhash.exporters.csv import export_csv

csv_string = export_csv(
    rows=slot.rows,
    columns=slot.columns,
    col_types=slot.col_types,
    select_columns=["ticker", "price", "volume"],  # optional subset
    sort_by="price",
    sort_desc=True,
    max_rows=500,
)
```

Via LLM tool:

```python
result = handle_tool_call(specs, "bh_export_csv", {
    "key": "market_data_abc123",
    "sort_by": "price",
    "max_rows": 100,
})
```

## Markdown Export

Produces a GitHub Flavored Markdown table with pipe-escaped values:

```python
from binomialhash.exporters.markdown import export_markdown

md = export_markdown(
    rows=slot.rows,
    columns=slot.columns,
    col_types=slot.col_types,
    max_rows=50,
)
```

Truncated tables include a footer: `*... and N more rows (M total)*`

## Artifact Export

Wraps an export in a structured artifact envelope with metadata, suitable for chat frontends that support downloadable files:

```python
result = handle_tool_call(specs, "bh_export_artifact", {
    "key": "market_data_abc123",
    "format": "csv",          # "csv", "markdown", or "json"
    "max_rows": 200,
    "title": "Market Data Export",
})
```

## Configurable Row Limits

Export row caps are defined in `BinomialHashPolicy` and enforced at the tool handler level (not inside the exporter functions themselves):

| Policy Field | Default | Controls |
|-------------|---------|----------|
| `export_csv_max_rows` | 50,000 | CSV export cap |
| `export_excel_max_rows` | 10,000 | Excel export cap |
| `export_markdown_max_rows` | 200 | Markdown table cap |
| `export_rows_max_rows` | 50,000 | Raw rows export cap |

The actual exported row count is `min(caller_max_rows, policy_max_rows)`. Direct calls to exporter functions bypass the policy and respect only the `max_rows` parameter.

## Chunked Export

Split data into sequential chunks for streaming or pagination:

```python
chunks = bh.to_chunks("market_data_abc123", chunk_size=100)
# Returns: [{"chunk": 0, "rows": [...], "total_chunks": 5}, ...]
```
