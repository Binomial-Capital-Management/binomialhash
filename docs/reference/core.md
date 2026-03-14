# Core

The central module containing the `BinomialHash` class, the `BinomialHashSlot` data structure, and the `BinomialHashPolicy` configuration.

## BinomialHash

::: binomialhash.core.BinomialHash
    options:
      show_bases: true
      members:
        - ingest
        - retrieve
        - aggregate
        - query
        - group_by
        - schema
        - keys
        - context_stats
        - log_summary
        - to_chunks
        - to_excel_batch
        - aingest
        - aretrieve
        - aaggregate
        - aquery
        - agroup_by
        - aschema
        - akeys
        - acontext_stats

## BinomialHashSlot

::: binomialhash.core.BinomialHashSlot

## BinomialHashPolicy

::: binomialhash.core.BinomialHashPolicy

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `INGEST_THRESHOLD_CHARS` | 3000 | Payloads under this size pass through unchanged |
| `MAX_PREVIEW_ROWS` | 3 | Rows included in the ingest summary |
| `MAX_RETRIEVE_ROWS` | 50 | Default cap on retrieved rows |
| `MAX_SLOTS` | 50 | Maximum number of stored datasets |
| `BUDGET_BYTES` | 50 MB | Total memory budget across all slots |
