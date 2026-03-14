# Schema

Schema inference, column typing, and statistics computation.

## Type Constants

| Constant | Value |
|----------|-------|
| `T_NUMERIC` | `"numeric"` |
| `T_STRING` | `"string"` |
| `T_DATE` | `"date"` |
| `T_DATETIME` | `"datetime"` |
| `T_BOOL` | `"bool"` |
| `T_DICT` | `"dict"` |
| `T_LIST` | `"list"` |
| `T_MIXED` | `"mixed"` |
| `T_NULL` | `"null"` |

## Schema Inference

::: binomialhash.schema.infer_schema

## Data Classes

::: binomialhash.schema.SchemaFeatureProfile

::: binomialhash.schema.SchemaDecision

## Helpers

::: binomialhash.schema.compute_col_stats

::: binomialhash.schema.to_float_strict

::: binomialhash.schema.try_parse_date
