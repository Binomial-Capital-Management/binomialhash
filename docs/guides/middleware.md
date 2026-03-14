# Middleware

BinomialHash middleware automatically intercepts large tool outputs and compresses them, with zero changes to existing tool functions.

## Decorator Form

Use `@bh_intercept` to wrap any function that returns structured data:

```python
from binomialhash.middleware import bh_intercept

@bh_intercept(label="market_data")
def fetch_data(ticker: str) -> dict:
    return huge_json_response  # auto-compacted if > 3000 chars
```

When `fetch_data` is called:

- If the return value is a dict/list whose JSON serialisation exceeds the threshold (default 3000 chars), it is ingested into BinomialHash and the compact summary is returned instead.
- Small or non-structured return values pass through unchanged.

## Wrapper Form

For third-party or dynamically-registered functions, use `wrap_tool_with_bh`:

```python
from binomialhash.middleware import wrap_tool_with_bh

wrapped = wrap_tool_with_bh(third_party_fetch, label="external_data")
result = wrapped("AAPL")  # auto-compacted
```

## Parameters

Both `bh_intercept` and `wrap_tool_with_bh` accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | `str` | (required) | Storage key label used during ingest |
| `bh` | `BinomialHash` | `None` | Explicit instance. Falls back to the request-scoped instance from `context.get_binomial_hash()` |
| `threshold` | `int` | `3000` | Minimum character count for ingestion to kick in |

## Raw Mode Bypass

When you need the original payload (e.g., for internal processing or provider-specific formatting), use `raw_mode()`:

```python
from binomialhash.middleware import raw_mode

@bh_intercept(label="market_data")
def fetch_data(ticker: str) -> dict:
    return huge_json_response

# Normal call -- returns BH summary
summary = fetch_data("AAPL")

# Raw mode -- returns original dict
with raw_mode():
    native = fetch_data("AAPL")
```

`raw_mode()` is:

- **Thread-safe** -- uses `contextvars` for isolation
- **Async-safe** -- each async task has its own context
- **Nestable** -- the outer scope's value is restored on exit

## Async Support

Both forms work transparently with async functions:

```python
@bh_intercept(label="async_data")
async def async_fetch(query: str) -> dict:
    return await external_api(query)

# Uses aingest() internally to avoid blocking the event loop
result = await async_fetch("SELECT * FROM ...")
```

## Request-Scoped Instances

The middleware automatically uses the request-scoped `BinomialHash` instance from `context.get_binomial_hash()`. In a web framework like FastAPI, initialise it per-request:

```python
from binomialhash.context import init_binomial_hash

@app.middleware("http")
async def bh_middleware(request, call_next):
    init_binomial_hash()  # fresh instance for this request
    return await call_next(request)
```

There is also a separate `bh_raw_mode()` context manager in `binomialhash.context` that uses a depth counter for nested bypass contexts -- useful when you need raw data inside a tool that is itself intercepted.
