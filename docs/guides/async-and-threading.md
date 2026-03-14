# Async & Threading

BinomialHash is designed for concurrent use in both threaded and async Python applications.

## Thread Safety

All mutable state in `BinomialHash` is protected by a `threading.RLock`:

- `_slots` -- the slot dictionary
- `_fingerprints` -- the fingerprint-to-key mapping
- `_used_bytes` -- memory accounting
- `_ctx_chars_in`, `_ctx_chars_out`, `_ctx_tool_calls` -- context budget counters

The lock is **reentrant** (`RLock`, not `Lock`), which means internal methods like `_get_slot` can acquire the lock even when called from within `ingest` or other methods that already hold it.

### What Is Protected

Every public method that reads or writes shared state acquires the lock:

```python
def retrieve(self, key, ...):
    with self._lock:
        slot = self._slots.get(key)
        ...
```

For `ingest`, the CPU-bound pre-processing (JSON parsing, row extraction) runs **outside** the lock. Only the critical section that checks fingerprints and mutates `_slots` is locked:

```python
def ingest(self, raw_text, label):
    # Pre-processing (no lock needed)
    data = json.loads(raw_text)
    rows, meta = extract_rows(data)
    fp = self._fingerprint(raw_text)

    with self._lock:
        # Check-then-act on shared state
        if fp in self._fingerprints:
            ...
        self._slots[key] = slot
        ...
```

This means multiple threads can parse JSON concurrently; serialisation only happens at the slot-mutation boundary.

## Async Wrappers

Every public method has an async counterpart prefixed with `a`:

| Sync | Async |
|------|-------|
| `ingest()` | `aingest()` |
| `retrieve()` | `aretrieve()` |
| `aggregate()` | `aaggregate()` |
| `query()` | `aquery()` |
| `group_by()` | `agroup_by()` |
| `schema()` | `aschema()` |
| `keys()` | `akeys()` |
| `context_stats()` | `acontext_stats()` |

Async wrappers use `asyncio.to_thread()` to offload CPU-bound work to the default thread pool executor:

```python
async def aingest(self, raw_text: str, label: str) -> str:
    return await asyncio.to_thread(self.ingest, raw_text, label)
```

This ensures the event loop is never blocked by data processing while the thread pool benefits from the `RLock` protection.

### Usage

```python
import asyncio
from binomialhash import BinomialHash

bh = BinomialHash()

async def main():
    summary = await bh.aingest(huge_json, "market_data")
    rows = await bh.aretrieve("market_data_abc123")
    agg = await bh.aaggregate("market_data_abc123", "price", "mean")

asyncio.run(main())
```

### Concurrent Async Ingestion

Multiple async ingestions can run concurrently:

```python
async def ingest_all(bh, datasets):
    tasks = [bh.aingest(data, label) for data, label in datasets]
    return await asyncio.gather(*tasks)
```

Each task runs in a separate thread via `to_thread`, and the `RLock` ensures slot mutations are serialised.

## Request-Scoped Isolation

In web applications, use `contextvars` for per-request isolation:

```python
from binomialhash.context import init_binomial_hash, get_binomial_hash

# In middleware (e.g., FastAPI)
@app.middleware("http")
async def bh_middleware(request, call_next):
    init_binomial_hash()  # fresh instance for this request
    return await call_next(request)

# In route handlers
async def my_route():
    bh = get_binomial_hash()  # returns the request's instance
    await bh.aingest(data, "label")
```

The `ContextVar` ensures each request/task sees its own `BinomialHash` instance. For multi-tenant isolation with `TaskGroup`, use `contextvars.copy_context()`:

```python
import asyncio
import contextvars

async def tenant_task(data):
    ctx = contextvars.copy_context()
    await ctx.run(process_tenant, data)
```

## Async Middleware

The middleware module (`bh_intercept`, `wrap_tool_with_bh`) automatically detects async functions and uses `aingest` internally:

```python
@bh_intercept(label="data")
async def fetch(query: str) -> dict:
    return await external_api(query)
# aingest is called, event loop is not blocked
```

## Raw Mode

Both `raw_mode()` (from `middleware`) and `bh_raw_mode()` (from `context`) are `contextvars`-based and safe for concurrent use. Each thread or async task has its own context, so enabling raw mode in one task does not affect others.
