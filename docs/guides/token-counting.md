# Token Counting

BinomialHash includes provider-aware token counting to help monitor and optimise context budget usage.

## Basic Usage

```python
from binomialhash.tokenizers import count_tokens, is_exact

n = count_tokens("Hello world", provider="openai")
print(n)  # exact count with tiktoken, or heuristic estimate

if is_exact("openai"):
    print("Using tiktoken for exact counts")
else:
    print("Using chars/4 heuristic")
```

## Supported Providers

| Provider | Exact Counting | Requirement |
|----------|---------------|-------------|
| `openai` | Yes (with tiktoken) | `pip install tiktoken` or `pip install binomialhash[openai]` |
| `xai` | Yes (uses tiktoken) | Same as OpenAI |
| `anthropic` | No (heuristic only) | No offline tokenizer available |
| `gemini` | No (heuristic only) | No offline tokenizer available |

## Heuristic Fallback

When exact counting is unavailable, BinomialHash uses `ceil(len(text) / 4)` as an estimate. This is roughly correct for English prose and JSON but can be off by 20-30% on code or non-Latin scripts.

Each provider warns once per process when falling back to the heuristic.

## OpenAI Encoding

The OpenAI tokenizer defaults to `o200k_base` (used by GPT-4o, GPT-4.1, and GPT-5 models). You can specify an alternative encoding:

```python
n = count_tokens("Hello world", provider="openai", encoding="cl100k_base")
```

## Context Stats

Every `BinomialHash` instance tracks context budget usage:

```python
bh = BinomialHash()
# ... after several ingestions and tool calls ...

stats = bh.context_stats()
print(stats)
```

Returns:

| Field | Description |
|-------|-------------|
| `tool_calls` | Total number of tool invocations |
| `chars_in_raw` | Total raw characters ingested |
| `chars_out_to_llm` | Total characters returned to the LLM |
| `compression_ratio` | `chars_in_raw / chars_out_to_llm` |
| `est_tokens_out` | Estimated tokens sent to the LLM (`chars_out / 4`) |
| `slots` | Number of active data slots |
| `mem_bytes` | Total memory used by stored data |

## Logging

Call `bh.log_summary()` at the end of a request to emit a structured log line with all context stats and per-slot details:

```python
bh.log_summary()
# [BH-perf] REQUEST | 5 calls | in=120000 -> out=8000 (15.0x, ~2000 tok) | 2 slots 1.0MB | ...
```
