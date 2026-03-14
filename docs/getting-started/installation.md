# Installation

## Requirements

- Python 3.9 or later
- NumPy >= 1.26 (installed automatically)

## Basic Install

```bash
pip install binomialhash
```

## Optional Extras

BinomialHash has several optional dependency groups for specific features:

=== "Exact Token Counting (OpenAI / xAI)"

    ```bash
    pip install binomialhash[openai]
    ```

    Installs `tiktoken` for exact token counting instead of the `chars / 4` heuristic.

=== "Excel Export"

    ```bash
    pip install binomialhash[excel]
    ```

    Installs `openpyxl` for the Excel batch export feature.

=== "SciPy Extensions"

    ```bash
    pip install binomialhash[scipy]
    ```

    Installs `scipy` for advanced statistical methods.

=== "Everything"

    ```bash
    pip install binomialhash[all]
    ```

    Installs all optional dependencies: `tiktoken`, `openpyxl`, and `scipy`.

## Development Install

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/Binomial-Capital-Management/binomialhash.git
cd binomialhash
pip install -e ".[dev]"
```

This includes `pytest`, `pytest-asyncio`, `tiktoken`, and `scipy` for running the test suite.

## Verify Installation

```python
from binomialhash import BinomialHash

bh = BinomialHash()
print(bh.context_stats())
# {'tool_calls': 0, 'chars_in_raw': 0, 'chars_out_to_llm': 0, ...}
```

## Optional: Check Token Counting

```python
from binomialhash.tokenizers import count_tokens, is_exact

if is_exact("openai"):
    print("tiktoken available -- exact token counting enabled")
else:
    print("Using chars/4 heuristic -- install tiktoken for exact counts")

print(count_tokens("Hello world", provider="openai"))
```
