"""Backward-compat shim — real implementation lives in core.py."""

from .core import (  # noqa: F401
    BUDGET_BYTES,
    DEFAULT_BINOMIAL_HASH_POLICY,
    INGEST_THRESHOLD_CHARS,
    MAX_PREVIEW_ROWS,
    MAX_RETRIEVE_ROWS,
    MAX_SLOTS,
    BinomialHash,
    BinomialHashPolicy,
    BinomialHashSlot,
)
