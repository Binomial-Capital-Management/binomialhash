"""Request-scoped convenience helpers for the extracted BinomialHash package."""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Optional

from .core import BinomialHash

logger = logging.getLogger(__name__)

_bh_instance: contextvars.ContextVar[Optional[BinomialHash]] = contextvars.ContextVar(
    "binomial_hash",
    default=None,
)

# Depth counter (not a bool) so nested bh_raw_mode() contexts work correctly.
_bh_raw_mode_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "binomial_hash_raw_mode_depth",
    default=0,
)


def init_binomial_hash() -> BinomialHash:
    """Initialize a fresh BinomialHash for the current request."""
    bh = BinomialHash()
    _bh_instance.set(bh)
    logger.info("[BH] initialized new instance for request")
    return bh


def get_binomial_hash() -> BinomialHash:
    """Get the current request's BinomialHash (auto-init if needed)."""
    bh = _bh_instance.get(None)
    if bh is None:
        bh = init_binomial_hash()
    return bh


@contextmanager
def bh_raw_mode():
    """Temporarily bypass BH compaction for internal tool fetches."""
    token = _bh_raw_mode_depth.set(_bh_raw_mode_depth.get(0) + 1)
    try:
        yield
    finally:
        _bh_raw_mode_depth.reset(token)


def is_raw_mode() -> bool:
    """Return True when inside a ``bh_raw_mode()`` context."""
    return _bh_raw_mode_depth.get(0) > 0


def bh_ingest(raw_text: str, label: str) -> str:
    """Ingest a tool output using the current request-scoped instance."""
    return get_binomial_hash().ingest(raw_text, label)


async def async_bh_ingest(raw_text: str, label: str) -> str:
    """Async variant of :func:`bh_ingest`."""
    return await get_binomial_hash().aingest(raw_text, label)


__all__ = [
    "async_bh_ingest",
    "bh_ingest",
    "bh_raw_mode",
    "get_binomial_hash",
    "init_binomial_hash",
    "is_raw_mode",
    "_bh_raw_mode_depth",
]
