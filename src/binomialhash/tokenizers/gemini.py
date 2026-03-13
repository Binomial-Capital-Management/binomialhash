"""Google Gemini token estimation.

The Gemini API provides ``client.models.count_tokens(...)`` but that
requires an API key and a network call.  No offline tokenizer library
is publicly available.

This module uses the character heuristic (``chars / 4``).  If Google
publishes an offline tokenizer, this module will be updated.
"""

from __future__ import annotations

from .common import FallbackCounter

_counter = FallbackCounter("gemini")


def count_tokens(text: str) -> int:
    """Estimate token count for Gemini models."""
    return _counter.count_tokens(text)


def is_exact() -> bool:
    """Return ``True`` if exact counting is available (currently never)."""
    return False
