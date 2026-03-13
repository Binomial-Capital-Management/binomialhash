"""Anthropic token estimation.

Anthropic does not publish an offline tokenizer.  Their API offers a
``/v1/messages/count_tokens`` endpoint, but that requires a network
round-trip and an API key, which is inappropriate for a local utility.

This module uses the character heuristic (``chars / 4``) as a
reasonable approximation for Claude's tokenizer on English / JSON text.

If Anthropic releases an offline tokenizer in the future, this module
will be updated to use it.
"""

from __future__ import annotations

from .common import FallbackCounter

_counter = FallbackCounter("anthropic")


def count_tokens(text: str) -> int:
    """Estimate token count for Anthropic / Claude models."""
    return _counter.count_tokens(text)


def is_exact() -> bool:
    """Return ``True`` if exact counting is available (currently never)."""
    return False
