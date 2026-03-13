"""Common token-counting interface and character-based fallback.

Every provider module implements a ``count_tokens(text) -> int`` function.
When the provider's native tokenizer is unavailable, the fallback
heuristic of ``ceil(len(text) / 4)`` is used with a logged warning.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4


class TokenCounter(Protocol):
    """Minimal interface every provider tokenizer must satisfy."""

    def count_tokens(self, text: str) -> int: ...


def chars_fallback(text: str) -> int:
    """Estimate token count from character length (ceil(len / 4)).

    This is a coarse heuristic — roughly correct for English prose and
    JSON, but can be off by 20-30 % on code or non-Latin scripts.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


class FallbackCounter:
    """Token counter that uses the character heuristic.

    Emits a one-time warning per provider name so users know exact
    counting is not active.
    """

    _warned: set = set()

    def __init__(self, provider: str = "unknown") -> None:
        self._provider = provider

    def count_tokens(self, text: str) -> int:
        if self._provider not in self._warned:
            logger.warning(
                "[BH-tokenizer] No native tokenizer for '%s'; "
                "using chars/4 heuristic.  Install the provider's "
                "tokenizer package for exact counts.",
                self._provider,
            )
            self._warned.add(self._provider)
        return chars_fallback(text)
