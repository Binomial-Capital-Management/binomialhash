"""xAI / Grok token counting.

Grok models use a tokenizer compatible with the OpenAI token-counting
family.  This module delegates to the OpenAI tokenizer (``tiktoken``)
when available, otherwise falls back to the character heuristic.
"""

from __future__ import annotations

from .openai import count_tokens as _openai_count
from .openai import is_exact as _openai_is_exact


def count_tokens(text: str) -> int:
    """Return token count for xAI / Grok models.

    Uses tiktoken (via the OpenAI tokenizer module) if available.
    """
    return _openai_count(text)


def is_exact() -> bool:
    """Return ``True`` if exact tiktoken counting is available."""
    return _openai_is_exact()
