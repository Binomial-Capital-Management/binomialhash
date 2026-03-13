"""Provider-aware token counting for BinomialHash.

Provides a single entry point for estimating how many tokens a piece
of text will consume under a given provider's tokenizer.

Usage::

    from binomialhash.tokenizers import count_tokens, is_exact

    n = count_tokens("Hello world", provider="openai")
    if not is_exact("openai"):
        print("Using heuristic — install tiktoken for exact counts")

Exact counting requires optional dependencies:

* **OpenAI / xAI**: ``pip install tiktoken``
* **Anthropic**: no offline tokenizer available (heuristic only)
* **Gemini**: no offline tokenizer available (heuristic only)
"""

from __future__ import annotations

from typing import Any

from . import anthropic as _anthropic
from . import gemini as _gemini
from . import openai as _openai
from . import xai as _xai
from .common import CHARS_PER_TOKEN_ESTIMATE, FallbackCounter, chars_fallback

_PROVIDERS = {
    "openai": _openai,
    "anthropic": _anthropic,
    "gemini": _gemini,
    "xai": _xai,
}


def count_tokens(text: str, *, provider: str = "openai", **kwargs: Any) -> int:
    """Count (or estimate) tokens for *text* under *provider*'s tokenizer.

    Parameters
    ----------
    text:
        The string to measure.
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"gemini"``, ``"xai"``.
    **kwargs:
        Forwarded to the provider's ``count_tokens`` function
        (e.g. ``encoding="o200k_base"`` for OpenAI).

    Returns
    -------
    int
        Token count (exact when the provider library is installed,
        otherwise a ``ceil(chars / 4)`` estimate).
    """
    mod = _PROVIDERS.get(provider)
    if mod is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {sorted(_PROVIDERS.keys())}"
        )
    return mod.count_tokens(text, **kwargs)


def is_exact(provider: str = "openai") -> bool:
    """Return ``True`` if *provider* has an exact offline tokenizer loaded."""
    mod = _PROVIDERS.get(provider)
    if mod is None:
        return False
    return mod.is_exact()


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "FallbackCounter",
    "chars_fallback",
    "count_tokens",
    "is_exact",
]
