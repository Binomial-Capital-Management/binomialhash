"""OpenAI token counting via ``tiktoken``.

Uses the ``o200k_base`` encoding (GPT-4o / GPT-4.1 / GPT-5 family).
Falls back to the character heuristic if ``tiktoken`` is not installed.

Install exact counting::

    pip install binomialhash[openai]   # once extras are wired
    # or directly:
    pip install tiktoken
"""

from __future__ import annotations

from .common import FallbackCounter, chars_fallback

_DEFAULT_ENCODING = "o200k_base"

try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding(_DEFAULT_ENCODING)
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _tiktoken = None  # type: ignore[assignment]
    _enc = None
    _TIKTOKEN_AVAILABLE = False


def count_tokens(text: str, *, encoding: str = _DEFAULT_ENCODING) -> int:
    """Return exact token count for *text* using tiktoken.

    Falls back to ``chars / 4`` if tiktoken is not installed.
    """
    if _TIKTOKEN_AVAILABLE and _enc is not None:
        # Reuse the pre-built encoder for the default encoding; create a fresh one for non-default encodings.
        if encoding == _DEFAULT_ENCODING:
            return len(_enc.encode(text))
        enc = _tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    return FallbackCounter("openai").count_tokens(text)


def is_exact() -> bool:
    """Return ``True`` if exact tiktoken counting is available."""
    return _TIKTOKEN_AVAILABLE
