"""Tests for binomialhash.tokenizers — counting, fallback, exact detection."""

from __future__ import annotations

import pytest


class TestCharsFallback:
    def test_empty_string(self):
        from binomialhash.tokenizers.common import chars_fallback
        assert chars_fallback("") == 0

    def test_short_string(self):
        from binomialhash.tokenizers.common import chars_fallback
        assert chars_fallback("hi") == 1

    def test_exact_multiple(self):
        from binomialhash.tokenizers.common import chars_fallback
        assert chars_fallback("abcd") == 1

    def test_rounds_up(self):
        from binomialhash.tokenizers.common import chars_fallback
        assert chars_fallback("abcde") == 2

    def test_longer_text(self):
        from binomialhash.tokenizers.common import chars_fallback
        text = "a" * 400
        assert chars_fallback(text) == 100


class TestFallbackCounter:
    def test_returns_int(self):
        from binomialhash.tokenizers.common import FallbackCounter
        c = FallbackCounter("test_provider")
        assert isinstance(c.count_tokens("hello world"), int)

    def test_consistent_with_chars_fallback(self):
        from binomialhash.tokenizers.common import FallbackCounter, chars_fallback
        c = FallbackCounter("test2")
        text = "The quick brown fox jumps over the lazy dog."
        assert c.count_tokens(text) == chars_fallback(text)


class TestOpenAITokenizer:
    def test_count_returns_positive_int(self):
        from binomialhash.tokenizers.openai import count_tokens
        n = count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0

    def test_is_exact_returns_bool(self):
        from binomialhash.tokenizers.openai import is_exact
        assert isinstance(is_exact(), bool)

    def test_longer_text_more_tokens(self):
        from binomialhash.tokenizers.openai import count_tokens
        short = count_tokens("hi")
        long = count_tokens("This is a much longer piece of text that should tokenize into more tokens.")
        assert long > short

    def test_empty_string(self):
        from binomialhash.tokenizers.openai import count_tokens
        assert count_tokens("") == 0


class TestAnthropicTokenizer:
    def test_count_returns_positive_int(self):
        from binomialhash.tokenizers.anthropic import count_tokens
        n = count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0

    def test_is_exact_returns_false(self):
        from binomialhash.tokenizers.anthropic import is_exact
        assert is_exact() is False


class TestGeminiTokenizer:
    def test_count_returns_positive_int(self):
        from binomialhash.tokenizers.gemini import count_tokens
        n = count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0

    def test_is_exact_returns_false(self):
        from binomialhash.tokenizers.gemini import is_exact
        assert is_exact() is False


class TestXAITokenizer:
    def test_matches_openai(self):
        from binomialhash.tokenizers.openai import count_tokens as openai_count
        from binomialhash.tokenizers.xai import count_tokens as xai_count
        text = "Test consistency between OpenAI and xAI tokenizers."
        assert xai_count(text) == openai_count(text)

    def test_is_exact_matches_openai(self):
        from binomialhash.tokenizers.openai import is_exact as openai_exact
        from binomialhash.tokenizers.xai import is_exact as xai_exact
        assert xai_exact() == openai_exact()


class TestTopLevelAPI:
    def test_count_tokens_openai(self):
        from binomialhash.tokenizers import count_tokens
        n = count_tokens("hello", provider="openai")
        assert isinstance(n, int)

    def test_count_tokens_anthropic(self):
        from binomialhash.tokenizers import count_tokens
        n = count_tokens("hello", provider="anthropic")
        assert isinstance(n, int)

    def test_count_tokens_gemini(self):
        from binomialhash.tokenizers import count_tokens
        n = count_tokens("hello", provider="gemini")
        assert isinstance(n, int)

    def test_count_tokens_xai(self):
        from binomialhash.tokenizers import count_tokens
        n = count_tokens("hello", provider="xai")
        assert isinstance(n, int)

    def test_unknown_provider_raises(self):
        from binomialhash.tokenizers import count_tokens
        with pytest.raises(ValueError, match="Unknown provider"):
            count_tokens("hello", provider="deepseek")

    def test_is_exact_returns_bool(self):
        from binomialhash.tokenizers import is_exact
        for p in ("openai", "anthropic", "gemini", "xai"):
            assert isinstance(is_exact(p), bool)

    def test_is_exact_unknown_returns_false(self):
        from binomialhash.tokenizers import is_exact
        assert is_exact("nonexistent") is False
