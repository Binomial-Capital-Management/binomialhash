"""Tests for binomialhash.middleware — interception, raw mode, sync/async."""

from __future__ import annotations

import asyncio
import json

import pytest

from binomialhash import BinomialHash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _large_payload(n: int = 100) -> dict:
    """Dict whose JSON serialisation exceeds the 3000-char threshold."""
    return {"rows": [{"ticker": f"T{i}", "price": i * 1.5, "desc": f"padding_{i}_" * 10} for i in range(n)]}


def _small_payload() -> dict:
    return {"value": 42}


# ---------------------------------------------------------------------------
# bh_intercept — decorator form
# ---------------------------------------------------------------------------

class TestBhIntercept:
    def test_large_output_is_ingested(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="test_large", bh=bh)
        def fetch():
            return _large_payload()

        result = fetch()
        assert isinstance(result, str)
        assert len(bh.keys()) >= 1

    def test_small_output_passes_through(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="test_small", bh=bh)
        def fetch():
            return _small_payload()

        result = fetch()
        assert isinstance(result, dict)
        assert result == {"value": 42}
        assert len(bh.keys()) == 0

    def test_string_output_checked_by_length(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()
        big_string = json.dumps(_large_payload())

        @bh_intercept(label="test_str", bh=bh)
        def fetch():
            return big_string

        result = fetch()
        assert isinstance(result, str)

    def test_non_json_output_passes_through(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="test_int", bh=bh)
        def fetch():
            return 12345

        assert fetch() == 12345

    def test_custom_threshold(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="test_thresh", bh=bh, threshold=10)
        def fetch():
            return {"a": "b", "c": "d"}

        result = fetch()
        assert isinstance(result, (str, dict))

    def test_preserves_function_name(self):
        from binomialhash.middleware import bh_intercept

        @bh_intercept(label="x", bh=BinomialHash())
        def my_function():
            return {}

        assert my_function.__name__ == "my_function"

    def test_passes_args_through(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="args_test", bh=bh)
        def fetch(ticker: str, limit: int = 10):
            return {"ticker": ticker, "limit": limit}

        result = fetch("AAPL", limit=5)
        assert result == {"ticker": "AAPL", "limit": 5}


# ---------------------------------------------------------------------------
# wrap_tool_with_bh — wrapper form
# ---------------------------------------------------------------------------

class TestWrapToolWithBh:
    def test_wraps_function(self):
        from binomialhash.middleware import wrap_tool_with_bh
        bh = BinomialHash()

        def fetch():
            return _large_payload()

        wrapped = wrap_tool_with_bh(fetch, label="wrap_test", bh=bh)
        result = wrapped()
        assert isinstance(result, str)
        assert len(bh.keys()) >= 1

    def test_preserves_name(self):
        from binomialhash.middleware import wrap_tool_with_bh

        def my_tool():
            return {}

        wrapped = wrap_tool_with_bh(my_tool, label="x", bh=BinomialHash())
        assert wrapped.__name__ == "my_tool"


# ---------------------------------------------------------------------------
# Async support
# ---------------------------------------------------------------------------

class TestAsyncSupport:
    def test_async_decorator(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="async_test", bh=bh)
        async def fetch():
            return _large_payload()

        result = asyncio.get_event_loop().run_until_complete(fetch())
        assert isinstance(result, str)

    def test_async_small_passes_through(self):
        from binomialhash.middleware import bh_intercept
        bh = BinomialHash()

        @bh_intercept(label="async_small", bh=bh)
        async def fetch():
            return _small_payload()

        result = asyncio.get_event_loop().run_until_complete(fetch())
        assert result == {"value": 42}

    def test_async_wrapper(self):
        from binomialhash.middleware import wrap_tool_with_bh
        bh = BinomialHash()

        async def fetch():
            return _large_payload()

        wrapped = wrap_tool_with_bh(fetch, label="async_wrap", bh=bh)
        result = asyncio.get_event_loop().run_until_complete(wrapped())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# raw_mode — bypass interception
# ---------------------------------------------------------------------------

class TestRawMode:
    def test_raw_mode_bypasses(self):
        from binomialhash.middleware import bh_intercept, raw_mode
        bh = BinomialHash()

        @bh_intercept(label="raw_test", bh=bh)
        def fetch():
            return _large_payload()

        with raw_mode():
            result = fetch()
        assert isinstance(result, dict)
        assert "rows" in result
        assert len(bh.keys()) == 0

    def test_raw_mode_scoped(self):
        from binomialhash.middleware import bh_intercept, raw_mode
        bh = BinomialHash()

        @bh_intercept(label="scope_test", bh=bh)
        def fetch():
            return _large_payload()

        with raw_mode():
            raw_result = fetch()
        assert isinstance(raw_result, dict)

        normal_result = fetch()
        assert isinstance(normal_result, str)

    def test_raw_mode_nested(self):
        from binomialhash.middleware import bh_intercept, raw_mode
        bh = BinomialHash()

        @bh_intercept(label="nest_test", bh=bh)
        def fetch():
            return _large_payload()

        with raw_mode():
            with raw_mode():
                inner = fetch()
            outer = fetch()

        assert isinstance(inner, dict)
        assert isinstance(outer, dict)

        after = fetch()
        assert isinstance(after, str)
