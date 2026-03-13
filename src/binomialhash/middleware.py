"""Middleware for automatic BinomialHash interception of large tool outputs.

Provides a decorator and a wrapper that sit between a tool function and
the LLM.  When the tool returns a large structured payload (dict/list
whose JSON serialisation exceeds *threshold* characters), the middleware
ingests it into a BinomialHash instance and returns the compact summary.
Small or non-structured outputs pass through unchanged.

A contextvar-based **raw mode** lets callers bypass interception when
they need the native payload (e.g. for internal modelling or provider-
shape-sensitive data fetches).

Usage — decorator form::

    from binomialhash.middleware import bh_intercept

    @bh_intercept(label="market_data")
    def fetch_data(ticker: str) -> dict:
        return huge_json_response      # auto-compacted if large

Usage — wrapper form (for third-party / dynamically-registered tools)::

    from binomialhash.middleware import wrap_tool_with_bh

    wrapped = wrap_tool_with_bh(fetch_data, label="market_data")

Usage — raw-mode bypass::

    from binomialhash.middleware import raw_mode

    with raw_mode():
        result = fetch_data(ticker)    # returns native payload
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, overload

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 3000

_raw_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "bh_raw_mode",
    default=False,
)

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def raw_mode():
    """Context manager that disables BH interception for the enclosed scope.

    Thread-safe and async-safe (uses ``contextvars``).  Nested calls are
    fine — the outer scope's value is restored on exit.
    """
    token = _raw_mode.set(True)
    try:
        yield
    finally:
        _raw_mode.reset(token)


def _resolve_bh(explicit: Any) -> Any:
    """Return *explicit* if given, otherwise fall back to context.get_binomial_hash()."""
    if explicit is not None:
        return explicit
    from .context import get_binomial_hash
    return get_binomial_hash()


def _maybe_ingest(result: Any, *, label: str, bh: Any, threshold: int) -> Any:
    """Inspect *result* and ingest into BH if it is large structured data."""
    if _raw_mode.get(False):
        return result

    if isinstance(result, (dict, list)):
        try:
            serialised = json.dumps(result, default=str)
        except (TypeError, ValueError):
            return result
    elif isinstance(result, str):
        serialised = result
    else:
        return result

    if len(serialised) <= threshold:
        return result

    instance = _resolve_bh(bh)
    try:
        summary = instance.ingest(serialised, label)
        logger.debug(
            "[BH-middleware] ingested %d chars under label '%s'",
            len(serialised),
            label,
        )
        return summary
    except Exception:
        logger.warning(
            "[BH-middleware] ingest failed for label '%s', passing through",
            label,
            exc_info=True,
        )
        return result


def wrap_tool_with_bh(
    fn: F,
    *,
    label: str,
    bh: Any = None,
    threshold: int = _DEFAULT_THRESHOLD,
) -> F:
    """Wrap *fn* so its return value is auto-ingested when large.

    Works for both sync and async callables.

    Parameters
    ----------
    fn:
        The original tool function.
    label:
        BH storage key label used during ingest.
    bh:
        Explicit ``BinomialHash`` instance.  Falls back to the
        request-scoped instance from ``context.get_binomial_hash()``.
    threshold:
        Minimum character count for ingestion to kick in.
    """
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await fn(*args, **kwargs)
            return _maybe_ingest(result, label=label, bh=bh, threshold=threshold)
        return _async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return _maybe_ingest(result, label=label, bh=bh, threshold=threshold)
    return _sync_wrapper  # type: ignore[return-value]


def bh_intercept(
    *,
    label: str,
    bh: Any = None,
    threshold: int = _DEFAULT_THRESHOLD,
) -> Callable[[F], F]:
    """Decorator that auto-ingests large structured return values into BH.

    Parameters
    ----------
    label:
        BH storage key label used during ingest.
    bh:
        Explicit ``BinomialHash`` instance.  Falls back to the
        request-scoped instance from ``context.get_binomial_hash()``.
    threshold:
        Minimum character count for ingestion to kick in.

    Example::

        @bh_intercept(label="market_data")
        def fetch_data(ticker: str) -> dict:
            return huge_json_response
    """
    def decorator(fn: F) -> F:
        return wrap_tool_with_bh(fn, label=label, bh=bh, threshold=threshold)
    return decorator


__all__ = [
    "bh_intercept",
    "raw_mode",
    "wrap_tool_with_bh",
]
