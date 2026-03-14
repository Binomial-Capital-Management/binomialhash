"""Thread-safety and async-concurrency tests for BinomialHash."""

import asyncio
import json
import threading

import pytest

from binomialhash import BinomialHash
from binomialhash.core import MAX_SLOTS


def _make_payload(uid, n=200):
    return json.dumps(
        [{"id": i, "uid": uid, "v": i * 1.1, "s": "x" * 50} for i in range(n)]
    )


class TestThreadSafety:
    """Verify that concurrent threaded ingests respect MAX_SLOTS and keep
    ``_used_bytes`` consistent."""

    def test_slot_count_respects_max_slots(self):
        bh = BinomialHash()
        barrier = threading.Barrier(20)

        def worker(tid):
            for i in range(5):
                barrier.wait()
                bh.ingest(_make_payload(tid * 100 + i), f"data_{tid}_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(bh._slots) <= MAX_SLOTS

    def test_used_bytes_consistency(self):
        bh = BinomialHash()
        barrier = threading.Barrier(10)

        def worker(tid):
            for i in range(4):
                barrier.wait()
                bh.ingest(_make_payload(tid * 100 + i), f"bytes_{tid}_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = sum(s.byte_size for s in bh._slots.values())
        assert bh._used_bytes == expected

    def test_concurrent_retrieve_does_not_crash(self):
        bh = BinomialHash()
        key = None
        result = bh.ingest(_make_payload(1), "shared")
        for line in result.splitlines():
            if line.startswith("[BH] key="):
                key = line.split('"')[1]
                break
        assert key is not None

        errors = []

        def reader():
            try:
                for _ in range(50):
                    bh.retrieve(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestAsyncWrappers:
    """Verify async wrappers delegate correctly and handle concurrency."""

    @pytest.mark.asyncio
    async def test_aingest_basic(self):
        bh = BinomialHash()
        result = await bh.aingest(_make_payload(42), "async_test")
        assert "[BH]" in result
        assert len(bh._slots) == 1

    @pytest.mark.asyncio
    async def test_aingest_concurrent(self):
        bh = BinomialHash()
        coros = [
            bh.aingest(_make_payload(i), f"async_{i}") for i in range(60)
        ]
        await asyncio.gather(*coros)
        assert len(bh._slots) <= MAX_SLOTS

    @pytest.mark.asyncio
    async def test_aretrieve(self):
        bh = BinomialHash()
        summary = await bh.aingest(_make_payload(1), "for_retrieve")
        key = summary.splitlines()[0].split('"')[1]
        result = await bh.aretrieve(key)
        assert "rows" in result
        assert result["returned"] > 0

    @pytest.mark.asyncio
    async def test_aaggregate(self):
        bh = BinomialHash()
        summary = await bh.aingest(_make_payload(1), "for_agg")
        key = summary.splitlines()[0].split('"')[1]
        result = await bh.aaggregate(key, "v", "sum")
        assert "result" in result

    @pytest.mark.asyncio
    async def test_aschema(self):
        bh = BinomialHash()
        summary = await bh.aingest(_make_payload(1), "for_schema")
        key = summary.splitlines()[0].split('"')[1]
        result = await bh.aschema(key)
        assert "columns" in result
