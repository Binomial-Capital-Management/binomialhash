"""Tests for new exporters — csv, markdown, rows, artifact — and export tools."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _sample_rows(n: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [
        {"ticker": f"T{i:03d}", "price": round(rng.uniform(10, 500), 2),
         "volume": rng.randint(100_000, 90_000_000), "sector": f"S{i % 5}"}
        for i in range(n)
    ]

ROWS = _sample_rows()
COLUMNS = ["ticker", "price", "volume", "sector"]
COL_TYPES = {"ticker": "string", "price": "numeric", "volume": "numeric", "sector": "string"}


# ---------------------------------------------------------------------------
# exporters/csv.py
# ---------------------------------------------------------------------------

class TestExportCSV:
    def test_basic_csv(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv(ROWS, COLUMNS, COL_TYPES)
        lines = out.strip().split("\n")
        assert lines[0] == "ticker,price,volume,sector"
        assert len(lines) == 31  # header + 30 rows

    def test_column_selection(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv(ROWS, COLUMNS, COL_TYPES, select_columns=["ticker", "price"])
        header = out.strip().split("\n")[0]
        assert header == "ticker,price"

    def test_max_rows(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv(ROWS, COLUMNS, COL_TYPES, max_rows=5)
        lines = out.strip().split("\n")
        assert len(lines) == 6  # header + 5

    def test_no_header(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv(ROWS, COLUMNS, COL_TYPES, include_header=False, max_rows=3)
        lines = out.strip().split("\n")
        assert len(lines) == 3

    def test_sorting(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv(ROWS, COLUMNS, COL_TYPES, sort_by="price", sort_desc=True)
        lines = out.strip().split("\n")
        prices = [float(l.split(",")[1]) for l in lines[1:]]
        assert prices == sorted(prices, reverse=True)

    def test_empty_rows(self):
        from binomialhash.exporters.csv import export_csv
        out = export_csv([], COLUMNS, COL_TYPES)
        assert out.strip() == "ticker,price,volume,sector"


# ---------------------------------------------------------------------------
# exporters/markdown.py
# ---------------------------------------------------------------------------

class TestExportMarkdown:
    def test_basic_table(self):
        from binomialhash.exporters.markdown import export_markdown
        out = export_markdown(ROWS, COLUMNS, COL_TYPES, max_rows=5)
        lines = out.strip().split("\n")
        assert lines[0].startswith("| ticker")
        assert "---:" in lines[1]  # numeric alignment
        assert ":---" in lines[1]  # string alignment
        assert len([l for l in lines if l.startswith("|")]) == 7  # header + sep + 5 rows

    def test_footer_when_capped(self):
        from binomialhash.exporters.markdown import export_markdown
        out = export_markdown(ROWS, COLUMNS, COL_TYPES, max_rows=5, total_rows=30, label="market")
        assert "Showing 5 of 30" in out
        assert "market" in out

    def test_no_footer_when_all_shown(self):
        from binomialhash.exporters.markdown import export_markdown
        out = export_markdown(ROWS[:3], COLUMNS, COL_TYPES, max_rows=10)
        assert "Showing" not in out

    def test_column_selection(self):
        from binomialhash.exporters.markdown import export_markdown
        out = export_markdown(ROWS, COLUMNS, COL_TYPES, select_columns=["ticker", "price"], max_rows=3)
        header = out.split("\n")[0]
        assert "volume" not in header

    def test_truncation_of_long_values(self):
        from binomialhash.exporters.markdown import export_markdown
        long_rows = [{"text": "A" * 100}]
        out = export_markdown(long_rows, ["text"], {"text": "string"}, max_cell_width=20)
        assert "\u2026" in out

    def test_empty_rows(self):
        from binomialhash.exporters.markdown import export_markdown
        out = export_markdown([], COLUMNS, COL_TYPES)
        lines = [l for l in out.split("\n") if l.startswith("|")]
        assert len(lines) == 2  # header + alignment only


# ---------------------------------------------------------------------------
# exporters/rows.py
# ---------------------------------------------------------------------------

class TestExportRows:
    def test_basic_export(self):
        from binomialhash.exporters.rows import export_rows
        out = export_rows(ROWS, COLUMNS, COL_TYPES)
        assert isinstance(out, list)
        assert len(out) == 30

    def test_column_selection(self):
        from binomialhash.exporters.rows import export_rows
        out = export_rows(ROWS, COLUMNS, COL_TYPES, select_columns=["ticker"])
        for row in out:
            assert list(row.keys()) == ["ticker"]

    def test_offset_and_limit(self):
        from binomialhash.exporters.rows import export_rows
        out = export_rows(ROWS, COLUMNS, COL_TYPES, offset=5, limit=3)
        assert len(out) == 3

    def test_sorting(self):
        from binomialhash.exporters.rows import export_rows
        out = export_rows(ROWS, COLUMNS, COL_TYPES, sort_by="price", sort_desc=False)
        prices = [r["price"] for r in out]
        assert prices == sorted(prices)

    def test_returns_copies(self):
        from binomialhash.exporters.rows import export_rows
        out = export_rows(ROWS, COLUMNS, COL_TYPES, limit=1)
        out[0]["ticker"] = "MUTATED"
        assert ROWS[0]["ticker"] != "MUTATED"


# ---------------------------------------------------------------------------
# exporters/artifact.py
# ---------------------------------------------------------------------------

class TestBuildArtifact:
    def test_csv_artifact(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="csv", label="test_data")
        assert art["type"] == "artifact"
        assert art["filename"] == "test_data.csv"
        assert art["mime_type"] == "text/csv"
        assert "ticker" in art["content"]
        assert art["format"] == "csv"

    def test_markdown_artifact(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="markdown", label="md_test")
        assert art["mime_type"] == "text/markdown"
        assert art["filename"] == "md_test.md"
        assert "| ticker" in art["content"]

    def test_json_artifact(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="json", label="json_test")
        assert art["mime_type"] == "application/json"
        parsed = json.loads(art["content"])
        assert isinstance(parsed, list)
        assert len(parsed) == 30

    def test_jsonl_artifact(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="jsonl", label="jsonl_test")
        assert art["mime_type"] == "application/x-ndjson"
        lines = art["content"].strip().split("\n")
        assert len(lines) == 30
        json.loads(lines[0])

    def test_unknown_format_raises(self):
        from binomialhash.exporters.artifact import build_artifact
        with pytest.raises(ValueError, match="Unknown format"):
            build_artifact(ROWS, COLUMNS, COL_TYPES, format="parquet")

    def test_filename_sanitisation(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="csv", label="bad/name here!")
        assert "/" not in art["filename"]
        assert " " not in art["filename"]

    def test_column_selection_forwarded(self):
        from binomialhash.exporters.artifact import build_artifact
        art = build_artifact(ROWS, COLUMNS, COL_TYPES, format="csv",
                             label="cols", select_columns=["ticker", "price"])
        header = art["content"].split("\n")[0]
        assert "volume" not in header


# ---------------------------------------------------------------------------
# Export tools integration — via BinomialHash
# ---------------------------------------------------------------------------

def _make_bh_with_data() -> "BinomialHash":
    from binomialhash import BinomialHash
    bh = BinomialHash()
    rows = [
        {"ticker": f"T{i}", "price": i * 2.5, "vol": i * 100,
         "desc": f"padding_text_{i}_" * 8}
        for i in range(80)
    ]
    payload = json.dumps(rows)
    assert len(payload) > 3000
    bh.ingest(payload, "tool_test")
    return bh


def _get_key(bh) -> str:
    keys = bh.keys()
    assert len(keys) >= 1
    return keys[0]["key"]


class TestExportToolSpecs:
    def test_export_tools_registered(self):
        from binomialhash.tools import get_all_tools, get_tools_by_group
        bh = _make_bh_with_data()
        all_tools = get_all_tools(bh)
        export_tools = get_tools_by_group(bh, "export")
        assert len(export_tools) == 3
        names = {t.name for t in export_tools}
        assert names == {"bh_to_csv", "bh_to_markdown", "bh_export"}

    def test_total_tool_count(self):
        from binomialhash.tools import get_all_tools
        bh = _make_bh_with_data()
        assert len(get_all_tools(bh)) == 68

    def test_bh_to_csv_tool(self):
        from binomialhash.tools import get_tools_by_group
        bh = _make_bh_with_data()
        key = _get_key(bh)
        csv_tool = [t for t in get_tools_by_group(bh, "export") if t.name == "bh_to_csv"][0]
        result = csv_tool.handler(key=key)
        assert result["type"] == "artifact"
        assert result["mime_type"] == "text/csv"

    def test_bh_to_markdown_tool(self):
        from binomialhash.tools import get_tools_by_group
        bh = _make_bh_with_data()
        key = _get_key(bh)
        md_tool = [t for t in get_tools_by_group(bh, "export") if t.name == "bh_to_markdown"][0]
        result = md_tool.handler(key=key)
        assert isinstance(result, str)
        assert "| ticker" in result or "| " in result

    def test_bh_export_tool_json(self):
        from binomialhash.tools import get_tools_by_group
        bh = _make_bh_with_data()
        key = _get_key(bh)
        export_tool = [t for t in get_tools_by_group(bh, "export") if t.name == "bh_export"][0]
        result = export_tool.handler(key=key, format="json")
        assert result["type"] == "artifact"
        assert result["mime_type"] == "application/json"

    def test_missing_key_returns_error(self):
        from binomialhash.tools import get_tools_by_group
        bh = _make_bh_with_data()
        csv_tool = [t for t in get_tools_by_group(bh, "export") if t.name == "bh_to_csv"][0]
        result = csv_tool.handler(key="nonexistent_key")
        assert "error" in result
