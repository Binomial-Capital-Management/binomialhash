"""BinomialHash — Content-addressed, schema-aware in-memory data structure.

Intercepts large MCP/tool outputs, infers schema + stats, deduplicates by
content fingerprint, and returns a compact summary for the LLM.
"""

import asyncio
import hashlib
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .extract import NestingProfile, analyze_nesting, extract_rows
from .predicates import apply_sort_slice_project, build_predicate, sort_rows
from .schema import T_BOOL, T_DATE, T_NUMERIC, T_STRING, infer_schema, to_float_strict
from .stats import _ALL_AGG_FUNCS, _NUMERIC_FUNCS, run_agg, to_float_permissive
from ._stats_methods import _StatsMethodsMixin
from ._manifold_methods import _ManifoldMethodsMixin

logger = logging.getLogger(__name__)

INGEST_THRESHOLD_CHARS = 3000
MAX_PREVIEW_ROWS = 3
MAX_RETRIEVE_ROWS = 50
MAX_SLOTS = 50
BUDGET_BYTES = 50 * 1024 * 1024


@dataclass
class BinomialHashSlot:
    """Single dataset stored in the BinomialHash."""
    key: str
    label: str
    fingerprint: str
    rows: List[Dict[str, Any]]
    columns: List[str]
    col_types: Dict[str, str]
    col_stats: Dict[str, Dict[str, Any]]
    row_count: int
    byte_size: int
    nesting: Optional[NestingProfile] = None
    manifold: Any = None
    created_at: float = field(default_factory=time.monotonic)
    access_count: int = 0


@dataclass(frozen=True)
class BinomialHashPolicy:
    """Named policy values for core and analysis behaviour."""
    key_label_prefix_length: int = 20
    keys_preview_column_count: int = 10
    ingest_key_scan_row_count: int = 100
    ingest_max_column_count: int = 50
    summary_preview_column_count: int = 12
    summary_preview_char_limit: int = 1200
    error_preview_column_count: int = 20
    group_by_agg_limit: int = 20
    manifold_non_null_preview_column_count: int = 50
    manifold_diagnostic_preview_column_count: int = 20
    manifold_insights_default_top_k: int = 5
    manifold_insights_driver_limit: int = 20
    manifold_insights_driver_bins: int = 10
    manifold_insights_target_bins: int = 6
    manifold_insights_regime_z_threshold: float = 1.5
    manifold_insights_branch_context_limit: int = 8
    manifold_insights_branch_min_rows: int = 20
    manifold_insights_branch_min_values: int = 10
    orbit_default_resolution: int = 16
    multiscale_default_resolution: int = 16
    # Export row caps — enforced by tool handlers, not the exporter functions
    export_csv_max_rows: int = 50_000
    export_excel_max_rows: int = 10_000
    export_markdown_max_rows: int = 200
    export_rows_max_rows: int = 50_000


DEFAULT_BINOMIAL_HASH_POLICY = BinomialHashPolicy()


def _to_float(v: Any) -> Optional[float]:
    return to_float_permissive(v)


def _fmt_num(n: Any) -> str:
    if n is None:
        return "?"
    f = float(n)
    if not math.isfinite(f):
        return str(f)
    if abs(f) >= 1e9:
        return f"{f / 1e9:.1f}B"
    if abs(f) >= 1e6:
        return f"{f / 1e6:.1f}M"
    if abs(f) >= 1e3:
        return f"{f / 1e3:.1f}K"
    return str(int(f)) if f == int(f) else f"{f:.2f}"


class BinomialHash(_StatsMethodsMixin, _ManifoldMethodsMixin):
    """Content-addressed, schema-aware data store.  One per request."""

    __slots__ = ("_lock", "_slots", "_fingerprints", "_used_bytes",
                 "_ctx_chars_in", "_ctx_chars_out", "_ctx_tool_calls", "_policy")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: Dict[str, BinomialHashSlot] = {}
        self._fingerprints: Dict[str, str] = {}
        self._used_bytes: int = 0
        self._ctx_chars_in: int = 0
        self._ctx_chars_out: int = 0
        self._ctx_tool_calls: int = 0
        self._policy = DEFAULT_BINOMIAL_HASH_POLICY

    @staticmethod
    def _make_key(label: str, fp: str,
                  prefix_length: int = DEFAULT_BINOMIAL_HASH_POLICY.key_label_prefix_length) -> str:
        clean = "".join(c if c.isalnum() else "_" for c in label[:prefix_length]).strip("_").lower()
        return f"{clean}_{fp[:6]}"

    @staticmethod
    def _fingerprint(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _estimate_bytes(rows: List[Dict]) -> int:
        return sys.getsizeof(rows) + sum(sys.getsizeof(r) for r in rows[:100]) * max(1, len(rows) // 100)

    def _evict_if_needed(self, needed: int) -> None:
        while self._used_bytes + needed > BUDGET_BYTES and self._slots:
            vk = min(self._slots, key=lambda k: (self._slots[k].access_count, self._slots[k].created_at))
            v = self._slots.pop(vk)
            self._fingerprints.pop(v.fingerprint, None)
            self._used_bytes -= v.byte_size
            logger.info("[BH] evicted '%s' (%d bytes)", vk, v.byte_size)

    def _get_slot(self, key: str) -> Optional[BinomialHashSlot]:
        with self._lock:
            slot = self._slots.get(key)
            if slot:
                slot.access_count += 1
            return slot

    def keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"key": s.key, "label": s.label, "row_count": s.row_count,
                 "columns": s.columns[:self._policy.keys_preview_column_count]}
                for s in self._slots.values()
            ]

    def _track(self, chars_in: int, chars_out: int) -> None:
        with self._lock:
            self._ctx_chars_in += chars_in
            self._ctx_chars_out += chars_out
            self._ctx_tool_calls += 1

    def context_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tool_calls": self._ctx_tool_calls,
                "chars_in_raw": self._ctx_chars_in,
                "chars_out_to_llm": self._ctx_chars_out,
                "compression_ratio": round(self._ctx_chars_in / max(self._ctx_chars_out, 1), 1),
                "est_tokens_out": self._ctx_chars_out // 4,
                "slots": len(self._slots),
                "mem_bytes": self._used_bytes,
            }

    def log_summary(self) -> None:
        with self._lock:
            s = self.context_stats()
            slot_details = ", ".join(
                f"{sl.key}({sl.row_count}r/{sl.access_count}acc)"
                for sl in self._slots.values()
            ) or "(empty)"
            logger.info(
                "[BH-perf] REQUEST | %d calls | in=%d → out=%d (%.1fx, ~%d tok) | %d slots %.1fMB | %s",
                s["tool_calls"], s["chars_in_raw"], s["chars_out_to_llm"],
                s["compression_ratio"], s["est_tokens_out"],
                s["slots"], s["mem_bytes"] / 1e6, slot_details,
            )

    # -- ingest --

    def ingest(self, raw_text: str, label: str) -> str:
        """Ingest tool output.  Returns compact summary if stored, raw text if small."""
        t0 = time.perf_counter()
        raw_len = len(raw_text)

        if raw_len <= INGEST_THRESHOLD_CHARS:
            self._track(raw_len, raw_len)
            return raw_text

        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            out = raw_text[:INGEST_THRESHOLD_CHARS] + "\n... [truncated]"
            self._track(raw_len, len(out))
            return out

        nesting = analyze_nesting(data)
        rows, meta = extract_rows(data)
        if not rows or len(rows) < 3:
            out = raw_text[:INGEST_THRESHOLD_CHARS] + "\n... [truncated]"
            self._track(raw_len, len(out))
            return out

        fp = self._fingerprint(raw_text)

        with self._lock:
            if fp in self._fingerprints:
                existing = self._fingerprints[fp]
                self._slots[existing].access_count += 1
                summary = self._build_summary(self._slots[existing])
                self._track(raw_len, len(summary))
                return summary

            if len(self._slots) >= MAX_SLOTS:
                self._evict_if_needed(0)
                if len(self._slots) >= MAX_SLOTS:
                    out = raw_text[:INGEST_THRESHOLD_CHARS] + "\n... [truncated, store full]"
                    self._track(raw_len, len(out))
                    return out

            columns = list(
                dict.fromkeys(
                    k for r in rows[:self._policy.ingest_key_scan_row_count] for k in r.keys()
                )
            )[:self._policy.ingest_max_column_count]

            col_types, col_stats = infer_schema(rows, columns, _to_float)
            byte_size = self._estimate_bytes(rows)
            if byte_size > BUDGET_BYTES:
                logger.warning("[BH] payload '%s' exceeds budget (%d > %d), returning text summary",
                               label, byte_size, BUDGET_BYTES)
                out = raw_text[:INGEST_THRESHOLD_CHARS] + "\n... [payload too large to cache]"
                self._track(raw_len, len(out))
                return out
            self._evict_if_needed(byte_size)
            key = self._make_key(label, fp, self._policy.key_label_prefix_length)

            from .manifold.builder import build_manifold
            manifold = build_manifold(rows, columns, col_types, col_stats)

            slot = BinomialHashSlot(
                key=key, label=label, fingerprint=fp, rows=rows, columns=columns,
                col_types=col_types, col_stats=col_stats, row_count=len(rows),
                byte_size=byte_size, nesting=nesting, manifold=manifold,
            )
            self._slots[key] = slot
            self._fingerprints[fp] = key
            self._used_bytes += byte_size
            summary = self._build_summary(slot)
            self._track(raw_len, len(summary))
            logger.info(
                "[BH-perf] ingest '%s' → '%s' | %d rows %d cols | %.0fx compression | %.1fms",
                label, key, len(rows), len(columns),
                raw_len / max(len(summary), 1), (time.perf_counter() - t0) * 1000,
            )
            return summary

    def _build_summary(self, slot: BinomialHashSlot) -> str:
        parts = []
        for col in slot.columns[:self._policy.summary_preview_column_count]:
            ct = slot.col_types.get(col, "?")
            st = slot.col_stats.get(col, {})
            detail = ""
            if ct == T_NUMERIC and "min" in st:
                detail = f", {_fmt_num(st['min'])}..{_fmt_num(st['max'])}"
            elif ct == T_STRING and "unique_count" in st:
                detail = f", {st['unique_count']} unique"
            elif ct == T_DATE and "min_date" in st:
                detail = f", {st['min_date'][:10]}..{st['max_date'][:10]}"
            parts.append(f"{col}({ct[:3]}{detail})")
        schema_line = ", ".join(parts)
        if len(slot.columns) > self._policy.summary_preview_column_count:
            schema_line += f" +{len(slot.columns) - self._policy.summary_preview_column_count} more"
        preview = json.dumps(slot.rows[:MAX_PREVIEW_ROWS], default=str)
        if len(preview) > self._policy.summary_preview_char_limit:
            preview = preview[:self._policy.summary_preview_char_limit] + "...]"
        return (
            f"[BH] key=\"{slot.key}\" | {slot.row_count} records | {slot.label}\n"
            f"Schema: {schema_line}\n"
            f"Preview: {preview}\n"
            f"Use bh_retrieve/bh_query/bh_aggregate/bh_group_by/bh_to_excel. Do NOT re-fetch."
        )

    # -- retrieval --

    def retrieve(self, key: str, offset: int = 0, limit: int = 25,
                 sort_by: Optional[str] = None, sort_desc: bool = True,
                 columns: Optional[List[str]] = None) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found. Available: {list(self._slots.keys())}"}
            rows = slot.rows
            if sort_by and sort_by in slot.col_types:
                rows = sort_rows(rows, sort_by, slot.col_types[sort_by], sort_desc)
            effective_limit = min(limit, MAX_RETRIEVE_ROWS)
            sliced = rows[offset:offset + effective_limit]
            if columns:
                col_set = set(columns)
                sliced = [{k: v for k, v in r.items() if k in col_set} for r in sliced]
            result = {"key": key, "label": slot.label, "total_rows": slot.row_count,
                      "offset": offset, "returned": len(sliced), "rows": sliced}
            out_chars = len(json.dumps(result, default=str))
            self._track(0, out_chars)
            logger.info("[BH-perf] retrieve '%s' %d/%d rows | %.1fms",
                        key, len(sliced), slot.row_count, (time.perf_counter() - t0) * 1000)
            return result

    def aggregate(self, key: str, column: str, func: str) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found."}
            if column not in slot.col_types:
                return {"error": f"Column '{column}' not found. Available: {slot.columns[:self._policy.error_preview_column_count]}"}
            if func not in _ALL_AGG_FUNCS:
                return {"error": f"Unknown func '{func}'. Use: {', '.join(sorted(_ALL_AGG_FUNCS))}"}
            result_val = run_agg(slot.rows, column, func)
            out = {"key": key, "column": column, "func": func, "result": result_val}
            self._track(0, len(json.dumps(out, default=str)))
            logger.info("[BH-perf] aggregate '%s' %s(%s)=%s | %.1fms",
                        key, func, column, result_val, (time.perf_counter() - t0) * 1000)
            return out

    def query(self, key: str, where_json: str, sort_by: Optional[str] = None,
              sort_desc: bool = True, limit: int = 25,
              columns: Optional[List[str]] = None) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found."}
            try:
                where = json.loads(where_json)
            except (json.JSONDecodeError, TypeError):
                return {"error": f"Invalid where_json: {where_json[:100]}"}
            predicate = build_predicate(where, slot.col_types)
            if predicate is None:
                return {"error": f"Invalid where clause: {where_json[:200]}"}
            filtered = [r for r in slot.rows if predicate(r)]
            sliced = apply_sort_slice_project(filtered, slot, sort_by, sort_desc, limit, columns, MAX_RETRIEVE_ROWS)
            result = {"key": key, "label": slot.label, "total_rows": slot.row_count,
                      "matched": len(filtered), "returned": len(sliced), "rows": sliced}
            self._track(0, len(json.dumps(result, default=str)))
            logger.info("[BH-perf] query '%s' %d→%d→%d | %.1fms",
                        key, slot.row_count, len(filtered), len(sliced),
                        (time.perf_counter() - t0) * 1000)
            return result

    def group_by(self, key: str, group_cols: List[str], agg_json: str,
                 sort_by: Optional[str] = None, sort_desc: bool = True,
                 limit: int = 50) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found."}
            for gc in group_cols:
                if gc not in slot.col_types:
                    return {"error": f"Group column '{gc}' not found. Available: {slot.columns[:self._policy.error_preview_column_count]}"}
            try:
                aggs = json.loads(agg_json)
            except (json.JSONDecodeError, TypeError):
                return {"error": f"Invalid agg_json: {agg_json[:100]}"}
            if not isinstance(aggs, list) or not aggs:
                return {"error": "agg_json must be a non-empty list of {column, func} objects."}
            groups: Dict[str, List[Dict]] = {}
            for row in slot.rows:
                gk = "|".join(str(row.get(gc, "")) for gc in group_cols)
                groups.setdefault(gk, []).append(row)
            result_rows = []
            for grp_rows in groups.values():
                out = {gc: grp_rows[0].get(gc) for gc in group_cols}
                for agg in aggs[:self._policy.group_by_agg_limit]:
                    alias = agg.get("alias", f"{agg.get('func', 'count')}_{agg.get('column', '')}")
                    out[alias] = run_agg(grp_rows, agg.get("column", ""), agg.get("func", "count"))
                result_rows.append(out)
            if sort_by:
                is_num = any(a.get("func") in _NUMERIC_FUNCS for a in aggs
                             if a.get("alias", f"{a.get('func', '')}_{a.get('column', '')}") == sort_by)
                if is_num:
                    result_rows.sort(key=lambda r: _to_float(r.get(sort_by)) or 0, reverse=sort_desc)
                else:
                    result_rows.sort(key=lambda r: str(r.get(sort_by, "")), reverse=sort_desc)
            sliced = result_rows[:min(limit, MAX_RETRIEVE_ROWS)]
            result = {"key": key, "label": slot.label, "total_rows": slot.row_count,
                      "groups": len(groups), "returned": len(sliced), "rows": sliced}
            self._track(0, len(json.dumps(result, default=str)))
            logger.info("[BH-perf] group_by '%s' by=%s %d groups | %.1fms",
                        key, group_cols, len(groups), (time.perf_counter() - t0) * 1000)
            return result

    def to_excel_batch(self, key: str, columns: Optional[List[str]] = None,
                       sort_by: Optional[str] = None, sort_desc: bool = True,
                       max_rows: int = 200) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found."}
            from .exporters.excel import export_excel_batch
            capped = min(max_rows, self._policy.export_excel_max_rows)
            result = export_excel_batch(
                slot.rows, slot.columns, slot.col_types, key, slot.label, slot.row_count,
                select_columns=columns, sort_by=sort_by, sort_desc=sort_desc, max_rows=capped,
            )
            self._track(0, len(json.dumps(result, default=str)))
            logger.info("[BH-perf] excel '%s' | %.1fms",
                        key, (time.perf_counter() - t0) * 1000)
            return result

    def schema(self, key: str) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            slot = self._get_slot(key)
            if slot is None:
                return {"error": f"Key '{key}' not found."}
            result = {"key": key, "label": slot.label, "row_count": slot.row_count,
                      "byte_size": slot.byte_size, "columns": slot.columns,
                      "col_types": slot.col_types, "col_stats": slot.col_stats}
            self._track(0, len(json.dumps(result, default=str)))
            logger.info("[BH-perf] schema '%s' | %.1fms",
                        key, (time.perf_counter() - t0) * 1000)
            return result

    # -- export --

    def to_chunks(self) -> List[Dict[str, Any]]:
        with self._lock:
            from .exporters.chunks import slot_to_chunk
            return [
                slot_to_chunk(
                    slot.key, slot.label, slot.fingerprint, slot.row_count,
                    slot.rows, slot.columns, slot.col_types, slot.col_stats,
                )
                for slot in self._slots.values()
            ]

    # -- async wrappers --

    async def aingest(self, raw_text: str, label: str) -> str:
        return await asyncio.to_thread(self.ingest, raw_text, label)

    async def aretrieve(self, key: str, offset: int = 0, limit: int = 25,
                        sort_by: Optional[str] = None, sort_desc: bool = True,
                        columns: Optional[List[str]] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.retrieve, key, offset, limit, sort_by, sort_desc, columns,
        )

    async def aaggregate(self, key: str, column: str, func: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.aggregate, key, column, func)

    async def aquery(self, key: str, where_json: str, sort_by: Optional[str] = None,
                     sort_desc: bool = True, limit: int = 25,
                     columns: Optional[List[str]] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.query, key, where_json, sort_by, sort_desc, limit, columns,
        )

    async def agroup_by(self, key: str, group_cols: List[str], agg_json: str,
                        sort_by: Optional[str] = None, sort_desc: bool = True,
                        limit: int = 50) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.group_by, key, group_cols, agg_json, sort_by, sort_desc, limit,
        )

    async def aschema(self, key: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.schema, key)

    async def ato_excel_batch(self, key: str, columns: Optional[List[str]] = None,
                              sort_by: Optional[str] = None, sort_desc: bool = True,
                              max_rows: int = 200) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.to_excel_batch, key, columns, sort_by, sort_desc, max_rows,
        )
