"""Schema inference and full-scan column statistics for BinomialHash."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

T_NUMERIC = "numeric"
T_STRING = "string"
T_DATE = "date"
T_DATETIME = "datetime"
T_BOOL = "bool"
T_DICT = "dict"
T_LIST = "list"
T_MIXED = "mixed"
T_NULL = "null"

# Stricter than float() to avoid false positives on dates, IDs, or version strings.
_STRICT_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}")
# Semantic subtype patterns — detected values still count as T_STRING but get tagged.
_CURRENCY_RE = re.compile(r"^\s*[$€£¥]\s*[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*$")
_PERCENT_RE = re.compile(r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)\s*%\s*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:/-]{2,64}$")


@dataclass
class SchemaFeatureProfile:
    """Deterministic, auditable feature summary for one column."""

    non_null_count: int
    unique_count: int
    avg_length: float
    string_entropy: float
    value_kind_counts: Dict[str, int]
    normalized_strings: List[str] = field(default_factory=list, repr=False)
    avg_list_length: Optional[float] = None
    max_list_length: Optional[int] = None
    top_dict_keys: List[str] = field(default_factory=list)

    @property
    def unique_ratio(self) -> float:
        return self.unique_count / max(self.non_null_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "non_null_count": self.non_null_count,
            "unique_count": self.unique_count,
            "unique_ratio": round(self.unique_ratio, 6),
            "avg_length": self.avg_length,
            "string_entropy": self.string_entropy,
            "value_kind_counts": self.value_kind_counts,
        }
        if self.avg_list_length is not None:
            out["avg_list_length"] = self.avg_list_length
        if self.max_list_length is not None:
            out["max_list_length"] = self.max_list_length
        if self.top_dict_keys:
            out["top_dict_keys"] = self.top_dict_keys
        return out


@dataclass
class SchemaDecision:
    """Final type decision and scored alternatives for one column."""

    base_type: str
    confidence: float
    candidate_scores: Dict[str, float]
    semantic_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_type": self.base_type,
            "confidence": self.confidence,
            "candidate_scores": self.candidate_scores,
            "semantic_tags": self.semantic_tags,
        }


def to_float_strict(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and _STRICT_NUMERIC_RE.match(stripped):
            try:
                return float(stripped)
            except ValueError:
                return None
    return None


def _parse_datetime(value: str) -> Optional[datetime]:
    stripped = value.strip()
    if not stripped:
        return None
    normalized = stripped.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized[:32])
    except (TypeError, ValueError):
        return None


def try_parse_date(value: str) -> bool:
    return _parse_datetime(value) is not None


def _parse_jsonish_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _safe_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 6)


def _score_candidates(non_null: int, counts: Dict[str, int]) -> Dict[str, float]:
    if non_null <= 0:
        return {T_NULL: 1.0}
    return {name: round(count / non_null, 6) for name, count in counts.items() if count > 0}


def _decision_from_profile(profile: SchemaFeatureProfile) -> SchemaDecision:
    # Import lazily to keep the policy model decoupled without creating a module cycle.
    from .typing_model import decision_from_profile

    return decision_from_profile(profile, _IDENTIFIER_RE)


def _column_profile(values: List[Any]) -> SchemaFeatureProfile:
    non_null_values = [value for value in values if value is not None and value != ""]
    non_null = len(non_null_values)
    if non_null == 0:
        return SchemaFeatureProfile(
            non_null_count=0,
            unique_count=0,
            avg_length=0.0,
            string_entropy=0.0,
            value_kind_counts={},
        )

    counters: Dict[str, int] = Counter()
    normalized_strings: List[str] = []
    string_counter: Counter[str] = Counter()
    list_lengths: List[int] = []
    dict_key_counter: Counter[str] = Counter()

    for value in non_null_values:
        if isinstance(value, bool):
            counters[T_BOOL] += 1
            continue
        if isinstance(value, dict):
            counters[T_DICT] += 1
            dict_key_counter.update(str(key) for key in value.keys())
            continue
        if isinstance(value, list):
            counters[T_LIST] += 1
            list_lengths.append(len(value))
            # Sample first 20 items to classify list contents without scanning huge arrays.
            if value and all(isinstance(item, dict) for item in value[: min(len(value), 20)]):
                counters["list_of_dicts"] += 1
            elif value and all(not isinstance(item, (dict, list)) for item in value[: min(len(value), 20)]):
                counters["list_of_scalars"] += 1
            continue
        if isinstance(value, (int, float)):
            counters[T_NUMERIC] += 1
            continue

        if isinstance(value, str):
            stripped = value.strip()
            normalized_strings.append(stripped)
            string_counter[stripped] += 1
            lowered = stripped.lower()
            if lowered in {"true", "false", "yes", "no", "0", "1", "y", "n", "t", "f"}:
                counters[T_BOOL] += 1
                continue
            parsed_json = _parse_jsonish_string(stripped)
            if isinstance(parsed_json, dict):
                counters["json_dict_string"] += 1
                counters[T_DICT] += 1
                dict_key_counter.update(str(key) for key in parsed_json.keys())
                continue
            if isinstance(parsed_json, list):
                counters["json_list_string"] += 1
                counters[T_LIST] += 1
                list_lengths.append(len(parsed_json))
                if parsed_json and all(isinstance(item, dict) for item in parsed_json[: min(len(parsed_json), 20)]):
                    counters["list_of_dicts"] += 1
                elif parsed_json and all(not isinstance(item, (dict, list)) for item in parsed_json[: min(len(parsed_json), 20)]):
                    counters["list_of_scalars"] += 1
                continue
            if to_float_strict(stripped) is not None:
                counters[T_NUMERIC] += 1
                continue
            parsed_dt = _parse_datetime(stripped)
            if parsed_dt is not None:
                if _DATETIME_RE.match(stripped):
                    counters[T_DATETIME] += 1
                elif _DATE_ONLY_RE.match(stripped):
                    counters[T_DATE] += 1
                else:
                    counters[T_DATETIME] += 1
                continue
            if _CURRENCY_RE.match(stripped):
                counters["currency_like"] += 1
            if _PERCENT_RE.match(stripped):
                counters["percent_like"] += 1
            counters[T_STRING] += 1
            continue

        counters[T_MIXED] += 1

    # Serialize to JSON for consistent hashing of heterogeneous types (dicts, lists, scalars).
    unique_count = len({json.dumps(value, sort_keys=True, default=str) for value in non_null_values})
    avg_length = (
        round(sum(len(value) for value in normalized_strings) / len(normalized_strings), 6)
        if normalized_strings
        else 0.0
    )
    return SchemaFeatureProfile(
        non_null_count=non_null,
        unique_count=unique_count,
        avg_length=avg_length,
        string_entropy=_safe_entropy(string_counter) if string_counter else 0.0,
        value_kind_counts=dict(counters),
        normalized_strings=normalized_strings,
        avg_list_length=round(sum(list_lengths) / len(list_lengths), 6) if list_lengths else None,
        max_list_length=max(list_lengths) if list_lengths else None,
        top_dict_keys=[item[0] for item in dict_key_counter.most_common(10)] if dict_key_counter else [],
    )


def compute_col_stats(
    rows: List[Dict[str, Any]],
    column: str,
    col_type: str,
    to_float: Callable[[Any], Optional[float]],
    profile: SchemaFeatureProfile,
    decision: SchemaDecision,
) -> Dict[str, Any]:
    values = [row.get(column) for row in rows]
    non_null = [value for value in values if value is not None and value != ""]
    stats: Dict[str, Any] = {
        "nulls": len(values) - len(non_null),
        "scanned": len(values),
        **profile.to_dict(),
        **decision.to_dict(),
    }

    if col_type == T_NUMERIC:
        numeric_values = [number for number in (to_float(value) for value in non_null) if number is not None]
        if numeric_values:
            sorted_values = sorted(numeric_values)
            stats["min"], stats["max"] = sorted_values[0], sorted_values[-1]
            stats["mean"] = round(sum(numeric_values) / len(numeric_values), 6)
            mid = len(sorted_values) // 2
            stats["median"] = (
                sorted_values[mid]
                if len(sorted_values) % 2
                else round((sorted_values[mid - 1] + sorted_values[mid]) / 2, 6)
            )
            if len(numeric_values) > 1:
                mean = sum(numeric_values) / len(numeric_values)
                stats["std"] = round(
                    math.sqrt(sum((value - mean) ** 2 for value in numeric_values) / len(numeric_values)),
                    6,
                )
    elif col_type in {T_STRING, T_BOOL, T_DATE, T_DATETIME}:
        frequencies = Counter(str(value) for value in non_null)
        stats["top_values"] = [item[0] for item in frequencies.most_common(5)]
        if col_type in {T_DATE, T_DATETIME}:
            sorted_values = sorted(str(value) for value in non_null if value)
            if sorted_values:
                stats["min_date"], stats["max_date"] = sorted_values[0], sorted_values[-1]
                if col_type == T_DATETIME:
                    stats["min_datetime"], stats["max_datetime"] = sorted_values[0], sorted_values[-1]
    elif col_type == T_LIST:
        stats["list_kind"] = (
            "list_of_dicts"
            if "list_of_dicts" in decision.semantic_tags
            else "list_of_scalars"
            if "list_of_scalars" in decision.semantic_tags
            else "list"
        )

    return stats


def infer_schema(
    rows: List[Dict[str, Any]],
    columns: List[str],
    to_float: Callable[[Any], Optional[float]],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    col_types: Dict[str, str] = {}
    col_stats: Dict[str, Dict[str, Any]] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        profile = _column_profile(values)
        decision = _decision_from_profile(profile)
        col_types[column] = decision.base_type
        col_stats[column] = compute_col_stats(rows, column, col_types[column], to_float, profile, decision)
    return col_types, col_stats
