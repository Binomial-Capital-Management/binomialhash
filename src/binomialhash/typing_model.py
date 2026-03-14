"""Explicit schema typing policy and decision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schema import (
    T_BOOL,
    T_DATE,
    T_DATETIME,
    T_DICT,
    T_LIST,
    T_MIXED,
    T_NULL,
    T_NUMERIC,
    T_STRING,
    SchemaDecision,
    SchemaFeatureProfile,
)


@dataclass(frozen=True)
class SchemaTypingPolicy:
    """Named policy values for heuristic schema typing."""

    mixed_max_best_score: float = 0.7
    mixed_min_second_score: float = 0.2
    structured_mixed_min_second_score: float = 0.05
    categorical_max_unique_ratio: float = 0.2
    categorical_min_unique_floor: int = 12
    categorical_max_unique_cap: int = 50
    identifier_min_unique_ratio: float = 0.9
    identifier_max_avg_length: float = 40.0
    identifier_validation_limit: int = 200
    free_text_min_avg_length: float = 48.0
    semantic_majority_ratio: float = 0.6
    record_like_min_unique_ratio: float = 0.9


DEFAULT_SCHEMA_TYPING_POLICY = SchemaTypingPolicy()


def score_candidates(profile: SchemaFeatureProfile) -> Dict[str, float]:
    if profile.non_null_count <= 0:
        return {T_NULL: 1.0}
    counts = profile.value_kind_counts
    return {
        name: round(count / profile.non_null_count, 6)
        for name, count in {
            T_NUMERIC: counts.get(T_NUMERIC, 0),
            T_BOOL: counts.get(T_BOOL, 0),
            T_DATE: counts.get(T_DATE, 0),
            T_DATETIME: counts.get(T_DATETIME, 0),
            T_DICT: counts.get(T_DICT, 0),
            T_LIST: counts.get(T_LIST, 0),
            T_STRING: counts.get(T_STRING, 0),
            T_MIXED: counts.get(T_MIXED, 0),
        }.items()
        if count > 0
    }


def classify_base_type(
    profile: SchemaFeatureProfile,
    scores: Dict[str, float],
    policy: SchemaTypingPolicy = DEFAULT_SCHEMA_TYPING_POLICY,
) -> Tuple[str, float]:
    if profile.non_null_count == 0:
        return T_NULL, 1.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Assign mixed when no single type dominates and a second type is significant.
    if (
        best_score < policy.mixed_max_best_score
        and second_score > policy.mixed_min_second_score
    ):
        return T_MIXED, round(best_score, 6)
    if (
        best_type in {T_DICT, T_LIST}
        and second_score > policy.structured_mixed_min_second_score
    ):
        return T_MIXED, round(best_score, 6)
    return best_type, round(best_score, 6)


def semantic_tags_for_profile(
    base_type: str,
    profile: SchemaFeatureProfile,
    identifier_regex,
    policy: SchemaTypingPolicy = DEFAULT_SCHEMA_TYPING_POLICY,
) -> List[str]:
    tags: List[str] = []
    unique_count = profile.unique_count
    unique_ratio = profile.unique_ratio
    avg_length = profile.avg_length
    non_null = profile.non_null_count
    counters = profile.value_kind_counts
    normalized_strings = profile.normalized_strings

    # Dynamic cap: min(50, max(12, 20% of rows)) prevents over-labeling high-cardinality strings as categorical.
    categorical_limit = min(
        policy.categorical_max_unique_cap,
        max(policy.categorical_min_unique_floor, int(non_null * policy.categorical_max_unique_ratio)),
    )
    if base_type in {T_STRING, T_DATE, T_DATETIME, T_BOOL} and unique_count <= categorical_limit:
        tags.append("categorical")
    if base_type == T_STRING and unique_ratio >= policy.identifier_min_unique_ratio and avg_length <= policy.identifier_max_avg_length:
        if normalized_strings and all(
            identifier_regex.match(value) and " " not in value
            for value in normalized_strings[: min(len(normalized_strings), policy.identifier_validation_limit)]
        ):
            tags.append("identifier")
    if base_type == T_STRING and avg_length >= policy.free_text_min_avg_length:
        tags.append("free_text")
    if counters.get("currency_like", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
        tags.append("currency")
    if counters.get("percent_like", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
        tags.append("percent")
    if counters.get("json_dict_string", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
        tags.append("json_stringified_dict")
    if counters.get("json_list_string", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
        tags.append("json_stringified_list")
    if base_type == T_LIST:
        if counters.get("list_of_dicts", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
            tags.append("list_of_dicts")
        if counters.get("list_of_scalars", 0) / max(non_null, 1) >= policy.semantic_majority_ratio:
            tags.append("list_of_scalars")
    if base_type == T_DICT and unique_ratio >= policy.record_like_min_unique_ratio:
        tags.append("record_like")
    return tags


def decision_from_profile(
    profile: SchemaFeatureProfile,
    identifier_regex,
    policy: SchemaTypingPolicy = DEFAULT_SCHEMA_TYPING_POLICY,
) -> SchemaDecision:
    scores = score_candidates(profile)
    base_type, confidence = classify_base_type(profile, scores, policy)
    return SchemaDecision(
        base_type=base_type,
        confidence=confidence,
        candidate_scores=scores,
        semantic_tags=semantic_tags_for_profile(base_type, profile, identifier_regex, policy),
    )
