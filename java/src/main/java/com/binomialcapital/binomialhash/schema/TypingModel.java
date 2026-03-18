package com.binomialcapital.binomialhash.schema;

import java.util.*;
import java.util.regex.Pattern;

import static com.binomialcapital.binomialhash.schema.ColumnType.*;

/**
 * Explicit schema typing policy and decision helpers.
 */
public final class TypingModel {

    private TypingModel() {}

    private static final List<String> CANDIDATE_TYPES = List.of(
            NUMERIC.value(), BOOL.value(), DATE.value(), DATETIME.value(),
            DICT.value(), LIST.value(), STRING.value(), MIXED.value());

    public static Map<String, Double> scoreCandidates(SchemaFeatureProfile profile) {
        if (profile.nonNullCount() <= 0) return Map.of(NULL.value(), 1.0);
        Map<String, Integer> counts = profile.valueKindCounts();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (String type : CANDIDATE_TYPES) {
            int count = counts.getOrDefault(type, 0);
            if (count > 0) scores.put(type, round6((double) count / profile.nonNullCount()));
        }
        return scores;
    }

    /**
     * Result of base-type classification: type string and confidence score.
     */
    public record BaseTypeResult(String baseType, double confidence) {}

    public static BaseTypeResult classifyBaseType(
            SchemaFeatureProfile profile, Map<String, Double> scores, SchemaTypingPolicy policy) {
        if (profile.nonNullCount() == 0) return new BaseTypeResult(NULL.value(), 1.0);
        List<Map.Entry<String, Double>> ranked = scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .toList();
        String bestType = ranked.get(0).getKey();
        double bestScore = ranked.get(0).getValue();
        double secondScore = ranked.size() > 1 ? ranked.get(1).getValue() : 0.0;
        if (bestScore < policy.mixedMaxBestScore() && secondScore > policy.mixedMinSecondScore()) {
            return new BaseTypeResult(MIXED.value(), round6(bestScore));
        }
        if ((DICT.value().equals(bestType) || LIST.value().equals(bestType))
                && secondScore > policy.structuredMixedMinSecondScore()) {
            return new BaseTypeResult(MIXED.value(), round6(bestScore));
        }
        return new BaseTypeResult(bestType, round6(bestScore));
    }

    public static BaseTypeResult classifyBaseType(SchemaFeatureProfile profile, Map<String, Double> scores) {
        return classifyBaseType(profile, scores, SchemaTypingPolicy.DEFAULT);
    }

    public static List<String> semanticTagsForProfile(
            String baseType, SchemaFeatureProfile profile, Pattern identifierRegex, SchemaTypingPolicy policy) {
        List<String> tags = new ArrayList<>();
        int uniqueCount = profile.uniqueCount();
        double uniqueRatio = profile.uniqueRatio();
        double avgLength = profile.avgLength();
        int nonNull = profile.nonNullCount();
        Map<String, Integer> counters = profile.valueKindCounts();
        List<String> normalized = profile.normalizedStrings();

        int categoricalLimit = Math.min(policy.categoricalMaxUniqueCap(),
                Math.max(policy.categoricalMinUniqueFloor(), (int) (nonNull * policy.categoricalMaxUniqueRatio())));
        if (Set.of(STRING.value(), DATE.value(), DATETIME.value(), BOOL.value()).contains(baseType)
                && uniqueCount <= categoricalLimit) {
            tags.add("categorical");
        }
        if (STRING.value().equals(baseType) && uniqueRatio >= policy.identifierMinUniqueRatio()
                && avgLength <= policy.identifierMaxAvgLength()) {
            int limit = Math.min(normalized.size(), policy.identifierValidationLimit());
            boolean allMatch = true;
            for (int i = 0; i < limit; i++) {
                String v = normalized.get(i);
                if (!identifierRegex.matcher(v).matches() || v.contains(" ")) { allMatch = false; break; }
            }
            if (!normalized.isEmpty() && allMatch) tags.add("identifier");
        }
        if (STRING.value().equals(baseType) && avgLength >= policy.freeTextMinAvgLength()) tags.add("free_text");
        if (ratio(counters, "currency_like", nonNull) >= policy.semanticMajorityRatio()) tags.add("currency");
        if (ratio(counters, "percent_like", nonNull) >= policy.semanticMajorityRatio()) tags.add("percent");
        if (ratio(counters, "json_dict_string", nonNull) >= policy.semanticMajorityRatio()) tags.add("json_stringified_dict");
        if (ratio(counters, "json_list_string", nonNull) >= policy.semanticMajorityRatio()) tags.add("json_stringified_list");
        if (LIST.value().equals(baseType)) {
            if (ratio(counters, "list_of_dicts", nonNull) >= policy.semanticMajorityRatio()) tags.add("list_of_dicts");
            if (ratio(counters, "list_of_scalars", nonNull) >= policy.semanticMajorityRatio()) tags.add("list_of_scalars");
        }
        if (DICT.value().equals(baseType) && uniqueRatio >= policy.recordLikeMinUniqueRatio()) tags.add("record_like");
        return tags;
    }

    public static List<String> semanticTagsForProfile(
            String baseType, SchemaFeatureProfile profile, Pattern identifierRegex) {
        return semanticTagsForProfile(baseType, profile, identifierRegex, SchemaTypingPolicy.DEFAULT);
    }

    public static SchemaDecision decisionFromProfile(
            SchemaFeatureProfile profile, Pattern identifierRegex, SchemaTypingPolicy policy) {
        Map<String, Double> scores = scoreCandidates(profile);
        BaseTypeResult btr = classifyBaseType(profile, scores, policy);
        List<String> tags = semanticTagsForProfile(btr.baseType(), profile, identifierRegex, policy);
        return new SchemaDecision(btr.baseType(), btr.confidence(), scores, tags);
    }

    public static SchemaDecision decisionFromProfile(SchemaFeatureProfile profile, Pattern identifierRegex) {
        return decisionFromProfile(profile, identifierRegex, SchemaTypingPolicy.DEFAULT);
    }

    private static double ratio(Map<String, Integer> counters, String key, int total) {
        return counters.getOrDefault(key, 0) / (double) Math.max(total, 1);
    }

    private static double round6(double v) {
        return Math.round(v * 1e6) / 1e6;
    }
}
