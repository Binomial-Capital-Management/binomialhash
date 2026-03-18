package com.binomialcapital.binomialhash.schema;

import java.util.*;

/**
 * Deterministic, auditable feature summary for one column.
 */
public final class SchemaFeatureProfile {

    private final int nonNullCount;
    private final int uniqueCount;
    private final double avgLength;
    private final double stringEntropy;
    private final Map<String, Integer> valueKindCounts;
    private final List<String> normalizedStrings;
    private final Double avgListLength;
    private final Integer maxListLength;
    private final List<String> topDictKeys;

    public SchemaFeatureProfile(
            int nonNullCount, int uniqueCount, double avgLength, double stringEntropy,
            Map<String, Integer> valueKindCounts, List<String> normalizedStrings,
            Double avgListLength, Integer maxListLength, List<String> topDictKeys) {
        this.nonNullCount = nonNullCount;
        this.uniqueCount = uniqueCount;
        this.avgLength = avgLength;
        this.stringEntropy = stringEntropy;
        this.valueKindCounts = valueKindCounts != null ? valueKindCounts : Map.of();
        this.normalizedStrings = normalizedStrings != null ? normalizedStrings : List.of();
        this.avgListLength = avgListLength;
        this.maxListLength = maxListLength;
        this.topDictKeys = topDictKeys != null ? topDictKeys : List.of();
    }

    public SchemaFeatureProfile(int nonNullCount, int uniqueCount, double avgLength,
                                double stringEntropy, Map<String, Integer> valueKindCounts) {
        this(nonNullCount, uniqueCount, avgLength, stringEntropy, valueKindCounts, List.of(), null, null, List.of());
    }

    public int nonNullCount() { return nonNullCount; }
    public int uniqueCount() { return uniqueCount; }
    public double avgLength() { return avgLength; }
    public double stringEntropy() { return stringEntropy; }
    public Map<String, Integer> valueKindCounts() { return valueKindCounts; }
    public List<String> normalizedStrings() { return normalizedStrings; }
    public Double avgListLength() { return avgListLength; }
    public Integer maxListLength() { return maxListLength; }
    public List<String> topDictKeys() { return topDictKeys; }

    public double uniqueRatio() {
        return (double) uniqueCount / Math.max(nonNullCount, 1);
    }

    public Map<String, Object> toDict() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("non_null_count", nonNullCount);
        out.put("unique_count", uniqueCount);
        out.put("unique_ratio", Math.round(uniqueRatio() * 1e6) / 1e6);
        out.put("avg_length", avgLength);
        out.put("string_entropy", stringEntropy);
        out.put("value_kind_counts", valueKindCounts);
        if (avgListLength != null) out.put("avg_list_length", avgListLength);
        if (maxListLength != null) out.put("max_list_length", maxListLength);
        if (!topDictKeys.isEmpty()) out.put("top_dict_keys", topDictKeys);
        return out;
    }
}
