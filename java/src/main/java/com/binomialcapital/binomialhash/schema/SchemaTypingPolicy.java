package com.binomialcapital.binomialhash.schema;

/**
 * Named policy values for heuristic schema typing.
 */
public record SchemaTypingPolicy(
    double mixedMaxBestScore,
    double mixedMinSecondScore,
    double structuredMixedMinSecondScore,
    double categoricalMaxUniqueRatio,
    int categoricalMinUniqueFloor,
    int categoricalMaxUniqueCap,
    double identifierMinUniqueRatio,
    double identifierMaxAvgLength,
    int identifierValidationLimit,
    double freeTextMinAvgLength,
    double semanticMajorityRatio,
    double recordLikeMinUniqueRatio
) {
    public static final SchemaTypingPolicy DEFAULT = new SchemaTypingPolicy(
            0.7, 0.2, 0.05, 0.2, 12, 50, 0.9, 40.0, 200, 48.0, 0.6, 0.9);

    public SchemaTypingPolicy() {
        this(0.7, 0.2, 0.05, 0.2, 12, 50, 0.9, 40.0, 200, 48.0, 0.6, 0.9);
    }
}
