package com.binomialcapital.binomialhash.predicates;

/**
 * Explicit predicate-builder limits.
 * Defaults are correctness-first and do not silently truncate user queries.
 */
public record QueryBuildPolicy(
    Integer maxDepth,
    Integer maxClausesPerNode
) {
    public static final QueryBuildPolicy DEFAULT = new QueryBuildPolicy(20, null);

    public QueryBuildPolicy() {
        this(20, null);
    }
}
