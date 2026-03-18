package com.binomialcapital.binomialhash;

import java.util.List;
import java.util.Map;

/**
 * Structural topology of the raw JSON before flattening.
 */
public record NestingProfile(
    int maxDepth,
    int totalNodes,
    int totalLeaves,
    Map<Integer, Double> branchingByDepth,
    Map<Integer, List<Integer>> arrayLengthsByDepth,
    String pathSignature,
    int nestedKeyCount
) {}
