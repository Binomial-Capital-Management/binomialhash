package com.binomialcapital.binomialhash.extract;

import com.binomialcapital.binomialhash.NestingProfile;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Walk raw parsed JSON and capture structural topology.
 */
public final class NestingAnalyzer {

    private static final int MAX_DEPTH = 12;
    private static final int MAX_LIST_TRAVERSE = 200;

    private NestingAnalyzer() {}

    @SuppressWarnings("unchecked")
    public static NestingProfile analyzeNesting(Object data) {
        int[] nodes = {0};
        int[] leaves = {0};
        Map<Integer, List<Integer>> branching = new HashMap<>();
        Map<Integer, List<Integer>> arrayLengths = new HashMap<>();
        List<String> signatureParts = new ArrayList<>();

        walk(data, 0, nodes, leaves, branching, arrayLengths, signatureParts);

        Map<Integer, Double> avgBranching = new LinkedHashMap<>();
        for (var entry : branching.entrySet()) {
            List<Integer> vals = entry.getValue();
            avgBranching.put(entry.getKey(),
                    Math.round(vals.stream().mapToInt(Integer::intValue).average().orElse(0) * 100.0) / 100.0);
        }

        Map<Integer, List<Integer>> truncated = new LinkedHashMap<>();
        for (var entry : arrayLengths.entrySet()) {
            truncated.put(entry.getKey(),
                    entry.getValue().stream().distinct().sorted().limit(10).collect(Collectors.toList()));
        }

        int nestedKeys = 0;
        for (var entry : branching.entrySet()) {
            if (entry.getKey() >= 1) nestedKeys += entry.getValue().size();
        }

        int maxD = 0;
        for (int d : branching.keySet()) maxD = Math.max(maxD, d);
        for (int d : arrayLengths.keySet()) maxD = Math.max(maxD, d);

        String sig = signatureParts.isEmpty() ? "flat"
                : String.join("\u2192", signatureParts.subList(0, Math.min(8, signatureParts.size())));

        return new NestingProfile(maxD, nodes[0], leaves[0], avgBranching, truncated, sig, nestedKeys);
    }

    @SuppressWarnings("unchecked")
    private static void walk(Object obj, int depth, int[] nodes, int[] leaves,
                             Map<Integer, List<Integer>> branching,
                             Map<Integer, List<Integer>> arrayLengths,
                             List<String> signatureParts) {
        if (depth > MAX_DEPTH) return;
        nodes[0]++;

        if (obj instanceof Map<?, ?> map) {
            int childCount = map.size();
            branching.computeIfAbsent(depth, k -> new ArrayList<>()).add(childCount);
            if (depth < 4) signatureParts.add("d" + depth + ":dict(" + childCount + ")");
            for (Object value : map.values()) {
                walk(value, depth + 1, nodes, leaves, branching, arrayLengths, signatureParts);
            }
        } else if (obj instanceof List<?> list) {
            arrayLengths.computeIfAbsent(depth, k -> new ArrayList<>()).add(list.size());
            if (depth < 4) signatureParts.add("d" + depth + ":list(" + list.size() + ")");
            int limit = Math.min(list.size(), MAX_LIST_TRAVERSE);
            for (int i = 0; i < limit; i++) {
                walk(list.get(i), depth + 1, nodes, leaves, branching, arrayLengths, signatureParts);
            }
        } else {
            leaves[0]++;
        }
    }
}
