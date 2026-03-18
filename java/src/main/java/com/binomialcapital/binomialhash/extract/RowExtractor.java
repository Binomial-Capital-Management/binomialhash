package com.binomialcapital.binomialhash.extract;

import java.util.*;
import java.util.logging.Logger;

import static com.binomialcapital.binomialhash.extract.RowFlattener.*;

/**
 * Find the main list-of-dicts in parsed JSON, normalize, flatten, and optionally explode.
 */
public final class RowExtractor {

    private static final Logger LOG = Logger.getLogger(RowExtractor.class.getName());

    private RowExtractor() {}

    /**
     * Result of {@link #extractRows}: the rows and any scalar metadata from the parent dict.
     */
    public record ExtractionResult(List<Map<String, Object>> rows, Map<String, Object> meta) {}

    @SuppressWarnings("unchecked")
    public static ExtractionResult extractRows(Object data) {
        Map<String, Object> meta = new LinkedHashMap<>();

        if (data instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof Map) {
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) rows.add(normalizeRow((Map<String, Object>) item));
            rows = explodeEmbeddedTable(rows);
            if (!rows.isEmpty() && hasNestedDict(rows.get(0))) {
                rows = flattenAll(rows);
            }
            return new ExtractionResult(rows, meta);
        }

        if (data instanceof Map<?, ?> map) {
            String[] bestKey = {""};
            List<Map<String, Object>>[] bestList = new List[]{List.of()};
            findLargestList((Map<String, Object>) map, 0, bestKey, bestList);

            if (bestList[0].size() >= 2) {
                for (var entry : ((Map<String, Object>) map).entrySet()) {
                    if (!(entry.getValue() instanceof List) && !(entry.getValue() instanceof Map)) {
                        meta.put(entry.getKey(), entry.getValue());
                    }
                }
                List<Map<String, Object>> rows = new ArrayList<>();
                for (var item : bestList[0]) rows.add(normalizeRow(item));
                rows = explodeEmbeddedTable(rows);
                if (!rows.isEmpty() && hasNestedDict(rows.get(0))) {
                    rows = flattenAll(rows);
                }
                LOG.info(String.format("[BH] extract_rows found %d rows at path '%s' (%d cols)",
                        rows.size(), bestKey[0], rows.isEmpty() ? 0 : rows.get(0).size()));
                return new ExtractionResult(rows, meta);
            }
        }

        return new ExtractionResult(List.of(), meta);
    }

    @SuppressWarnings("unchecked")
    private static void findLargestList(Map<String, Object> data, int depth,
                                        String[] bestKey, List<Map<String, Object>>[] bestList) {
        for (var entry : data.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof Map) {
                if (list.size() > bestList[0].size()) {
                    bestKey[0] = entry.getKey();
                    List<Map<String, Object>> typed = new ArrayList<>();
                    for (Object item : list) typed.add((Map<String, Object>) item);
                    bestList[0] = typed;
                }
            } else if (value instanceof Map && depth < 2) {
                String[] childKey = {""};
                List<Map<String, Object>>[] childList = new List[]{List.of()};
                findLargestList((Map<String, Object>) value, depth + 1, childKey, childList);
                if (childList[0].size() > bestList[0].size()) {
                    bestKey[0] = entry.getKey() + "." + childKey[0];
                    bestList[0] = childList[0];
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> explodeEmbeddedTable(List<Map<String, Object>> rows) {
        if (rows.isEmpty()) return rows;
        Map<String, double[]> candidateStats = new LinkedHashMap<>();
        int sampleSize = Math.min(rows.size(), 50);
        for (int i = 0; i < sampleSize; i++) {
            var row = rows.get(i);
            for (var entry : row.entrySet()) {
                Object value = parseEmbeddedJsonish(entry.getValue());
                if (isListOfDicts(value)) {
                    double[] stat = candidateStats.computeIfAbsent(entry.getKey(), k -> new double[2]);
                    stat[0]++;
                    stat[1] += ((List<?>) value).size();
                }
            }
        }
        if (candidateStats.isEmpty()) return rows;

        String bestCol = null;
        double bestRows = -1;
        double bestItems = -1;
        for (var entry : candidateStats.entrySet()) {
            double r = entry.getValue()[0];
            double it = entry.getValue()[1];
            if (r > bestRows || (r == bestRows && it > bestItems)) {
                bestRows = r;
                bestItems = it;
                bestCol = entry.getKey();
            }
        }
        double[] best = candidateStats.get(bestCol);
        if (best[0] < Math.max(2, sampleSize * 0.3)) return rows;

        List<Map<String, Object>> exploded = new ArrayList<>();
        for (var row : rows) {
            Object value = parseEmbeddedJsonish(row.get(bestCol));
            if (!isListOfDicts(value)) continue;
            Map<String, Object> base = new LinkedHashMap<>();
            for (var entry : row.entrySet()) {
                if (entry.getKey().equals(bestCol)) continue;
                Object parsed = parseEmbeddedJsonish(entry.getValue());
                if (parsed instanceof Map<?, ?> map) {
                    base.putAll(flattenRow((Map<String, Object>) map, entry.getKey()));
                } else {
                    base.put(entry.getKey(), parsed);
                }
            }
            for (Object item : (List<?>) value) {
                Map<String, Object> flatItem = flattenRow((Map<String, Object>) item, bestCol);
                Map<String, Object> merged = new LinkedHashMap<>(base);
                merged.putAll(flatItem);
                exploded.add(merged);
            }
        }
        return exploded.isEmpty() ? rows : exploded;
    }

    private static boolean hasNestedDict(Map<String, Object> row) {
        for (Object v : row.values()) {
            if (v instanceof Map) return true;
        }
        return false;
    }

    private static List<Map<String, Object>> flattenAll(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (var row : rows) out.add(flattenRow(row));
        return out;
    }
}
