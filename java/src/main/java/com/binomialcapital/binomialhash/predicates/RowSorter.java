package com.binomialcapital.binomialhash.predicates;

import com.binomialcapital.binomialhash.BinomialHashSlot;
import com.binomialcapital.binomialhash.stats.StatsHelpers;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Type-aware sorting and projection of row data.
 */
public final class RowSorter {

    private RowSorter() {}

    public static List<Map<String, Object>> sortRows(
            List<Map<String, Object>> rows, String col, String colType, boolean desc) {
        List<Map<String, Object>> sorted = new ArrayList<>(rows);
        if ("numeric".equals(colType)) {
            sorted.sort((a, b) -> {
                Double va = StatsHelpers.toFloatPermissive(a.get(col));
                Double vb = StatsHelpers.toFloatPermissive(b.get(col));
                if (va == null && vb == null) return 0;
                // Python tuple sort: (is_none, value). When desc, nulls sort first (is_none=True is "largest").
                if (desc) {
                    if (va == null) return -1;
                    if (vb == null) return 1;
                    return Double.compare(vb, va);
                } else {
                    if (va == null) return 1;
                    if (vb == null) return -1;
                    return Double.compare(va, vb);
                }
            });
        } else {
            sorted.sort((a, b) -> {
                String sa = String.valueOf(a.getOrDefault(col, ""));
                String sb = String.valueOf(b.getOrDefault(col, ""));
                return desc ? sb.compareTo(sa) : sa.compareTo(sb);
            });
        }
        return sorted;
    }

    public static List<Map<String, Object>> applySortSliceProject(
            List<Map<String, Object>> rows, BinomialHashSlot slot,
            String sortBy, boolean sortDesc, int limit,
            List<String> columns, int maxRetrieveRows) {
        List<Map<String, Object>> result = rows;
        if (sortBy != null && slot.colTypes().containsKey(sortBy)) {
            result = sortRows(result, sortBy, slot.colTypes().get(sortBy), sortDesc);
        }
        int effectiveLimit = Math.min(limit, maxRetrieveRows);
        result = result.subList(0, Math.min(result.size(), effectiveLimit));
        if (columns != null && !columns.isEmpty()) {
            Set<String> colSet = new HashSet<>(columns);
            result = result.stream().map(row -> {
                Map<String, Object> filtered = new LinkedHashMap<>();
                for (var entry : row.entrySet()) {
                    if (colSet.contains(entry.getKey())) filtered.put(entry.getKey(), entry.getValue());
                }
                return filtered;
            }).collect(Collectors.toList());
        }
        return result;
    }
}
