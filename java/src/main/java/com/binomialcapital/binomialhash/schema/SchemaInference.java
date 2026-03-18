package com.binomialcapital.binomialhash.schema;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeParseException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.binomialcapital.binomialhash.schema.ColumnType.*;

/**
 * Schema inference and full-scan column statistics.
 */
public final class SchemaInference {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS, true);

    private static final java.util.regex.Pattern STRICT_NUMERIC_RE =
            java.util.regex.Pattern.compile("^[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?$");
    private static final java.util.regex.Pattern DATE_ONLY_RE =
            java.util.regex.Pattern.compile("^\\d{4}-\\d{2}-\\d{2}$");
    private static final java.util.regex.Pattern DATETIME_RE =
            java.util.regex.Pattern.compile("^\\d{4}-\\d{2}-\\d{2}[T\\s]\\d{2}:\\d{2}");
    private static final java.util.regex.Pattern CURRENCY_RE =
            java.util.regex.Pattern.compile("^\\s*[$€£¥]\\s*[+-]?(?:\\d{1,3}(?:,\\d{3})*|\\d+)(?:\\.\\d+)?\\s*$");
    private static final java.util.regex.Pattern PERCENT_RE =
            java.util.regex.Pattern.compile("^\\s*[+-]?(?:\\d+\\.?\\d*|\\.\\d+)\\s*%\\s*$");
    static final java.util.regex.Pattern IDENTIFIER_RE =
            java.util.regex.Pattern.compile("^[A-Za-z0-9._:/-]{2,64}$");

    private static final Set<String> BOOL_STRINGS = Set.of(
            "true", "false", "yes", "no", "0", "1", "y", "n", "t", "f");

    private SchemaInference() {}

    public static Double toFloatStrict(Object value) {
        if (value == null || "".equals(value) || value instanceof Boolean) return null;
        if (value instanceof Number n) return n.doubleValue();
        if (value instanceof String s) {
            String stripped = s.strip();
            if (!stripped.isEmpty() && STRICT_NUMERIC_RE.matcher(stripped).matches()) {
                try { return Double.parseDouble(stripped); } catch (NumberFormatException e) { return null; }
            }
        }
        return null;
    }

    public static boolean tryParseDate(String value) {
        return parseDateTime(value) != null;
    }

    static String parseDateTime(String value) {
        if (value == null) return null;
        String stripped = value.strip();
        if (stripped.isEmpty()) return null;
        String normalized = stripped.replace("Z", "+00:00");
        String attempt = normalized.length() > 32 ? normalized.substring(0, 32) : normalized;
        try {
            if (DATE_ONLY_RE.matcher(stripped).matches()) {
                LocalDate.parse(attempt.substring(0, 10));
                return stripped;
            }
            // Use up to 32 chars like Python's fromisoformat(normalized[:32])
            LocalDateTime.parse(attempt.substring(0, Math.min(attempt.length(), 19)));
            return stripped;
        } catch (DateTimeParseException e) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private static Object parseJsonishString(String value) {
        if (value == null) return null;
        String stripped = value.strip();
        if (stripped.isEmpty()) return null;
        if (!((stripped.startsWith("{") && stripped.endsWith("}")) || (stripped.startsWith("[") && stripped.endsWith("]"))))
            return null;
        try { return MAPPER.readValue(stripped, Object.class); } catch (JsonProcessingException e) { return null; }
    }

    private static double safeEntropy(Map<String, Integer> counter) {
        int total = 0;
        for (int c : counter.values()) total += c;
        if (total <= 0) return 0.0;
        double entropy = 0.0;
        for (int c : counter.values()) {
            double p = (double) c / total;
            entropy -= p * Math.log(p) / Math.log(2);
        }
        return Math.round(entropy * 1e6) / 1e6;
    }

    @SuppressWarnings("unchecked")
    public static SchemaFeatureProfile columnProfile(List<Object> values) {
        List<Object> nonNull = new ArrayList<>();
        for (Object v : values) if (v != null && !"".equals(v)) nonNull.add(v);
        if (nonNull.isEmpty()) {
            return new SchemaFeatureProfile(0, 0, 0.0, 0.0, Map.of());
        }

        Map<String, Integer> counters = new HashMap<>();
        List<String> normalizedStrings = new ArrayList<>();
        Map<String, Integer> stringCounter = new LinkedHashMap<>();
        List<Integer> listLengths = new ArrayList<>();
        Map<String, Integer> dictKeyCounter = new LinkedHashMap<>();

        for (Object value : nonNull) {
            if (value instanceof Boolean) {
                counters.merge(BOOL.value(), 1, Integer::sum);
                continue;
            }
            if (value instanceof Map<?, ?> map) {
                counters.merge(DICT.value(), 1, Integer::sum);
                for (Object key : map.keySet()) dictKeyCounter.merge(String.valueOf(key), 1, Integer::sum);
                continue;
            }
            if (value instanceof List<?> list) {
                counters.merge(LIST.value(), 1, Integer::sum);
                listLengths.add(list.size());
                int sample = Math.min(list.size(), 20);
                boolean allDicts = !list.isEmpty();
                boolean allScalars = !list.isEmpty();
                for (int i = 0; i < sample; i++) {
                    if (!(list.get(i) instanceof Map)) allDicts = false;
                    if (list.get(i) instanceof Map || list.get(i) instanceof List) allScalars = false;
                }
                if (allDicts) counters.merge("list_of_dicts", 1, Integer::sum);
                else if (allScalars) counters.merge("list_of_scalars", 1, Integer::sum);
                continue;
            }
            if (value instanceof Number) {
                counters.merge(NUMERIC.value(), 1, Integer::sum);
                continue;
            }
            if (value instanceof String s) {
                String stripped = s.strip();
                normalizedStrings.add(stripped);
                stringCounter.merge(stripped, 1, Integer::sum);
                if (BOOL_STRINGS.contains(stripped.toLowerCase())) {
                    counters.merge(BOOL.value(), 1, Integer::sum);
                    continue;
                }
                Object parsedJson = parseJsonishString(stripped);
                if (parsedJson instanceof Map<?, ?> m) {
                    counters.merge("json_dict_string", 1, Integer::sum);
                    counters.merge(DICT.value(), 1, Integer::sum);
                    for (Object key : m.keySet()) dictKeyCounter.merge(String.valueOf(key), 1, Integer::sum);
                    continue;
                }
                if (parsedJson instanceof List<?> l) {
                    counters.merge("json_list_string", 1, Integer::sum);
                    counters.merge(LIST.value(), 1, Integer::sum);
                    listLengths.add(l.size());
                    int sample = Math.min(l.size(), 20);
                    boolean allD = !l.isEmpty(), allS = !l.isEmpty();
                    for (int i = 0; i < sample; i++) {
                        if (!(l.get(i) instanceof Map)) allD = false;
                        if (l.get(i) instanceof Map || l.get(i) instanceof List) allS = false;
                    }
                    if (allD) counters.merge("list_of_dicts", 1, Integer::sum);
                    else if (allS) counters.merge("list_of_scalars", 1, Integer::sum);
                    continue;
                }
                if (toFloatStrict(stripped) != null) {
                    counters.merge(NUMERIC.value(), 1, Integer::sum);
                    continue;
                }
                if (parseDateTime(stripped) != null) {
                    if (DATETIME_RE.matcher(stripped).matches()) counters.merge(DATETIME.value(), 1, Integer::sum);
                    else if (DATE_ONLY_RE.matcher(stripped).matches()) counters.merge(DATE.value(), 1, Integer::sum);
                    else counters.merge(DATETIME.value(), 1, Integer::sum);
                    continue;
                }
                if (CURRENCY_RE.matcher(stripped).matches()) counters.merge("currency_like", 1, Integer::sum);
                if (PERCENT_RE.matcher(stripped).matches()) counters.merge("percent_like", 1, Integer::sum);
                counters.merge(STRING.value(), 1, Integer::sum);
                continue;
            }
            counters.merge(MIXED.value(), 1, Integer::sum);
        }

        // sort_keys=true via ORDER_MAP_ENTRIES_BY_KEYS on MAPPER for consistent hashing
        Set<String> uniqueSet = new HashSet<>();
        for (Object v : nonNull) {
            try { uniqueSet.add(MAPPER.writeValueAsString(v)); } catch (JsonProcessingException e) { uniqueSet.add(String.valueOf(v)); }
        }
        int uniqueCount = uniqueSet.size();
        double avgLength = normalizedStrings.isEmpty() ? 0.0
                : Math.round(normalizedStrings.stream().mapToInt(String::length).average().orElse(0) * 1e6) / 1e6;

        Double avgListLen = listLengths.isEmpty() ? null
                : Math.round(listLengths.stream().mapToInt(Integer::intValue).average().orElse(0) * 1e6) / 1e6;
        Integer maxListLen = listLengths.isEmpty() ? null
                : listLengths.stream().mapToInt(Integer::intValue).max().orElse(0);

        List<String> topDictKeys = dictKeyCounter.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(10).map(Map.Entry::getKey).collect(Collectors.toList());

        return new SchemaFeatureProfile(nonNull.size(), uniqueCount, avgLength,
                stringCounter.isEmpty() ? 0.0 : safeEntropy(stringCounter),
                counters, normalizedStrings, avgListLen, maxListLen, topDictKeys);
    }

    public static Map<String, Object> computeColStats(
            List<Map<String, Object>> rows, String column, String colType,
            Function<Object, Double> toFloat,
            SchemaFeatureProfile profile, SchemaDecision decision) {
        Map<String, Object> stats = new LinkedHashMap<>();
        List<Object> values = new ArrayList<>();
        for (var row : rows) values.add(row.get(column));
        long nonNullCount = values.stream().filter(v -> v != null && !"".equals(v)).count();
        stats.put("nulls", (int) (values.size() - nonNullCount));
        stats.put("scanned", values.size());
        stats.putAll(profile.toDict());
        stats.putAll(decision.toDict());

        if (NUMERIC.value().equals(colType)) {
            List<Double> nums = new ArrayList<>();
            for (Object v : values) {
                if (v == null || "".equals(v)) continue;
                Double d = toFloat.apply(v);
                if (d != null) nums.add(d);
            }
            if (!nums.isEmpty()) {
                Collections.sort(nums);
                stats.put("min", nums.get(0));
                stats.put("max", nums.get(nums.size() - 1));
                double mean = nums.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                stats.put("mean", Math.round(mean * 1e6) / 1e6);
                int mid = nums.size() / 2;
                double median = nums.size() % 2 != 0 ? nums.get(mid)
                        : Math.round((nums.get(mid - 1) + nums.get(mid)) / 2.0 * 1e6) / 1e6;
                stats.put("median", median);
                if (nums.size() > 1) {
                    double variance = nums.stream().mapToDouble(x -> (x - mean) * (x - mean)).sum() / nums.size();
                    stats.put("std", Math.round(Math.sqrt(variance) * 1e6) / 1e6);
                }
            }
        } else if (Set.of(STRING.value(), BOOL.value(), DATE.value(), DATETIME.value()).contains(colType)) {
            Map<String, Integer> freq = new LinkedHashMap<>();
            for (Object v : values) {
                if (v == null || "".equals(v)) continue;
                freq.merge(String.valueOf(v), 1, Integer::sum);
            }
            stats.put("top_values", freq.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(5).map(Map.Entry::getKey).collect(Collectors.toList()));
            if (Set.of(DATE.value(), DATETIME.value()).contains(colType)) {
                List<String> sorted = values.stream()
                        .filter(v -> v != null && !"".equals(v))
                        .map(String::valueOf).sorted().collect(Collectors.toList());
                if (!sorted.isEmpty()) {
                    stats.put("min_date", sorted.get(0));
                    stats.put("max_date", sorted.get(sorted.size() - 1));
                    if (DATETIME.value().equals(colType)) {
                        stats.put("min_datetime", sorted.get(0));
                        stats.put("max_datetime", sorted.get(sorted.size() - 1));
                    }
                }
            }
        } else if (LIST.value().equals(colType)) {
            String kind = decision.semanticTags().contains("list_of_dicts") ? "list_of_dicts"
                    : decision.semanticTags().contains("list_of_scalars") ? "list_of_scalars" : "list";
            stats.put("list_kind", kind);
        }
        return stats;
    }

    /**
     * Infer schema for all columns.
     *
     * @param rows    the data rows
     * @param columns ordered column names
     * @param toFloat caller-supplied numeric coercion function
     */
    public static InferenceResult inferSchema(
            List<Map<String, Object>> rows, List<String> columns, Function<Object, Double> toFloat) {
        Map<String, String> colTypes = new LinkedHashMap<>();
        Map<String, Map<String, Object>> colStats = new LinkedHashMap<>();
        for (String column : columns) {
            List<Object> values = new ArrayList<>();
            for (var row : rows) values.add(row.get(column));
            SchemaFeatureProfile profile = columnProfile(values);
            SchemaDecision decision = TypingModel.decisionFromProfile(profile, IDENTIFIER_RE);
            colTypes.put(column, decision.baseType());
            colStats.put(column, computeColStats(rows, column, colTypes.get(column), toFloat, profile, decision));
        }
        return new InferenceResult(colTypes, colStats);
    }

    public record InferenceResult(Map<String, String> colTypes, Map<String, Map<String, Object>> colStats) {}
}
