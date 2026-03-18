package com.binomialcapital.binomialhash.extract;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.*;

/**
 * Flatten nested dicts and parse embedded JSON-ish strings.
 */
public final class RowFlattener {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private RowFlattener() {}

    @SuppressWarnings("unchecked")
    public static Map<String, Object> flattenRow(Map<String, Object> row, String prefix) {
        Map<String, Object> flat = new LinkedHashMap<>();
        for (var entry : row.entrySet()) {
            String fullKey = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            Object value = entry.getValue();
            if (value instanceof Map<?, ?> map) {
                flat.putAll(flattenRow((Map<String, Object>) map, fullKey));
            } else {
                flat.put(fullKey, value);
            }
        }
        return flat;
    }

    public static Map<String, Object> flattenRow(Map<String, Object> row) {
        return flattenRow(row, "");
    }

    @SuppressWarnings("unchecked")
    public static Object parseEmbeddedJsonish(Object value) {
        if (!(value instanceof String s)) return value;
        String stripped = s.strip();
        if (stripped.isEmpty()) return value;
        if (!((stripped.startsWith("{") && stripped.endsWith("}"))
                || (stripped.startsWith("[") && stripped.endsWith("]")))) {
            return value;
        }
        try {
            return MAPPER.readValue(stripped, Object.class);
        } catch (JsonProcessingException e) {
            return value;
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> normalizeRow(Map<String, Object> row) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (var entry : row.entrySet()) {
            Object parsed = parseEmbeddedJsonish(entry.getValue());
            if (parsed instanceof Map<?, ?> map) {
                out.putAll(flattenRow((Map<String, Object>) map, entry.getKey()));
            } else {
                out.put(entry.getKey(), parsed);
            }
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    public static boolean isListOfDicts(Object value) {
        if (!(value instanceof List<?> list) || list.isEmpty()) return false;
        int limit = Math.min(list.size(), 5);
        for (int i = 0; i < limit; i++) {
            if (!(list.get(i) instanceof Map)) return false;
        }
        return true;
    }
}
