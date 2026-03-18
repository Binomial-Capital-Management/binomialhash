package com.binomialcapital.binomialhash.predicates;

import com.binomialcapital.binomialhash.stats.StatsHelpers;

import java.util.*;
import java.util.function.Predicate;

/**
 * Build AND/OR predicate trees from JSON-style where clauses.
 */
public final class PredicateBuilder {

    private static final Set<String> KNOWN_OPS = Set.of("=", "!=", ">", "<", ">=", "<=", "contains", "in", "not_in");

    private PredicateBuilder() {}

    @SuppressWarnings("unchecked")
    public static Predicate<Map<String, Object>> buildLeafPredicate(
            String col, String op, Object val, String colType) {
        if (!KNOWN_OPS.contains(op)) return null;

        if ("in".equals(op)) {
            Set<Object> values = val instanceof List ? new HashSet<>((List<?>) val) : Set.of(val);
            return row -> values.contains(row.get(col));
        }
        if ("not_in".equals(op)) {
            Set<Object> values = val instanceof List ? new HashSet<>((List<?>) val) : Set.of(val);
            return row -> !values.contains(row.get(col));
        }
        if ("contains".equals(op)) {
            if ("numeric".equals(colType)) {
                Double target = StatsHelpers.toFloatPermissive(val);
                return row -> {
                    Double rv = StatsHelpers.toFloatPermissive(row.get(col));
                    return rv != null && target != null
                            && String.valueOf(rv).contains(String.valueOf(target));
                };
            }
            return row -> {
                Object a = row.get(col);
                return a != null && val != null
                        && String.valueOf(a).toLowerCase().contains(String.valueOf(val).toLowerCase());
            };
        }
        if ("numeric".equals(colType)) {
            Double target = StatsHelpers.toFloatPermissive(val);
            return row -> {
                Double rv = StatsHelpers.toFloatPermissive(row.get(col));
                return cmpNumeric(rv, target, op);
            };
        }
        return row -> cmpGeneric(row.get(col), val, op);
    }

    @SuppressWarnings("unchecked")
    public static Predicate<Map<String, Object>> buildPredicate(
            Map<String, Object> where, Map<String, String> colTypes,
            int depth, QueryBuildPolicy policy) {
        if (policy.maxDepth() != null && depth > policy.maxDepth()) return null;

        for (String logicOp : List.of("and", "or")) {
            if (where.containsKey(logicOp)) {
                Object subsObj = where.get(logicOp);
                if (!(subsObj instanceof List<?> subs)) return null;
                List<?> effectiveSubs = subs;
                if (policy.maxClausesPerNode() != null) {
                    effectiveSubs = subs.subList(0, Math.min(subs.size(), policy.maxClausesPerNode()));
                }
                List<Predicate<Map<String, Object>>> predicates = new ArrayList<>();
                for (Object sub : effectiveSubs) {
                    if (!(sub instanceof Map)) return null;
                    Predicate<Map<String, Object>> p = buildPredicate(
                            (Map<String, Object>) sub, colTypes, depth + 1, policy);
                    if (p == null) return null;
                    predicates.add(p);
                }
                if ("and".equals(logicOp)) {
                    return row -> {
                        for (var p : predicates) if (!p.test(row)) return false;
                        return true;
                    };
                } else {
                    return row -> {
                        for (var p : predicates) if (p.test(row)) return true;
                        return false;
                    };
                }
            }
        }

        String col = Objects.toString(where.getOrDefault("column", ""), "");
        String op = Objects.toString(where.getOrDefault("op", "="), "=");
        Object val = where.get("value");
        String colType = colTypes.getOrDefault(col, "string");
        return buildLeafPredicate(col, op, val, colType);
    }

    public static Predicate<Map<String, Object>> buildPredicate(
            Map<String, Object> where, Map<String, String> colTypes) {
        return buildPredicate(where, colTypes, 0, QueryBuildPolicy.DEFAULT);
    }

    public static List<Map<String, Object>> filterRowsByCondition(
            List<Map<String, Object>> rows, String column, String op, Object value) {
        List<Map<String, Object>> result = new ArrayList<>();
        for (var row : rows) {
            Object rv = row.get(column);
            if (rv == null) continue;
            Double fv, tv;
            try {
                fv = Double.parseDouble(String.valueOf(rv));
                tv = Double.parseDouble(String.valueOf(value));
            } catch (NumberFormatException e) {
                if ("=".equals(op) && String.valueOf(rv).equals(String.valueOf(value))) result.add(row);
                else if ("!=".equals(op) && !String.valueOf(rv).equals(String.valueOf(value))) result.add(row);
                continue;
            }
            boolean match = switch (op) {
                case ">" -> fv > tv;
                case ">=" -> fv >= tv;
                case "<" -> fv < tv;
                case "<=" -> fv <= tv;
                case "=" -> Math.abs(fv - tv) < 1e-9;
                case "!=" -> Math.abs(fv - tv) >= 1e-9;
                default -> false;
            };
            if (match) result.add(row);
        }
        return result;
    }

    private static boolean cmpNumeric(Double a, Double b, String op) {
        if (a == null || b == null) {
            return switch (op) {
                case "=" -> Objects.equals(a, b);
                case "!=" -> !Objects.equals(a, b);
                default -> false;
            };
        }
        return switch (op) {
            case "=" -> Objects.equals(a, b);
            case "!=" -> !Objects.equals(a, b);
            case ">" -> a > b;
            case "<" -> a < b;
            case ">=" -> a >= b;
            case "<=" -> a <= b;
            default -> false;
        };
    }

    private static boolean cmpGeneric(Object a, Object b, String op) {
        return switch (op) {
            case "=" -> Objects.equals(a, b);
            case "!=" -> !Objects.equals(a, b);
            case ">" -> a != null && b != null && String.valueOf(a).compareTo(String.valueOf(b)) > 0;
            case "<" -> a != null && b != null && String.valueOf(a).compareTo(String.valueOf(b)) < 0;
            case ">=" -> a != null && b != null && String.valueOf(a).compareTo(String.valueOf(b)) >= 0;
            case "<=" -> a != null && b != null && String.valueOf(a).compareTo(String.valueOf(b)) <= 0;
            default -> false;
        };
    }
}
