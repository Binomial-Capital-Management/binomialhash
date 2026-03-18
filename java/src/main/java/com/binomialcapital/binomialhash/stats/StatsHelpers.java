package com.binomialcapital.binomialhash.stats;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Shared helpers: numeric coercion, aggregation, correlation, and linear fits.
 */
public final class StatsHelpers {

    public static final Set<String> NUMERIC_FUNCS =
            Set.of("sum", "mean", "median", "min", "max", "std");

    public static final Set<String> ALL_AGG_FUNCS;
    static {
        var s = new HashSet<>(NUMERIC_FUNCS);
        s.add("count");
        s.add("count_distinct");
        ALL_AGG_FUNCS = Collections.unmodifiableSet(s);
    }

    private StatsHelpers() {}

    // ── Numeric coercion ────────────────────────────────────────────

    public static Double toFloatPermissive(Object v) {
        if (v == null) return null;
        if (v instanceof Boolean) return null;
        if (v instanceof Number n) {
            double d = n.doubleValue();
            return Double.isFinite(d) ? d : null;
        }
        if (!(v instanceof String)) return null;
        String s = (String) v;
        if (s.isEmpty()) return null;
        try {
            double d = Double.parseDouble(s);
            if (Double.isFinite(d)) return d;
        } catch (NumberFormatException ignored) {}
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isDigit(c) || c == '.' || c == '+' || c == '-' || c == 'e' || c == 'E') {
                sb.append(c);
            }
        }
        if (sb.isEmpty()) return null;
        try {
            double d = Double.parseDouble(sb.toString());
            return Double.isFinite(d) ? d : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    // ── Aggregation ─────────────────────────────────────────────────

    public static Double aggNumeric(List<Double> nums, String func) {
        if (nums.isEmpty()) return null;
        return switch (func) {
            case "sum" -> round4(nums.stream().mapToDouble(Double::doubleValue).sum());
            case "mean" -> round4(nums.stream().mapToDouble(Double::doubleValue).average().orElse(0));
            case "min" -> nums.stream().mapToDouble(Double::doubleValue).min().orElse(0);
            case "max" -> nums.stream().mapToDouble(Double::doubleValue).max().orElse(0);
            case "median" -> {
                List<Double> sorted = nums.stream().sorted().collect(Collectors.toList());
                int mid = sorted.size() / 2;
                yield sorted.size() % 2 != 0
                        ? sorted.get(mid)
                        : round4((sorted.get(mid - 1) + sorted.get(mid)) / 2.0);
            }
            case "std" -> {
                double mean = nums.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double variance = nums.stream().mapToDouble(x -> (x - mean) * (x - mean)).sum() / nums.size();
                yield round4(Math.sqrt(variance));
            }
            default -> null;
        };
    }

    public static Object runAgg(List<Map<String, Object>> rows, String col, String func) {
        if (NUMERIC_FUNCS.contains(func)) {
            List<Double> nums = new ArrayList<>();
            for (var row : rows) {
                Double val = toFloatPermissive(row.get(col));
                if (val != null) nums.add(val);
            }
            return nums.isEmpty() ? null : aggNumeric(nums, func);
        }
        if ("count".equals(func)) {
            return (int) rows.stream().filter(r -> r.get(col) != null && !"".equals(r.get(col))).count();
        }
        if ("count_distinct".equals(func)) {
            return (int) rows.stream()
                    .filter(r -> r.get(col) != null)
                    .map(r -> String.valueOf(r.get(col)))
                    .distinct().count();
        }
        return null;
    }

    // ── Numeric extraction ──────────────────────────────────────────

    public static List<double[]> numericColumnValues(List<Map<String, Object>> rows, String column) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) {
            Double val = toFloatPermissive(rows.get(i).get(column));
            if (val != null) out.add(new double[]{i, val});
        }
        return out;
    }

    public static double[][] extractNumericPairs(List<Map<String, Object>> rows, String colA, String colB) {
        List<double[]> pairs = new ArrayList<>();
        for (var row : rows) {
            Double a = toFloatPermissive(row.get(colA));
            Double b = toFloatPermissive(row.get(colB));
            if (a != null && b != null) pairs.add(new double[]{a, b});
        }
        double[] xs = new double[pairs.size()];
        double[] ys = new double[pairs.size()];
        for (int i = 0; i < pairs.size(); i++) {
            xs[i] = pairs.get(i)[0];
            ys[i] = pairs.get(i)[1];
        }
        return new double[][]{xs, ys};
    }

    public static List<double[]> extractNumericMatrix(List<Map<String, Object>> rows, List<String> fields) {
        List<double[]> out = new ArrayList<>();
        for (var row : rows) {
            double[] vals = new double[fields.size()];
            boolean ok = true;
            for (int j = 0; j < fields.size(); j++) {
                Double v = toFloatPermissive(row.get(fields.get(j)));
                if (v == null) { ok = false; break; }
                vals[j] = v;
            }
            if (ok) out.add(vals);
        }
        return out;
    }

    // ── Correlation ─────────────────────────────────────────────────

    public static double pearsonCorr(double[] xs, double[] ys) {
        if (xs.length != ys.length || xs.length < 3) return 0.0;
        int n = xs.length;
        double mx = mean(xs), my = mean(ys);
        double num = 0, dx = 0, dy = 0;
        for (int i = 0; i < n; i++) {
            double xi = xs[i] - mx, yi = ys[i] - my;
            num += xi * yi;
            dx += xi * xi;
            dy += yi * yi;
        }
        double den = Math.sqrt(dx) * Math.sqrt(dy);
        return den == 0 ? 0.0 : Math.max(-1.0, Math.min(1.0, num / den));
    }

    // ── Linear fit ──────────────────────────────────────────────────

    /**
     * Fit y = slope*x + intercept. Returns {slope, intercept, r2}.
     */
    public static double[] fitLinear(double[] xs, double[] ys) {
        if (xs.length != ys.length || xs.length < 3) return new double[]{0, 0, 0};
        double mx = mean(xs), my = mean(ys);
        double varX = 0, cov = 0;
        for (int i = 0; i < xs.length; i++) {
            double dx = xs[i] - mx;
            varX += dx * dx;
            cov += dx * (ys[i] - my);
        }
        if (varX == 0) return new double[]{0, my, 0};
        double slope = cov / varX;
        double intercept = my - slope * mx;
        double corr = pearsonCorr(xs, ys);
        return new double[]{slope, intercept, corr * corr};
    }

    // ── Quantiles ───────────────────────────────────────────────────

    public static double[] quantileEdges(double[] values, int bins) {
        if (values.length == 0) return new double[0];
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        double[] edges = new double[bins + 1];
        for (int b = 0; b <= bins; b++) {
            int pos = (int) Math.round((n - 1) * ((double) b / bins));
            edges[b] = sorted[pos];
        }
        for (int i = 1; i < edges.length; i++) {
            if (edges[i] < edges[i - 1]) edges[i] = edges[i - 1];
        }
        return edges;
    }

    public static int bucketIndex(double val, double[] edges) {
        if (edges.length < 2) return 0;
        for (int i = 0; i < edges.length - 1; i++) {
            if (val <= edges[i + 1]) return i;
        }
        return edges.length - 2;
    }

    // ── Other helpers ───────────────────────────────────────────────

    public static double normalCdf(double z) {
        return 0.5 * (1.0 + org.apache.commons.math3.special.Erf.erf(z / Math.sqrt(2.0)));
    }

    public static double[] spearmanRank(double[] vals) {
        int n = vals.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> vals[i]));
        double[] ranks = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n - 1 && vals[idx[j + 1]] == vals[idx[j]]) j++;
            double avgRank = (i + j) / 2.0 + 1.0;
            for (int k = i; k <= j; k++) ranks[idx[k]] = avgRank;
            i = j + 1;
        }
        return ranks;
    }

    public static double shannonEntropy(int[] counts) {
        int total = 0;
        for (int c : counts) total += c;
        if (total == 0) return 0.0;
        double h = 0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / total;
                h -= p * Math.log(p);
            }
        }
        return h;
    }

    /**
     * OLS R-squared via normal equations with Gaussian elimination (partial pivoting).
     */
    public static double olsR2(List<double[]> xsAll, double[] ys) {
        int n = ys.length;
        int p = xsAll.isEmpty() ? 0 : xsAll.get(0).length;
        if (n < p + 2) return 0.0;

        double[][] xtx = new double[p + 1][p + 1];
        double[] xty = new double[p + 1];
        for (int i = 0; i < n; i++) {
            double[] rowExt = new double[p + 1];
            rowExt[0] = 1.0;
            System.arraycopy(xsAll.get(i), 0, rowExt, 1, p);
            for (int a = 0; a <= p; a++) {
                xty[a] += rowExt[a] * ys[i];
                for (int b = 0; b <= p; b++) {
                    xtx[a][b] += rowExt[a] * rowExt[b];
                }
            }
        }

        double[][] aug = new double[p + 1][p + 2];
        for (int r = 0; r <= p; r++) {
            System.arraycopy(xtx[r], 0, aug[r], 0, p + 1);
            aug[r][p + 1] = xty[r];
        }

        for (int col = 0; col <= p; col++) {
            int maxRow = col;
            for (int ri = col + 1; ri <= p; ri++) {
                if (Math.abs(aug[ri][col]) > Math.abs(aug[maxRow][col])) maxRow = ri;
            }
            double[] tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp;
            if (Math.abs(aug[col][col]) < 1e-12) return 0.0;
            for (int ri = col + 1; ri <= p; ri++) {
                double factor = aug[ri][col] / aug[col][col];
                for (int j = col; j <= p + 1; j++) {
                    aug[ri][j] -= factor * aug[col][j];
                }
            }
        }

        double[] beta = new double[p + 1];
        for (int r = p; r >= 0; r--) {
            beta[r] = aug[r][p + 1];
            for (int j = r + 1; j <= p; j++) beta[r] -= aug[r][j] * beta[j];
            beta[r] /= aug[r][r];
        }

        double meanY = mean(ys);
        double ssRes = 0, ssTot = 0;
        for (int r = 0; r < n; r++) {
            double pred = beta[0];
            for (int j = 0; j < p; j++) pred += beta[j + 1] * xsAll.get(r)[j];
            ssRes += (ys[r] - pred) * (ys[r] - pred);
            ssTot += (ys[r] - meanY) * (ys[r] - meanY);
        }
        return ssTot > 1e-12 ? 1.0 - ssRes / ssTot : 0.0;
    }

    // ── Internal ────────────────────────────────────────────────────

    private static double mean(double[] vals) {
        double s = 0;
        for (double v : vals) s += v;
        return s / vals.length;
    }

    private static double round4(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }

    // erf is provided by org.apache.commons.math3.special.Erf — no approximation needed
}
