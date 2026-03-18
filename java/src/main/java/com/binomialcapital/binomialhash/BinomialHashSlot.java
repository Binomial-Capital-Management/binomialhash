package com.binomialcapital.binomialhash;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Single dataset stored in the BinomialHash.
 * Mutable {@code accessCount} uses AtomicInteger for thread safety.
 */
public final class BinomialHashSlot {

    private final String key;
    private final String label;
    private final String fingerprint;
    private final List<Map<String, Object>> rows;
    private final List<String> columns;
    private final Map<String, String> colTypes;
    private final Map<String, Map<String, Object>> colStats;
    private final int rowCount;
    private final long byteSize;
    private final NestingProfile nesting;
    private final Object manifold;
    private final long createdAt;
    private final AtomicInteger accessCount;

    public BinomialHashSlot(
            String key, String label, String fingerprint,
            List<Map<String, Object>> rows, List<String> columns,
            Map<String, String> colTypes, Map<String, Map<String, Object>> colStats,
            int rowCount, long byteSize,
            NestingProfile nesting, Object manifold) {
        this.key = key;
        this.label = label;
        this.fingerprint = fingerprint;
        this.rows = rows;
        this.columns = columns;
        this.colTypes = colTypes;
        this.colStats = colStats;
        this.rowCount = rowCount;
        this.byteSize = byteSize;
        this.nesting = nesting;
        this.manifold = manifold;
        this.createdAt = System.nanoTime();
        this.accessCount = new AtomicInteger(0);
    }

    public String key() { return key; }
    public String label() { return label; }
    public String fingerprint() { return fingerprint; }
    public List<Map<String, Object>> rows() { return rows; }
    public List<String> columns() { return columns; }
    public Map<String, String> colTypes() { return colTypes; }
    public Map<String, Map<String, Object>> colStats() { return colStats; }
    public int rowCount() { return rowCount; }
    public long byteSize() { return byteSize; }
    public NestingProfile nesting() { return nesting; }
    public Object manifold() { return manifold; }
    public long createdAt() { return createdAt; }
    public int accessCount() { return accessCount.get(); }
    public int incrementAccess() { return accessCount.incrementAndGet(); }
}
