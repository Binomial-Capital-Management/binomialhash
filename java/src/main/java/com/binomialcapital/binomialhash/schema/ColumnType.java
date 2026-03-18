package com.binomialcapital.binomialhash.schema;

public enum ColumnType {
    NUMERIC("numeric"),
    STRING("string"),
    DATE("date"),
    DATETIME("datetime"),
    BOOL("bool"),
    DICT("dict"),
    LIST("list"),
    MIXED("mixed"),
    NULL("null");

    private final String value;

    ColumnType(String value) {
        this.value = value;
    }

    public String value() {
        return value;
    }

    public static ColumnType fromString(String s) {
        if (s == null) return NULL;
        for (ColumnType ct : values()) {
            if (ct.value.equals(s)) return ct;
        }
        return STRING;
    }

    @Override
    public String toString() {
        return value;
    }
}
