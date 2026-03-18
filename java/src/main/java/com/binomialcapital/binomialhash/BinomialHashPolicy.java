package com.binomialcapital.binomialhash;

/**
 * Named policy values for core and analysis behaviour.
 * All fields have sensible defaults; use the builder for overrides.
 */
public final class BinomialHashPolicy {

    public static final BinomialHashPolicy DEFAULT = new BinomialHashPolicy();

    public final int keyLabelPrefixLength;
    public final int keysPreviewColumnCount;
    public final int ingestKeyScanRowCount;
    public final int ingestMaxColumnCount;
    public final int summaryPreviewColumnCount;
    public final int summaryPreviewCharLimit;
    public final int errorPreviewColumnCount;
    public final int groupByAggLimit;
    public final int manifoldNonNullPreviewColumnCount;
    public final int manifoldDiagnosticPreviewColumnCount;
    public final int manifoldInsightsDefaultTopK;
    public final int manifoldInsightsDriverLimit;
    public final int manifoldInsightsDriverBins;
    public final int manifoldInsightsTargetBins;
    public final double manifoldInsightsRegimeZThreshold;
    public final int manifoldInsightsBranchContextLimit;
    public final int manifoldInsightsBranchMinRows;
    public final int manifoldInsightsBranchMinValues;
    public final int orbitDefaultResolution;
    public final int multiscaleDefaultResolution;
    public final int exportCsvMaxRows;
    public final int exportExcelMaxRows;
    public final int exportMarkdownMaxRows;
    public final int exportRowsMaxRows;

    public BinomialHashPolicy() {
        this.keyLabelPrefixLength = 20;
        this.keysPreviewColumnCount = 10;
        this.ingestKeyScanRowCount = 100;
        this.ingestMaxColumnCount = 50;
        this.summaryPreviewColumnCount = 12;
        this.summaryPreviewCharLimit = 1200;
        this.errorPreviewColumnCount = 20;
        this.groupByAggLimit = 20;
        this.manifoldNonNullPreviewColumnCount = 50;
        this.manifoldDiagnosticPreviewColumnCount = 20;
        this.manifoldInsightsDefaultTopK = 5;
        this.manifoldInsightsDriverLimit = 20;
        this.manifoldInsightsDriverBins = 10;
        this.manifoldInsightsTargetBins = 6;
        this.manifoldInsightsRegimeZThreshold = 1.5;
        this.manifoldInsightsBranchContextLimit = 8;
        this.manifoldInsightsBranchMinRows = 20;
        this.manifoldInsightsBranchMinValues = 10;
        this.orbitDefaultResolution = 16;
        this.multiscaleDefaultResolution = 16;
        this.exportCsvMaxRows = 50_000;
        this.exportExcelMaxRows = 10_000;
        this.exportMarkdownMaxRows = 200;
        this.exportRowsMaxRows = 50_000;
    }
}
