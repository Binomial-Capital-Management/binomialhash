package com.binomialcapital.binomialhash.stats;

/**
 * Named statistical policy values for BH analysis routines.
 */
public final class StatsPolicy {

    public static final StatsPolicy DEFAULT = new StatsPolicy();

    public final int errorPreviewColumnLimit = 20;
    public final int regressionMinExtraSamples = 2;
    public final int partialCorrMinExtraSamples = 3;
    public final int pcaDefaultComponentCount = 3;
    public final int pcaDefaultFieldCount = 10;
    public final int pcaMinFieldCount = 2;
    public final int pcaMinCompleteRows = 5;
    public final int pcaTopLoadingCount = 5;
    public final int dependencyDefaultTopK = 8;
    public final int dependencyMinExtraRows = 3;
    public final double dependencyScorePartialWeight = 0.45;
    public final double dependencyScoreRawWeight = 0.35;
    public final double dependencyScoreCoefWeight = 0.20;
    public final double dependencyScoreCoefClip = 1.0;
    public final int dependencyRankCap = 50;
    public final int solverDefaultTopK = 5;
    public final int solverSolutionCap = 50;

    public final int distributionDefaultBins = 15;
    public final int distributionMinValues = 5;
    public final double outlierDefaultZscore = 3.0;
    public final double outlierDefaultIqrMultiplier = 1.5;
    public final int outlierMaxFlagged = 50;
    public final int benfordMinValues = 50;
    public final double vifHighThreshold = 10.0;
    public final int effectiveDimDefaultK = 5;
    public final int effectiveDimMinFields = 3;
    public final int effectiveDimMinRows = 10;

    public final int rankCorrMinSamples = 5;
    public final int chiSquaredDefaultBins = 10;
    public final double chiSquaredMinExpected = 1.0;
    public final int anovaMinGroupSize = 2;
    public final int anovaMinGroups = 2;
    public final int miDefaultBins = 10;
    public final int miMinSamples = 20;
    public final int hsicDefaultPermutations = 100;
    public final int hsicMinSamples = 10;
    public final double copulaDefaultTail = 0.05;
    public final int copulaMinSamples = 30;

    public final int polynomialMinSamples = 10;
    public final int interactionDefaultTopK = 5;
    public final int interactionMinSamples = 15;
    public final int sparseDefaultMaxFeatures = 10;
    public final int sparseCvFolds = 5;
    public final int sparseNAlphas = 20;
    public final int sparseMaxIter = 1000;
    public final int importanceNShuffles = 10;
    public final int ibDefaultClusters = 5;
    public final double ibDefaultBeta = 1.0;
    public final int ibMaxIter = 100;

    public final int clusterMaxK = 8;
    public final int clusterMaxIter = 100;
    public final int clusterNInit = 5;
    public final int spectralDefaultNeighbors = 10;
    public final int spectralDefaultComponents = 5;
    public final int icaMaxIter = 200;
    public final double icaTol = 1e-4;
    public final double graphicalDefaultAlpha = 0.01;
    public final int graphicalGlassoMaxIter = 100;
    public final int topologyMaxPoints = 200;
    public final int topologyNThresholds = 50;

    public final double causalDefaultAlpha = 0.05;
    public final int causalMaxConditioning = 3;
    public final int causalMinSamples = 20;
    public final int teDefaultBins = 8;
    public final int teMaxLag = 5;
    public final int teSurrogates = 100;
    public final int doDefaultBins = 5;
    public final int doMinPerStratum = 5;
    public final int synthMinPrePeriods = 5;

    public final int acfMaxLag = 20;
    public final int acfMinSamples = 30;
    public final int changepointMinSegment = 10;
    public final double changepointDefaultThreshold = 2.0;
    public final int rollingDefaultWindow = 20;
    public final int phaseMaxEmbedding = 10;
    public final double phaseFnnThreshold = 15.0;
    public final int recurrenceDefaultEmbed = 3;

    public final int entropyMaxScale = 10;
    public final int entropyDefaultEmbed = 2;
    public final int renormMaxScale = 5;
}
