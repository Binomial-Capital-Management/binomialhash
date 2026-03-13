"""Statistical analysis tool specs for BinomialHash."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import ToolSpec, _prop

if TYPE_CHECKING:
    from ..core import BinomialHash


def _make_stats_specs(bh: "BinomialHash") -> List[ToolSpec]:
    return [
        ToolSpec(
            name="bh_regress",
            description=(
                "Multivariate OLS regression.  Fits target = β₀ + β₁·d₁ + … "
                "Returns coefficients, R², adjusted R², and individual correlations.  "
                "Use to find hidden drivers masked by confounders."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target": _prop("string", "Target column name."),
                    "drivers_json": _prop("string", "JSON list of driver column names."),
                },
                "required": ["key", "target", "drivers_json"],
            },
            handler=lambda key, target, drivers_json: bh.regress(key, target, drivers_json),
            group="stats",
        ),
        ToolSpec(
            name="bh_partial_corr",
            description=(
                "Partial correlation: how correlated are A and B AFTER removing "
                "the effect of control variables?  If raw correlation is high but "
                "partial is low, the relationship is spurious."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "field_a": _prop("string", "First field."),
                    "field_b": _prop("string", "Second field."),
                    "controls_json": _prop("string", "JSON list of control columns to remove."),
                },
                "required": ["key", "field_a", "field_b", "controls_json"],
            },
            handler=lambda key, field_a, field_b, controls_json: (
                bh.partial_correlate(key, field_a, field_b, controls_json)
            ),
            group="stats",
        ),
        ToolSpec(
            name="bh_pca_surface",
            description=(
                "PCA on selected numeric fields.  Exposes latent low-dimensional "
                "structure — eigenvalues, explained variance, and loadings."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "fields_json": _prop("string", "JSON list of numeric field names."),
                    "n_components": _prop("integer", "Number of principal components.", default=3),
                },
                "required": ["key", "fields_json"],
            },
            handler=lambda key, fields_json, n_components=3: (
                bh.pca_surface(key, fields_json, n_components)
            ),
            group="stats",
        ),
        ToolSpec(
            name="bh_dependency_screen",
            description=(
                "Rank candidate drivers by raw correlation, partial correlation, "
                "and regression coefficient strength against a target."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target": _prop("string", "Target column."),
                    "candidates_json": _prop("string", "JSON list of candidate driver columns."),
                    "controls_json": _prop("string", "JSON list of control columns.", default="[]"),
                    "top_k": _prop("integer", "Max results to return.", default=8),
                },
                "required": ["key", "target", "candidates_json"],
            },
            handler=lambda key, target, candidates_json, controls_json="[]", top_k=8: (
                bh.dependency_screen(key, target, candidates_json, controls_json, top_k)
            ),
            group="stats",
        ),
        ToolSpec(
            name="bh_solver",
            description=(
                "Goal-seek over stored rows.  Finds feasible high-quality "
                "solutions for a target field under constraints on controllable "
                "variables."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "target_field": _prop("string", "Field to optimize."),
                    "goal_json": _prop("string", "Goal spec, e.g. '{\"direction\":\"maximize\"}'."),
                    "controllable_vars_json": _prop("string", "JSON list of controllable column names."),
                    "constraints_json": _prop("string", "JSON list of constraints.", default="[]"),
                    "top_k": _prop("integer", "Max solutions to return.", default=5),
                },
                "required": ["key", "target_field", "goal_json", "controllable_vars_json"],
            },
            handler=lambda key, target_field, goal_json, controllable_vars_json, constraints_json="[]", top_k=5: (
                bh.solver(key, target_field, goal_json, controllable_vars_json, constraints_json, top_k)
            ),
            group="stats",
        ),
        ToolSpec(
            name="bh_distribution",
            description="Full distributional profile of a numeric column: count, mean, median, std, skewness, kurtosis, quantiles, histogram, normality p-value, and shape label.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column to profile."),
                "bins": _prop("integer", "Histogram bin count.", default=15),
            }, "required": ["key", "field"]},
            handler=lambda key, field, bins=None: bh.distribution(key, field, bins),
            group="stats",
        ),
        ToolSpec(
            name="bh_outliers",
            description="Flag anomalous rows via z-score, IQR, or both. Returns per-field outlier counts and a ranked list of most anomalous rows with severity scores.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields (empty = all numeric)."),
                "method": _prop("string", "Detection method: 'zscore', 'iqr', or 'both'.", default="both"),
                "threshold": _prop("number", "Z-score threshold.", default=3.0),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, method="both", threshold=None: bh.outliers(key, fields_json, method, threshold),
            group="stats",
        ),
        ToolSpec(
            name="bh_benford",
            description="Benford's Law first-digit test. Compares leading-digit frequencies against the expected log10(1+1/d) distribution. Flags data fabrication or anomalies.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column to test."),
            }, "required": ["key", "field"]},
            handler=lambda key, field: bh.benford(key, field),
            group="stats",
        ),
        ToolSpec(
            name="bh_vif",
            description="Variance Inflation Factor: detects multicollinearity among numeric columns. Returns VIF per field, R² with others, and drop candidates.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields (empty = all numeric)."),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json: bh.vif(key, fields_json),
            group="stats",
        ),
        ToolSpec(
            name="bh_effective_dimension",
            description="Intrinsic dimensionality via participation ratio and MLE. Reveals how many independent degrees of freedom the data actually has.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields (empty = all numeric)."),
                "method": _prop("string", "'pr', 'mle', or 'both'.", default="both"),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, method="both": bh.effective_dimension(key, fields_json, method),
            group="stats",
        ),
        ToolSpec(
            name="bh_rank_corr",
            description="Spearman rank correlation matrix with Pearson comparison. Flags nonlinear relationships where rank and linear correlations diverge.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "method": _prop("string", "Currently 'spearman'.", default="spearman"),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, method="spearman": bh.rank_corr(key, fields_json, method),
            group="stats",
        ),
        ToolSpec(
            name="bh_chi_squared",
            description="Chi-squared test of independence between two columns. Returns chi² statistic, p-value, degrees of freedom, and Cramér's V effect size.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field_a": _prop("string", "First field."),
                "field_b": _prop("string", "Second field."),
                "bins": _prop("integer", "Bins for continuous→categorical conversion.", default=10),
            }, "required": ["key", "field_a", "field_b"]},
            handler=lambda key, field_a, field_b, bins=None: bh.chi_squared(key, field_a, field_b, bins),
            group="stats",
        ),
        ToolSpec(
            name="bh_anova",
            description="One-way ANOVA: does a categorical group field explain variance in a numeric target? Returns F-statistic, p-value, eta-squared, and group means.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "group_field": _prop("string", "Categorical grouping column."),
                "target_field": _prop("string", "Numeric target column."),
            }, "required": ["key", "group_field", "target_field"]},
            handler=lambda key, group_field, target_field: bh.anova(key, group_field, target_field),
            group="stats",
        ),
        ToolSpec(
            name="bh_mutual_info_matrix",
            description="Pairwise mutual information matrix. Captures ALL dependencies (linear and nonlinear). Flags pairs where MI greatly exceeds Pearson correlation.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "bins": _prop("integer", "Discretization bins.", default=10),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, bins=None: bh.mutual_info_matrix(key, fields_json, bins),
            group="stats",
        ),
        ToolSpec(
            name="bh_hsic",
            description="Hilbert-Schmidt Independence Criterion: kernel-based independence test with permutation p-value. Detects arbitrary nonlinear dependencies.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field_a": _prop("string", "First field."),
                "field_b": _prop("string", "Second field."),
                "kernel": _prop("string", "Kernel type.", default="gaussian"),
                "n_permutations": _prop("integer", "Permutation count.", default=100),
            }, "required": ["key", "field_a", "field_b"]},
            handler=lambda key, field_a, field_b, kernel="gaussian", n_permutations=None: bh.hsic(key, field_a, field_b, kernel, n_permutations),
            group="stats",
        ),
        ToolSpec(
            name="bh_copula_tail",
            description="Tail dependence via empirical copula. Measures upper and lower tail co-movement that correlations miss. Critical for risk/crisis analysis.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field_a": _prop("string", "First field."),
                "field_b": _prop("string", "Second field."),
                "tail_threshold": _prop("number", "Quantile threshold for tail.", default=0.05),
            }, "required": ["key", "field_a", "field_b"]},
            handler=lambda key, field_a, field_b, tail_threshold=None: bh.copula_tail(key, field_a, field_b, tail_threshold),
            group="stats",
        ),
        ToolSpec(
            name="bh_polynomial_test",
            description="Fit polynomial regressions degree 1–3 and compare R². Detects curvilinear relationships that OLS misses.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field_x": _prop("string", "Predictor column."),
                "field_y": _prop("string", "Response column."),
                "max_degree": _prop("integer", "Max polynomial degree.", default=3),
            }, "required": ["key", "field_x", "field_y"]},
            handler=lambda key, field_x, field_y, max_degree=3: bh.polynomial_test(key, field_x, field_y, max_degree),
            group="stats",
        ),
        ToolSpec(
            name="bh_interaction_screen",
            description="Screen pairs of candidates for non-additive (synergy/suppression) effects on a target. Finds combinations stronger than individual effects.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "target": _prop("string", "Target column."),
                "candidates_json": _prop("string", "JSON list of candidate driver columns."),
                "top_k": _prop("integer", "Max interactions to return.", default=5),
            }, "required": ["key", "target", "candidates_json"]},
            handler=lambda key, target, candidates_json, top_k=None: bh.interaction_screen(key, target, candidates_json, top_k),
            group="stats",
        ),
        ToolSpec(
            name="bh_sparse_drivers",
            description="LASSO (L1) variable selection with cross-validated regularization. Returns which features survive penalization and which are eliminated.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "target": _prop("string", "Target column."),
                "candidates_json": _prop("string", "JSON list of candidate columns."),
                "alpha": _prop("number", "L1 penalty (auto if omitted)."),
                "max_features": _prop("integer", "Max features to select.", default=10),
            }, "required": ["key", "target", "candidates_json"]},
            handler=lambda key, target, candidates_json, alpha=None, max_features=None: bh.sparse_drivers(key, target, candidates_json, alpha, max_features),
            group="stats",
        ),
        ToolSpec(
            name="bh_feature_importance",
            description="Permutation-based model-agnostic feature importance. Shuffles each feature and measures R² drop — no model assumptions.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "target": _prop("string", "Target column."),
                "candidates_json": _prop("string", "JSON list of candidate columns."),
                "n_shuffles": _prop("integer", "Number of shuffles per feature.", default=10),
            }, "required": ["key", "target", "candidates_json"]},
            handler=lambda key, target, candidates_json, n_shuffles=None: bh.feature_importance(key, target, candidates_json, n_shuffles),
            group="stats",
        ),
        ToolSpec(
            name="bh_information_bottleneck",
            description="Information Bottleneck: finds optimal lossy compression of input fields that maximally preserves information about the target. Reveals essential structure.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "input_fields_json": _prop("string", "JSON list of input field names."),
                "target_field": _prop("string", "Target column."),
                "beta": _prop("number", "Compression-relevance trade-off.", default=1.0),
                "n_clusters": _prop("integer", "Compressed state count.", default=5),
            }, "required": ["key", "input_fields_json", "target_field"]},
            handler=lambda key, input_fields_json, target_field, beta=None, n_clusters=None: bh.information_bottleneck(key, input_fields_json, target_field, beta, n_clusters),
            group="stats",
        ),
        ToolSpec(
            name="bh_cluster",
            description="K-means clustering with auto-k selection via silhouette scoring. Returns cluster centroids, sizes, and per-row assignments.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "k": _prop("integer", "Fixed cluster count (auto if omitted)."),
                "max_k": _prop("integer", "Max k to try for auto selection.", default=8),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, k=None, max_k=None: bh.cluster(key, fields_json, k, max_k),
            group="stats",
        ),
        ToolSpec(
            name="bh_spectral_decomposition",
            description="Graph Laplacian eigenspectrum of k-NN similarity graph. Reveals cluster structure via spectral gap and Fiedler partition.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "n_neighbors": _prop("integer", "k-NN neighbors.", default=10),
                "n_components": _prop("integer", "Eigenvalues to return.", default=5),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, n_neighbors=None, n_components=None: bh.spectral_decomposition(key, fields_json, n_neighbors, n_components),
            group="stats",
        ),
        ToolSpec(
            name="bh_latent_sources",
            description="Independent Component Analysis (FastICA). Decomposes observed fields into statistically independent latent sources with non-Gaussianity scores.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "n_sources": _prop("integer", "Number of sources (auto if omitted)."),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, n_sources=None: bh.latent_sources(key, fields_json, n_sources),
            group="stats",
        ),
        ToolSpec(
            name="bh_graphical_model",
            description="Sparse precision matrix → conditional dependency graph. Reveals which variables are directly linked after controlling for all others.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "method": _prop("string", "Estimation method.", default="threshold"),
                "alpha": _prop("number", "Regularization strength.", default=0.01),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, method="threshold", alpha=None: bh.graphical_model(key, fields_json, method, alpha),
            group="stats",
        ),
        ToolSpec(
            name="bh_persistent_topology",
            description="Persistent homology via Vietoris-Rips filtration. Computes H0 (connected components) and H1 (loops) persistence diagrams and Betti number traces.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "max_dimension": _prop("integer", "Max homology dimension.", default=1),
                "max_points": _prop("integer", "Subsample size for performance.", default=200),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, max_dimension=1, max_points=None: bh.persistent_topology(key, fields_json, max_dimension, max_points),
            group="stats",
        ),
        ToolSpec(
            name="bh_causal_graph",
            description="PC algorithm: discovers causal DAG skeleton and orients v-structures via conditional independence tests. Shows directed and undirected edges.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "alpha": _prop("number", "Significance level.", default=0.05),
                "max_conditioning_set": _prop("integer", "Max conditioning set size.", default=3),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, alpha=None, max_conditioning_set=None: bh.causal_graph(key, fields_json, alpha, max_conditioning_set),
            group="stats",
        ),
        ToolSpec(
            name="bh_transfer_entropy",
            description="Directional information flow between time-ordered fields. Determines which variable 'Granger-causes' the other using information theory.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "source_field": _prop("string", "Putative cause field."),
                "target_field": _prop("string", "Putative effect field."),
                "order_by": _prop("string", "Time/ordering column."),
                "max_lag": _prop("integer", "Max lag to test.", default=5),
                "bins": _prop("integer", "Discretization bins.", default=8),
            }, "required": ["key", "source_field", "target_field", "order_by"]},
            handler=lambda key, source_field, target_field, order_by, max_lag=None, bins=None: bh.transfer_entropy(key, source_field, target_field, order_by, max_lag, bins),
            group="stats",
        ),
        ToolSpec(
            name="bh_do_estimate",
            description="Backdoor-criterion causal effect estimation via regression adjustment. Answers: what is the causal effect of treatment on outcome, controlling for confounders?",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "treatment": _prop("string", "Treatment/cause column."),
                "outcome": _prop("string", "Outcome/effect column."),
                "confounders_json": _prop("string", "JSON list of confounder columns."),
                "method": _prop("string", "Estimation method.", default="regress"),
            }, "required": ["key", "treatment", "outcome", "confounders_json"]},
            handler=lambda key, treatment, outcome, confounders_json, method="regress": bh.do_estimate(key, treatment, outcome, confounders_json, method),
            group="stats",
        ),
        ToolSpec(
            name="bh_counterfactual_impact",
            description="Synthetic control: builds a counterfactual from donor units to estimate what would have happened without an intervention.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "outcome_field": _prop("string", "Outcome column."),
                "time_field": _prop("string", "Time column."),
                "unit_field": _prop("string", "Unit/entity column."),
                "treated_unit": _prop("string", "Name of the treated unit."),
                "intervention_time": _prop("string", "Time point of intervention."),
                "donor_units_json": _prop("string", "JSON list of donor unit names (all if omitted)."),
            }, "required": ["key", "outcome_field", "time_field", "unit_field", "treated_unit", "intervention_time"]},
            handler=lambda key, outcome_field, time_field, unit_field, treated_unit, intervention_time, donor_units_json=None: bh.counterfactual_impact(key, outcome_field, time_field, unit_field, treated_unit, intervention_time, donor_units_json),
            group="stats",
        ),
        ToolSpec(
            name="bh_autocorrelation",
            description="Autocorrelation function at multiple lags with significance bounds and periodicity detection. Diagnoses temporal dependencies and dominant cycles.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column."),
                "order_by": _prop("string", "Time/ordering column."),
                "max_lag": _prop("integer", "Maximum lag.", default=20),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, max_lag=None: bh.autocorrelation(key, field, order_by, max_lag),
            group="stats",
        ),
        ToolSpec(
            name="bh_changepoints",
            description="CUSUM-based structural break detection in ordered data. Finds where the data-generating process shifted (regime changes, level shifts).",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column."),
                "order_by": _prop("string", "Time/ordering column."),
                "min_segment": _prop("integer", "Minimum segment length.", default=10),
                "threshold": _prop("number", "Detection threshold (std units).", default=2.0),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, min_segment=None, threshold=None: bh.changepoints(key, field, order_by, min_segment, threshold),
            group="stats",
        ),
        ToolSpec(
            name="bh_rolling_analysis",
            description="Windowed statistics (mean, std, min, max) over ordered data. Optionally computes rolling correlation with a second field. Tracks trend and volatility clustering.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column."),
                "order_by": _prop("string", "Time/ordering column."),
                "window": _prop("integer", "Rolling window size.", default=20),
                "field_b": _prop("string", "Optional second field for rolling correlation."),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, window=None, field_b=None: bh.rolling_analysis(key, field, order_by, window, field_b),
            group="stats",
        ),
        ToolSpec(
            name="bh_phase_space",
            description="Takens' delay embedding attractor reconstruction. Estimates optimal embedding dimension, Lyapunov exponent, and attractor type (chaotic, limit cycle, fixed point).",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric time series column."),
                "order_by": _prop("string", "Time/ordering column."),
                "max_embedding_dim": _prop("integer", "Max embedding dimension.", default=10),
                "tau": _prop("integer", "Delay parameter (auto if omitted)."),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, max_embedding_dim=None, tau=None: bh.phase_space(key, field, order_by, max_embedding_dim, tau),
            group="stats",
        ),
        ToolSpec(
            name="bh_ergodicity_test",
            description="Tests whether time averages converge to ensemble averages. Non-ergodic data means sample statistics can be deeply misleading.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric column."),
                "order_by": _prop("string", "Time/ordering column."),
                "window_sizes_json": _prop("string", "JSON list of window sizes to test.", default="[10, 25, 50, 100]"),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, window_sizes_json=None: bh.ergodicity_test(key, field, order_by, window_sizes_json),
            group="stats",
        ),
        ToolSpec(
            name="bh_recurrence_analysis",
            description="Recurrence Quantification Analysis: determinism, laminarity, trapping time, max diagonal length, and entropy of diagonal lines. Characterizes dynamical complexity.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "order_by": _prop("string", "Time/ordering column."),
                "threshold": _prop("number", "Recurrence threshold (auto if omitted)."),
                "embedding_dim": _prop("integer", "Embedding dimension.", default=3),
            }, "required": ["key", "fields_json", "order_by"]},
            handler=lambda key, fields_json, order_by, threshold=None, embedding_dim=None: bh.recurrence_analysis(key, fields_json, order_by, threshold, embedding_dim),
            group="stats",
        ),
        ToolSpec(
            name="bh_entropy_spectrum",
            description="Multi-scale sample entropy. Reveals complexity at different time scales — is the signal simple at fine scales but complex at coarse? Or scale-invariant?",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "field": _prop("string", "Numeric time series column."),
                "order_by": _prop("string", "Time/ordering column."),
                "max_scale": _prop("integer", "Maximum coarsening scale.", default=10),
                "embedding_dim": _prop("integer", "Embedding dimension for sample entropy.", default=2),
            }, "required": ["key", "field", "order_by"]},
            handler=lambda key, field, order_by, max_scale=None, embedding_dim=None: bh.entropy_spectrum(key, field, order_by, max_scale, embedding_dim),
            group="stats",
        ),
        ToolSpec(
            name="bh_renormalization_flow",
            description="Multiscale coarse-graining: tracks how correlations between fields change as data is averaged at increasing block sizes. Identifies scale-invariant or diverging coupling.",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
                "max_scale": _prop("integer", "Maximum block-doubling scale.", default=5),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json, max_scale=None: bh.renormalization_flow(key, fields_json, max_scale),
            group="stats",
        ),
        ToolSpec(
            name="bh_symmetry_scan",
            description="Scans for data invariances: translation (differences simpler than levels), scale (log-transform normalizes), reflection (symmetric distributions), and permutation (exchangeable columns).",
            input_schema={"type": "object", "properties": {
                "key": _prop("string", "Dataset key."),
                "fields_json": _prop("string", "JSON list of numeric fields."),
            }, "required": ["key", "fields_json"]},
            handler=lambda key, fields_json: bh.symmetry_scan(key, fields_json),
            group="stats",
        ),
    ]
