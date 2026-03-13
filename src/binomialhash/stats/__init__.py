"""Stats package — backward-compatible re-exports from sub-modules."""

# ── helpers ──────────────────────────────────────────────────────────
from ._helpers import (
    DEFAULT_STATS_POLICY,
    StatsPolicy,
    _ALL_AGG_FUNCS,
    _NUMERIC_FUNCS,
    agg_numeric,
    bucket_index,
    extract_numeric_matrix,
    extract_numeric_pairs,
    fit_linear,
    normal_cdf,
    np,
    numeric_column_values,
    ols_r2,
    pearson_corr,
    quantile_edges,
    run_agg,
    shannon_entropy,
    spearman_rank,
    to_float_permissive,
)

# ── existing regression / screen / solver ────────────────────────────
from .regression import (
    dependency_screen_dataset,
    partial_correlate_dataset,
    pca_surface_dataset,
    regress_dataset,
    solve_over_rows,
)

# ── Stage 1: Data Quality ───────────────────────────────────────────
from .quality import (
    benford_dataset,
    distribution_dataset,
    effective_dimension_dataset,
    outliers_dataset,
    vif_dataset,
)

# ── Stage 2: Dependency Mapping ─────────────────────────────────────
from .dependency import (
    anova_dataset,
    chi_squared_dataset,
    copula_tail_dataset,
    hsic_dataset,
    mutual_info_matrix_dataset,
    rank_corr_dataset,
)

# ── Stage 3: Driver Discovery ──────────────────────────────────────
from .drivers import (
    feature_importance_dataset,
    information_bottleneck_dataset,
    interaction_screen_dataset,
    polynomial_test_dataset,
    sparse_drivers_dataset,
)

# ── Stage 4: Structure & Topology ──────────────────────────────────
from .structure import (
    cluster_dataset,
    graphical_model_dataset,
    latent_sources_dataset,
    persistent_topology_dataset,
    spectral_decomposition_dataset,
)

# ── Stage 5: Causal Inference ──────────────────────────────────────
from .causal import (
    causal_graph_dataset,
    counterfactual_impact_dataset,
    do_estimate_dataset,
    transfer_entropy_dataset,
)

# ── Stage 6: Temporal Dynamics ─────────────────────────────────────
from .dynamics import (
    autocorrelation_dataset,
    changepoints_dataset,
    ergodicity_test_dataset,
    phase_space_dataset,
    recurrence_analysis_dataset,
    rolling_analysis_dataset,
)

# ── Stage 7: Scale, Symmetry, Laws ────────────────────────────────
from .laws import (
    entropy_spectrum_dataset,
    renormalization_flow_dataset,
    symmetry_scan_dataset,
)

__all__ = [
    # helpers
    "_ALL_AGG_FUNCS",
    "_NUMERIC_FUNCS",
    "DEFAULT_STATS_POLICY",
    "StatsPolicy",
    "agg_numeric",
    "bucket_index",
    "extract_numeric_matrix",
    "extract_numeric_pairs",
    "fit_linear",
    "normal_cdf",
    "numeric_column_values",
    "ols_r2",
    "pearson_corr",
    "quantile_edges",
    "run_agg",
    "shannon_entropy",
    "spearman_rank",
    "to_float_permissive",
    # existing
    "dependency_screen_dataset",
    "partial_correlate_dataset",
    "pca_surface_dataset",
    "regress_dataset",
    "solve_over_rows",
    # stage 1
    "benford_dataset",
    "distribution_dataset",
    "effective_dimension_dataset",
    "outliers_dataset",
    "vif_dataset",
    # stage 2
    "anova_dataset",
    "chi_squared_dataset",
    "copula_tail_dataset",
    "hsic_dataset",
    "mutual_info_matrix_dataset",
    "rank_corr_dataset",
    # stage 3
    "feature_importance_dataset",
    "information_bottleneck_dataset",
    "interaction_screen_dataset",
    "polynomial_test_dataset",
    "sparse_drivers_dataset",
    # stage 4
    "cluster_dataset",
    "graphical_model_dataset",
    "latent_sources_dataset",
    "persistent_topology_dataset",
    "spectral_decomposition_dataset",
    # stage 5
    "causal_graph_dataset",
    "counterfactual_impact_dataset",
    "do_estimate_dataset",
    "transfer_entropy_dataset",
    # stage 6
    "autocorrelation_dataset",
    "changepoints_dataset",
    "ergodicity_test_dataset",
    "phase_space_dataset",
    "recurrence_analysis_dataset",
    "rolling_analysis_dataset",
    # stage 7
    "entropy_spectrum_dataset",
    "renormalization_flow_dataset",
    "symmetry_scan_dataset",
]
