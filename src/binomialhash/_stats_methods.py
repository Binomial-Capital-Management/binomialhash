"""Stats analysis methods mixed into BinomialHash."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from .predicates import build_predicate
from .stats import (
    regress_dataset, partial_correlate_dataset, pca_surface_dataset,
    dependency_screen_dataset, solve_over_rows,
    distribution_dataset, outliers_dataset, benford_dataset,
    vif_dataset, effective_dimension_dataset,
    rank_corr_dataset, chi_squared_dataset, anova_dataset,
    mutual_info_matrix_dataset, hsic_dataset, copula_tail_dataset,
    polynomial_test_dataset, interaction_screen_dataset,
    sparse_drivers_dataset, feature_importance_dataset,
    information_bottleneck_dataset,
    cluster_dataset, spectral_decomposition_dataset,
    latent_sources_dataset, graphical_model_dataset,
    persistent_topology_dataset,
    causal_graph_dataset, transfer_entropy_dataset,
    do_estimate_dataset, counterfactual_impact_dataset,
    autocorrelation_dataset, changepoints_dataset,
    rolling_analysis_dataset, phase_space_dataset,
    ergodicity_test_dataset, recurrence_analysis_dataset,
    entropy_spectrum_dataset, renormalization_flow_dataset,
    symmetry_scan_dataset,
)

logger = logging.getLogger(__name__)


class _StatsMethodsMixin:
    """All 39 statistical analysis methods.  Mixed into BinomialHash."""

    # Template method: look up slot, run analysis, track output size, log timing.
    def _stat(self, name: str, key: str, fn) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = self._get_slot(key)
        if slot is None:
            return {"error": f"Key '{key}' not found."}
        result = fn(slot)
        if isinstance(result, dict) and "error" in result:
            return result
        result = {"key": key, **result}
        self._track(0, len(json.dumps(result, default=str)))
        logger.info("[BH-perf] %s '%s' | %.1fms",
                    name, key, (time.perf_counter() - t0) * 1000)
        return result

    # -- regression / screening --

    def regress(self, key: str, target: str, drivers_json: str) -> Dict[str, Any]:
        return self._stat("regress", key,
            lambda s: regress_dataset(s.rows, s.columns, s.col_types, target, drivers_json))

    def partial_correlate(self, key: str, field_a: str, field_b: str,
                          controls_json: str) -> Dict[str, Any]:
        return self._stat("partial_corr", key,
            lambda s: partial_correlate_dataset(s.rows, s.col_types, field_a, field_b, controls_json))

    def pca_surface(self, key: str, fields_json: str,
                    n_components: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("pca", key,
            lambda s: pca_surface_dataset(s.rows, s.col_types, fields_json, n_components))

    def dependency_screen(self, key: str, target: str, candidates_json: str,
                          controls_json: str = "[]",
                          top_k: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("dep_screen", key,
            lambda s: dependency_screen_dataset(
                s.rows, s.col_types, target, candidates_json, controls_json, top_k))

    def solver(self, key: str, target_field: str, goal_json: str,
               controllable_vars_json: str, constraints_json: str = "[]",
               top_k: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("solver", key,
            lambda s: solve_over_rows(
                s.rows, s.col_types, target_field, goal_json,
                controllable_vars_json, constraints_json, top_k,
                predicate_builder=build_predicate))

    # -- data quality --

    def distribution(self, key: str, field: str,
                     bins: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("distribution", key,
            lambda s: distribution_dataset(s.rows, s.col_types, field, bins))

    def outliers(self, key: str, fields_json: str, method: str = "both",
                 threshold: Optional[float] = None) -> Dict[str, Any]:
        return self._stat("outliers", key,
            lambda s: outliers_dataset(s.rows, s.col_types, fields_json, method, threshold))

    def benford(self, key: str, field: str) -> Dict[str, Any]:
        return self._stat("benford", key,
            lambda s: benford_dataset(s.rows, s.col_types, field))

    def vif(self, key: str, fields_json: str) -> Dict[str, Any]:
        return self._stat("vif", key,
            lambda s: vif_dataset(s.rows, s.col_types, fields_json))

    def effective_dimension(self, key: str, fields_json: str,
                            method: str = "both") -> Dict[str, Any]:
        return self._stat("effective_dim", key,
            lambda s: effective_dimension_dataset(s.rows, s.col_types, fields_json, method))

    # -- dependency mapping --

    def rank_corr(self, key: str, fields_json: str,
                  method: str = "spearman") -> Dict[str, Any]:
        return self._stat("rank_corr", key,
            lambda s: rank_corr_dataset(s.rows, s.col_types, fields_json, method))

    def chi_squared(self, key: str, field_a: str, field_b: str,
                    bins: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("chi_sq", key,
            lambda s: chi_squared_dataset(s.rows, s.col_types, field_a, field_b, bins))

    def anova(self, key: str, group_field: str,
              target_field: str) -> Dict[str, Any]:
        return self._stat("anova", key,
            lambda s: anova_dataset(s.rows, s.col_types, group_field, target_field))

    def mutual_info_matrix(self, key: str, fields_json: str,
                           bins: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("mutual_info", key,
            lambda s: mutual_info_matrix_dataset(s.rows, s.col_types, fields_json, bins))

    def hsic(self, key: str, field_a: str, field_b: str,
             kernel: str = "gaussian",
             n_permutations: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("hsic", key,
            lambda s: hsic_dataset(s.rows, s.col_types, field_a, field_b, kernel, n_permutations))

    def copula_tail(self, key: str, field_a: str, field_b: str,
                    tail_threshold: Optional[float] = None) -> Dict[str, Any]:
        return self._stat("copula_tail", key,
            lambda s: copula_tail_dataset(s.rows, s.col_types, field_a, field_b, tail_threshold))

    # -- driver discovery --

    def polynomial_test(self, key: str, field_x: str, field_y: str,
                        max_degree: int = 3) -> Dict[str, Any]:
        return self._stat("poly_test", key,
            lambda s: polynomial_test_dataset(s.rows, s.col_types, field_x, field_y, max_degree))

    def interaction_screen(self, key: str, target: str, candidates_json: str,
                           top_k: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("interaction", key,
            lambda s: interaction_screen_dataset(s.rows, s.col_types, target, candidates_json, top_k))

    def sparse_drivers(self, key: str, target: str, candidates_json: str,
                       alpha: Optional[float] = None,
                       max_features: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("lasso", key,
            lambda s: sparse_drivers_dataset(
                s.rows, s.col_types, target, candidates_json, alpha, max_features))

    def feature_importance(self, key: str, target: str, candidates_json: str,
                           n_shuffles: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("feat_imp", key,
            lambda s: feature_importance_dataset(
                s.rows, s.col_types, target, candidates_json, n_shuffles))

    def information_bottleneck(self, key: str, input_fields_json: str,
                               target_field: str, beta: Optional[float] = None,
                               n_clusters: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("info_bottleneck", key,
            lambda s: information_bottleneck_dataset(
                s.rows, s.col_types, input_fields_json, target_field, beta, n_clusters))

    # -- structure & topology --

    def cluster(self, key: str, fields_json: str, k: Optional[int] = None,
                max_k: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("cluster", key,
            lambda s: cluster_dataset(s.rows, s.col_types, fields_json, k, max_k))

    def spectral_decomposition(self, key: str, fields_json: str,
                               n_neighbors: Optional[int] = None,
                               n_components: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("spectral", key,
            lambda s: spectral_decomposition_dataset(
                s.rows, s.col_types, fields_json, n_neighbors, n_components))

    def latent_sources(self, key: str, fields_json: str,
                       n_sources: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("ica", key,
            lambda s: latent_sources_dataset(s.rows, s.col_types, fields_json, n_sources))

    def graphical_model(self, key: str, fields_json: str,
                        method: str = "threshold",
                        alpha: Optional[float] = None) -> Dict[str, Any]:
        return self._stat("graph_model", key,
            lambda s: graphical_model_dataset(s.rows, s.col_types, fields_json, method, alpha))

    def persistent_topology(self, key: str, fields_json: str,
                            max_dimension: int = 1,
                            max_points: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("topo", key,
            lambda s: persistent_topology_dataset(
                s.rows, s.col_types, fields_json, max_dimension, max_points))

    # -- causal inference --

    def causal_graph(self, key: str, fields_json: str,
                     alpha: Optional[float] = None,
                     max_conditioning_set: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("causal", key,
            lambda s: causal_graph_dataset(
                s.rows, s.col_types, fields_json, alpha, max_conditioning_set))

    def transfer_entropy(self, key: str, source_field: str, target_field: str,
                         order_by: str, max_lag: Optional[int] = None,
                         bins: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("transfer_ent", key,
            lambda s: transfer_entropy_dataset(
                s.rows, s.col_types, source_field, target_field, order_by, max_lag, bins))

    def do_estimate(self, key: str, treatment: str, outcome: str,
                    confounders_json: str, method: str = "regress",
                    bins_count: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("do_estimate", key,
            lambda s: do_estimate_dataset(
                s.rows, s.col_types, treatment, outcome, confounders_json, method, bins_count))

    def counterfactual_impact(self, key: str, outcome_field: str, time_field: str,
                              unit_field: str, treated_unit: str,
                              intervention_time: str,
                              donor_units_json: Optional[str] = None) -> Dict[str, Any]:
        return self._stat("counterfactual", key,
            lambda s: counterfactual_impact_dataset(
                s.rows, s.col_types, outcome_field, time_field,
                unit_field, treated_unit, intervention_time, donor_units_json))

    # -- temporal dynamics --

    def autocorrelation(self, key: str, field: str, order_by: str,
                        max_lag: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("acf", key,
            lambda s: autocorrelation_dataset(s.rows, s.col_types, field, order_by, max_lag))

    def changepoints(self, key: str, field: str, order_by: str,
                     min_segment: Optional[int] = None,
                     threshold: Optional[float] = None) -> Dict[str, Any]:
        return self._stat("changepoints", key,
            lambda s: changepoints_dataset(
                s.rows, s.col_types, field, order_by, min_segment, threshold))

    def rolling_analysis(self, key: str, field: str, order_by: str,
                         window: Optional[int] = None,
                         field_b: Optional[str] = None) -> Dict[str, Any]:
        return self._stat("rolling", key,
            lambda s: rolling_analysis_dataset(s.rows, s.col_types, field, order_by, window, field_b))

    def phase_space(self, key: str, field: str, order_by: str,
                    max_embedding_dim: Optional[int] = None,
                    tau: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("phase_space", key,
            lambda s: phase_space_dataset(
                s.rows, s.col_types, field, order_by, max_embedding_dim, tau))

    def ergodicity_test(self, key: str, field: str, order_by: str,
                        window_sizes_json: Optional[str] = None) -> Dict[str, Any]:
        return self._stat("ergodicity", key,
            lambda s: ergodicity_test_dataset(s.rows, s.col_types, field, order_by, window_sizes_json))

    def recurrence_analysis(self, key: str, fields_json: str, order_by: str,
                            threshold: Optional[float] = None,
                            embedding_dim: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("recurrence", key,
            lambda s: recurrence_analysis_dataset(
                s.rows, s.col_types, fields_json, order_by, threshold, embedding_dim))

    # -- scale, symmetry, laws --

    def entropy_spectrum(self, key: str, field: str, order_by: str,
                         max_scale: Optional[int] = None,
                         embedding_dim: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("entropy", key,
            lambda s: entropy_spectrum_dataset(
                s.rows, s.col_types, field, order_by, max_scale, embedding_dim))

    def renormalization_flow(self, key: str, fields_json: str,
                             max_scale: Optional[int] = None) -> Dict[str, Any]:
        return self._stat("renorm_flow", key,
            lambda s: renormalization_flow_dataset(s.rows, s.col_types, fields_json, max_scale))

    def symmetry_scan(self, key: str, fields_json: str) -> Dict[str, Any]:
        return self._stat("symmetry", key,
            lambda s: symmetry_scan_dataset(s.rows, s.col_types, fields_json))
