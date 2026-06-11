# -*- coding: utf-8 -*-
"""
diagnostics.py

Computes statistical convergence validations including LOOIC, high-correlation checks,
multi-chain bulk and tail ESS checks, and parameter divergence detections [1].
"""

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def run_diagnostic_checks(
        idata: az.InferenceData,
        df_long: pd.DataFrame,
        struct_maps: Dict[str, Any]
) -> Tuple[List[str], pd.DataFrame]:
    """
    Computes diagnostics on chains, reporting issues like low ESS (bulk/tail),
    divergences, high correlations, and high Pareto k values [1].

    Args:
        idata: ArviZ InferenceData container.
        df_long: Long format observation DataFrame.
        struct_maps: Index mapping dictionaries.

    Returns:
        List of generated warnings, and the compiled ArviZ stats summary DataFrame.
    """
    print("\n--- RUNNING SYSTEM MODEL CONVERGENCE DIAGNOSTICS ---")
    warnings_found: List[str] = []

    # 1. Low sample sizes per team group
    responses_per_group = df_long.groupby("group").size()
    low_groups = responses_per_group[responses_per_group < config.MIN_RESPONSES_PER_GROUP]
    for group, size in low_groups.items():
        warnings_found.append(
            f"WARNING: Group '{group}' has only {size} responses (minimum threshold: {config.MIN_RESPONSES_PER_GROUP})."
        )

    # 2. Divergent Transitions Check
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats.data_vars:
        divergences = idata.sample_stats.diverging.sum().item()
        if divergences > 0:
            warnings_found.append(
                f"CRITICAL WARNING: {divergences} divergent transitions detected. Parameterization may be unstable!"
            )
        else:
            print("  Sampler Divergences: OK (0 found)")

    # 3. Compile ArviZ Summaries
    summary = az.summary(idata, var_names=["a", "theta_sec"], round_to=3)
    max_rhat = summary["r_hat"].max()
    print(f"  Maximum observed R-hat: {max_rhat:.3f}")
    if max_rhat > config.RHAT_THRESHOLD:
        warnings_found.append(
            f"WARNING: Chain convergence is unstable with R-hat > {config.RHAT_THRESHOLD} ({max_rhat:.3f})."
        )

    # Verify both ESS Bulk and ESS Tail are sufficient
    min_ess_bulk = summary["ess_bulk"].min()
    min_ess_tail = summary["ess_tail"].min()
    min_ess_observed = min(min_ess_bulk, min_ess_tail)
    target_ess = config.NEFF_RATIO_THRESHOLD * config.ITER_SAMPLING * config.CHAINS
    print(f"  Minimum observed Effective Sample Size (ESS): {min_ess_observed:.1f}")
    if min_ess_observed < target_ess:
        warnings_found.append(
            f"WARNING: Low ESS ({min_ess_observed:.1f}) detected (Threshold: {target_ess:.0f})."
        )

    # 4. Item Discrimination Checks
    questions_list = struct_maps["questions"]
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    a_means = posterior_means["a"].values
    for idx, q_name in enumerate(questions_list):
        if a_means[idx] < config.LOW_DISCRIMINATION_THRESHOLD:
            warnings_found.append(
                f"WARNING: Item '{q_name}' has extremely low discrimination of {a_means[idx]:.3f} (Threshold: {config.LOW_DISCRIMINATION_THRESHOLD})."
            )

    # 5. Dimension Redundancy Checks (HIGH_CORR_THRESHOLD) [1]
    # Evaluate Section-level correlations to recommend structural merges
    sections_list = struct_maps["sections"]
    S = len(sections_list)
    Omega_mean = posterior_means["Omega"].values
    if S > 1:
        redundant_pairs = []
        for i in range(S):
            for j in range(i + 1, S):
                corr_val = Omega_mean[i, j]
                if abs(corr_val) > config.HIGH_CORR_THRESHOLD:
                    redundant_pairs.append((sections_list[i], sections_list[j], corr_val))
                    warnings_found.append(
                        f"REDUNDANCY WARNING: Sections '{sections_list[i]}' and '{sections_list[j]}' are highly correlated "
                        f"({corr_val:.3f} > {config.HIGH_CORR_THRESHOLD}). Consider merging these sections [1]!"
                    )
        if redundant_pairs:
            print("\n  [STRUCTURAL DIAGNOSTIC] Redundancy Detected:")
            for p1, p2, val in redundant_pairs:
                print(f"    - '{p1}' <---> '{p2}' (Correlation: {val:.3f}) -> Suggest survey merge [1].")

    # 6. Model Fit LOOIC / Pareto k Evaluation [1]
    print("  Calculating Leave-One-Out Cross-Validation (LOOIC)...")
    try:
        # Prevent runtime warnings during log-likelihood checks if model coordinates are basic
        loo_results = az.loo(idata, pointwise=True)
        max_k = np.max(loo_results.pareto_k.values)
        print(f"  Maximum observed Pareto k value: {max_k:.3f}")

        problematic_k = np.sum(loo_results.pareto_k.values > config.PARETO_K_THRESHOLD)
        if problematic_k > 0:
            warnings_found.append(
                f"WARNING: {problematic_k} observations exhibit high Pareto k values > {config.PARETO_K_THRESHOLD} [1]. "
                "These data points are highly influential/outliers [1]."
            )
            # Save khat diagnostic graph
            az.plot_khat(loo_results, threshold=config.PARETO_K_THRESHOLD)
            plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "looic_pareto_k_diagnostic.png"), bbox_inches="tight")
            plt.close()
            print("  Saved LOOIC Pareto k diagnostic plot to diagnostics/looic_pareto_k_diagnostic.png.")
    except Exception as e:
        print(f"  [SKIPPED] LOOIC calculations skipped: {e}")

    return warnings_found, summary
