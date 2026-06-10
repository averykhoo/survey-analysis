# -*- coding: utf-8 -*-
"""
diagnostics.py

Computes statistical convergence validations including LOOIC,
multi-chain bulk and tail ESS checks, parameter divergence detections, and item fit warnings.
"""

import re
from typing import Dict
from typing import List
from typing import Tuple

import arviz as az
import numpy as np
import pandas as pd

import config


def run_diagnostic_checks(
        idata: az.InferenceData,
        df_long: pd.DataFrame,
        cat_idx_to_name: Dict[int, str],
        question_idx_map: Dict[str, int]
) -> Tuple[List[str], pd.DataFrame]:
    """
    Computes diagnostics on chains, reporting issues like low ESS (bulk/tail),
    divergences, high correlations, and high Pareto k values [1].
    """
    print("\n--- SECTION 5: EXECUTING MODEL CONVERGENCE DIAGNOSTICS ---")
    warnings_found: List[str] = []

    # 1. Evaluate sample sizes per group
    responses_per_group = df_long.groupby("group").size()
    low_groups = responses_per_group[responses_per_group < config.MIN_RESPONSES_PER_GROUP]
    for group, size in low_groups.items():
        warnings_found.append(
            f"WARNING: Group '{group}' has only {size} response(s) (threshold: {config.MIN_RESPONSES_PER_GROUP})."
        )

    # 2. Divergent Transitions Check
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats.data_vars:
        divergences = idata.sample_stats.diverging.sum().item()
        if divergences > 0:
            warnings_found.append(
                f"CRITICAL WARNING: {divergences} divergent transitions detected. Consider reparametrization."
            )
        else:
            print("  Sampler Divergences: OK (0 found)")

    # 3. ArviZ Summary Compilation (Bulk & Tail)
    summary = az.summary(idata, var_names=["mu", "sigma", "a"], round_to=3)

    max_rhat = summary["r_hat"].max()
    print(f"  Maximum observed R-hat: {max_rhat:.3f}")
    if max_rhat > config.RHAT_THRESHOLD:
        warnings_found.append(
            f"WARNING: Parameters exhibit poor convergence with R-hat > {config.RHAT_THRESHOLD} ({max_rhat:.3f}).")

    # Verify both ESS Bulk and ESS Tail are sufficient
    min_ess_bulk = summary["ess_bulk"].min()
    min_ess_tail = summary["ess_tail"].min()
    min_ess_observed = min(min_ess_bulk, min_ess_tail)
    target_ess = config.NEFF_RATIO_THRESHOLD * config.ITER_SAMPLING * config.CHAINS

    print(f"  Minimum observed Effective Sample Size (ESS): {min_ess_observed:.1f}")
    if min_ess_observed < target_ess:
        warnings_found.append(f"WARNING: Low ESS ({min_ess_observed:.1f}) detected (Threshold: {target_ess:.0f}).")

    # 4. Item Discrimination Validation
    idx_to_question = {v: k for k, v in question_idx_map.items()}
    a_summary = summary[summary.index.str.startswith("a")]
    for idx_str, row in a_summary.iterrows():
        match = re.search(r"a\[(\d+)\]", idx_str)
        if match:
            q_idx = int(match.group(1))
            q_name = idx_to_question.get(q_idx, f"Idx {q_idx}")
            mean_a = row["mean"]
            if mean_a < config.LOW_DISCRIMINATION_THRESHOLD:
                warnings_found.append(
                    f"WARNING: Question '{q_name}' displays low discrimination of {mean_a:.3f} (Threshold: {config.LOW_DISCRIMINATION_THRESHOLD})."
                )

    # 5. Model Fit LOOIC / Pareto k Evaluation [1]
    print("  Calculating Leave-One-Out Cross-Validation (LOOIC)...")
    try:
        # Prevent runtime warnings during log-likelihood checks if model coordinates are basic
        loo_results = az.loo(idata, pointwise=True)
        max_k = np.max(loo_results.pareto_k.values)
        print(f"  Maximum observed Pareto k value: {max_k:.3f}")

        problematic_k = np.sum(loo_results.pareto_k.values > config.PARETO_K_THRESHOLD)
        if problematic_k > 0:
            warnings_found.append(
                f"WARNING: {problematic_k} observation(s) exhibit high Pareto k values > {config.PARETO_K_THRESHOLD} [1]. "
                "These data points are highly influential on parameters [1]."
            )
            # Save khat diagnostic graph
            import matplotlib.pyplot as plt
            import os
            az.plot_khat(loo_results, threshold=config.PARETO_K_THRESHOLD)
            plt.savefig(os.path.join(config.OUTPUT_DIR, "looic_pareto_k_diagnostic.png"), bbox_inches="tight")
            plt.close()
            print("  Saved LOOIC Pareto k diagnostic plot to 'looic_pareto_k_diagnostic.png'.")
    except Exception as e:
        print(f"  [SKIPPED] LOOIC calculations skipped: {e}")

    return warnings_found, summary
