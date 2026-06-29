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
import xarray


def run_diagnostic_checks(
        idata: xarray.DataTree,
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
    print("\n--- RUNNING STATISTICAL CONVERGENCE DIAGNOSTICS ---")
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
                f"CRITICAL WARNING: {divergences} divergent transitions detected. Scale parameters may be unstable!"
            )
        else:
            print("  Sampler Divergences: OK (0 found)")

    summary = az.summary(idata, var_names=["a", "theta_sec", "sigma_cat_raw"], round_to=3)
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
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
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


    try:
        az.plot_energy(idata)
        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "energy_plot.svg"), bbox_inches="tight")
        plt.close()
        print("  Saved energy diagnostic plot to diagnostics/energy_plot.svg.")
    except Exception as e:
        print(f"  [SKIPPED] Energy plot failed: {e}")

    try:
        # Check scale-indeterminacy loop resolution (a vs sigma_cat)
        # Extract the raw 2D NumPy arrays of shape (chain, draw)
        posterior_dict = {
            "a_first":             idata.posterior["a"].values[:, :, 0],
            "sigma_cat_raw_first": idata.posterior["sigma_cat_raw"].values[:, :, 0]
        }

        try:
            # Syntax for older ArviZ (0.x)
            minimal_idata = az.from_dict(posterior=posterior_dict)
        except TypeError:
            # Fallback syntax for newer ArviZ (1.x / arviz-base)
            minimal_idata = az.from_dict({"posterior": posterior_dict})

        # Multi-stage fallback to handle ArviZ 0.x vs 1.x signature parameters
        try:
            # 1. Try legacy ArviZ signature (marginals=True)
            az.plot_pair(minimal_idata, marginals=True)
        except Exception:
            try:
                # 2. Try modern ArviZ/arviz-plots signature (marginal=True)
                az.plot_pair(minimal_idata, marginal=True)
            except Exception:
                # 3. Last-resort fallback: Plot standard pair plot without keyword arguments
                az.plot_pair(minimal_idata)

        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "pair_scale_indeterminacy_check.svg"), bbox_inches="tight")
        plt.close()
        print("  Saved scale-indeterminacy check plot to diagnostics/pair_scale_indeterminacy_check.svg.")
    except Exception as e:
        print(f"  [SKIPPED] Pair plot failed: {e}")



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
            plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "looic_pareto_k_diagnostic.svg"), bbox_inches="tight")
            plt.close()
            print("  Saved LOOIC Pareto k diagnostic plot to diagnostics/looic_pareto_k_diagnostic.svg.")
    except Exception as e:
        print(f"  [SKIPPED] LOOIC calculations skipped: {e}")

    return warnings_found, summary


def generate_cpu_posterior_predictive(idata: xarray.DataTree, model: Any) -> xarray.DataTree:
    """
    Safely compiles and runs PPC on the CPU using a thinned draw subset to prevent VRAM crashes.
    """
    print("\n--- RUNNING CPU-BASED POSTERIOR PREDICTIVE SIMULATIONS ---")
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # print("\n--- RUNNING POSTERIOR PREDICTIVE SIMULATIONS ---")
    # os.environ["JAX_PLATFORM_NAME"] = ""
    # del os.environ["JAX_PLATFORM_NAME"]  # allow using gpu
    import pymc as pm

    # Thin the inference data to PPC_DRAWS across draws to prevent OOM/VRAM crashes
    total_draws = len(idata.posterior.draw)
    step = max(1, total_draws // config.PPC_DRAWS)
    thinned_idata = idata.sel(draw=slice(None, None, step))

    with model:
        idata_ppc = pm.sample_posterior_predictive(
            thinned_idata,
            extend_inferencedata=True,
            random_seed=config.RANDOM_SEED
        )
    return idata_ppc


def plot_ppc_safely(idata: xarray.DataTree) -> None:
    """
    Plots PPC check safely, supporting legacy ArviZ, modern ArviZ 1.x, and arviz_plots.
    """
    # Check if group exists directly using standard dict-like membership check
    if "posterior_predictive" not in idata:
        print("  [SKIPPED] Posterior predictive plot skipped: 'posterior_predictive' group not found in container.")
        return

    try:
        # 1. Try legacy ArviZ plot_ppc (0.x)
        if hasattr(az, "plot_ppc"):
            az.plot_ppc(idata, num_pp_samples=100)
        else:
            # 2. Try modern ArviZ 1.x / arviz_plots plot_ppc_dist (1.x)
            try:
                import arviz_plots as azp
                azp.plot_ppc_dist(idata, num_samples=100)
            except ImportError:
                # 3. If arviz-plots is not imported directly, check if it's nested in az
                if hasattr(az, "plot_ppc_dist"):
                    az.plot_ppc_dist(idata, num_samples=100)
                else:
                    raise AttributeError(
                        "Neither legacy 'plot_ppc' nor modern 'plot_ppc_dist' could be located in your ArviZ environment."
                    )

        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "ppc_plot.svg"), bbox_inches="tight")
        plt.close()
        print("  Saved posterior predictive check plot to diagnostics/ppc_plot.svg.")
    except Exception as e:
        print(f"  [SKIPPED] PPC plot execution failed: {e}")