# -*- coding: utf-8 -*-
"""
main.py

Hierarchical Orchestration Pipeline. Generates hierarchical simulated data,
runs the H-MGRM, validates and exports dual-level Z-Scores, and saves plots.
Provides execution timing feedback block-by-block.

1. Capability Interpretation: Model scores (theta) and Z-scores represent relative team capabilities.
   Compare relative values across teams and focus on individual performance improvement. Standardized values
   represent performance deviations relative to the entire company population.
2. Focused Reorganizations: Linear transitions are mapped using slope charts. Track standard offsets
   to evaluate split/merged team performance relative to parent groups.
3. Item Validation: CRF, IIF, and empirical fit graphs provide item diagnostics. Identify low discrimination
   items (a < 0.3) for potential adjustment in future survey cycles.
4. Target precision mapping: The Test Information Function (TIF) maps region bounds where capability measurements
   are most precise (lowest SEM).
5. Compositional items (e.g., how time is spent) should not be averaged directly. Run separate evaluations
   to avoid bias.
6. Ensure that subjective capability ranks are regularly compared to objective delivery metrics to validate self-assessments.
"""

import os
import time

import arviz as az
import numpy as np
import pandas as pd

import config
import data_utils
import diagnostics
import model_pymc
import plotting


def main():
    total_start_time = time.time()

    print("--- SECTION 1: INITIATING SURVEY HIERARCHY ---")
    sec_start = time.time()

    sim_years = [2023, 2025, 2026]
    df_raw, true_params = data_utils.simulate_dora_data_reorg(
        sim_years_reorg=sim_years,
        lineage_map=config.REORG_LINEAGE_MAP,
        sim_n_resp_per_team_year=25,
        hierarchy_map=config.SURVEY_HIERARCHY,
        response_options_sim=config.RESPONSE_OPTIONS
    )
    print(f"Elapsed time for SECTION 1: {time.time() - sec_start:.2f} seconds")

    print("\n--- SECTION 2: PARSING STRUCTURAL LAYERS ---")
    sec_start = time.time()
    df_long, struct_maps = data_utils.load_and_preprocess_data(df_raw, config.SURVEY_HIERARCHY)
    print(f"Elapsed time for SECTION 2: {time.time() - sec_start:.2f} seconds")

    print("\n--- SECTION 3 & 4: SAMPLING HIERARCHICAL MODEL ---")
    sec_start = time.time()
    idata, model = model_pymc.build_and_run_model(df_long, struct_maps)
    print(f"Elapsed time for SECTION 3/4: {time.time() - sec_start:.2f} seconds")

    print("\n--- SECTION 5: EXECUTING CONVERGENCE DIAGNOSTICS ---")
    sec_start = time.time()
    warnings, summary = diagnostics.run_diagnostic_checks(idata, df_long, struct_maps)

    if warnings:
        print("\n  *** DIAGNOSTIC WARNING REPORT ***")
        for warning in warnings:
            print(f"    * {warning}")
    else:
        print("  All statistical convergence checks passed stably.")
    print(f"Elapsed time for SECTION 5: {time.time() - sec_start:.2f} seconds")

    print("\n--- SECTION 6: PROCESSING PARAMETER EXPORT FILES ---")
    sec_start = time.time()
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    hdis = az.hdi(idata, prob=0.95)

    sections_list = struct_maps["sections"]
    categories_list = struct_maps["categories"]
    J = len(struct_maps["group_idx_map"])

    # Export mu values (fixed to 0 in this model)
    mu_means = posterior_means["mu"].values.flatten()
    pd.DataFrame({
        "Section":      sections_list,
        "mu_mean":      mu_means,
        "mu_hdi_lower": np.zeros_like(mu_means),
        "mu_hdi_upper": np.zeros_like(mu_means)
    }).to_csv(os.path.join(config.OUTPUT_DIR, "mu_parameter_estimates.csv"), index=False)

    # Export sigma values (fixed to 1 in this model)
    sigma_means = posterior_means["sigma"].values.flatten()
    pd.DataFrame({
        "Section":         sections_list,
        "sigma_mean":      sigma_means,
        "sigma_hdi_lower": np.ones_like(sigma_means),
        "sigma_hdi_upper": np.ones_like(sigma_means)
    }).to_csv(os.path.join(config.OUTPUT_DIR, "sigma_parameter_estimates.csv"), index=False)

    export_df = pd.DataFrame()
    export_df["group_idx"] = np.arange(J)

    reverse_group_map = {v: k for k, v in struct_maps["group_idx_map"].items()}
    export_df["group"] = export_df["group_idx"].map(reverse_group_map)
    export_df[[config.ID_VAR, config.YEAR_COL]] = export_df["group"].str.split("|", expand=True)
    export_df[config.YEAR_COL] = export_df[config.YEAR_COL].astype(int)

    team_to_dept_map = df_raw[[config.ID_VAR, "dept"]].drop_duplicates().set_index(config.ID_VAR)["dept"].to_dict()
    export_df["dept"] = export_df[config.ID_VAR].map(team_to_dept_map)

    # Section Z-Scores
    sec_means = posterior_means["theta_sec"].values
    sec_hdis = hdis["theta_sec"].values
    for s_idx, sec in enumerate(sections_list):
        export_df[f"section_{sec}"] = sec_means[:, s_idx]
        export_df[f"section_{sec}_hdi_lower"] = sec_hdis[:, s_idx, 0]
        export_df[f"section_{sec}_hdi_upper"] = sec_hdis[:, s_idx, 1]

    # Category Z-Scores
    cat_means = posterior_means["theta_cat"].values
    cat_hdis = hdis["theta_cat"].values
    for c_idx, cat in enumerate(categories_list):
        export_df[f"category_{cat}"] = cat_means[:, c_idx]
        export_df[f"category_{cat}_hdi_lower"] = cat_hdis[:, c_idx, 0]
        export_df[f"category_{cat}_hdi_upper"] = cat_hdis[:, c_idx, 1]

    export_df.to_csv(os.path.join(config.OUTPUT_DIR, "hierarchical_team_capability_estimates.csv"), index=False)
    print("  Hierarchical results saved to 'hierarchical_team_capability_estimates.csv'.")

    # Export Section-level Correlations (Omega)
    Omega_mean = posterior_means["Omega"].values
    corr_df = pd.DataFrame(Omega_mean, index=sections_list, columns=sections_list)
    corr_df.to_csv(os.path.join(config.OUTPUT_DIR, "capability_correlations.csv"))
    print(f"Elapsed time for SECTION 6: {time.time() - sec_start:.2f} seconds")

    print("\n--- SECTION 7: RENDERING HIERARCHICAL VISUALIZATIONS ---")
    sec_start = time.time()

    # Hierarchical Slope and Ridge Curve Rendering
    for s_idx, sec in enumerate(sections_list):
        plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP)
        plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP,
                                               target_dept="Engineering")
        plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx)
        plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx, target_dept="Engineering")

    for c_idx, cat in enumerate(categories_list):
        plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP)
        plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP,
                                               target_dept="Engineering")
        plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx)
        plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx, target_dept="Engineering")

    # Global Hierarchical Clustermap
    plotting.plot_omega_clustermap(corr_df)

    # Likert Stacked Response Distributions (Sorted by Category Capabilities)
    for sec_name, cats in config.SURVEY_HIERARCHY.items():
        for cat, qs in cats.items():
            plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=sim_years[-1],
                                                        target_dept=None)
            plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=sim_years[-1],
                                                        target_dept="Engineering")

    # Item IRT Functions & Empirical Predictive Checks (PPC)
    for q_name in struct_maps["questions"]:
        plotting.plot_category_response_functions(q_name, idata, struct_maps)
        plotting.plot_predicted_vs_empirical_dist(q_name, idata, df_long, struct_maps)
        plotting.plot_item_information_function(q_name, idata, struct_maps)

    # Global Test Information Function (TIF) and SEM curves
    plotting.plot_test_information_function(idata, len(struct_maps["questions"]))

    # MCMC Parameter caterpillar traces and overall PPC density checks
    try:
        import matplotlib.pyplot as plt
        az.plot_trace(idata, var_names=["a"])
        plt.savefig(os.path.join(config.OUTPUT_DIR, "trace_plots.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] Trace Graph failed: {e}")

    try:
        import matplotlib.pyplot as plt
        az.plot_ppc(idata, num_pp_samples=100)
        plt.savefig(os.path.join(config.OUTPUT_DIR, "ppc_plot.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] PPC plot failed: {e}")

    print(f"Elapsed time for SECTION 7: {time.time() - sec_start:.2f} seconds")

    print(f"\n=======================================================")
    print(f"Complete Pipeline Run executed in {time.time() - total_start_time:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
