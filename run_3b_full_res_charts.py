# -*- coding: utf-8 -*-
"""
run_analysis.py

Orchestration pipeline for Part 2: Load serialized inference structures,
execute diagnostic validations, export CSV statistics, and generate plots.
"""

import time
import os
import pickle
import arviz as az
import numpy as np
import pandas as pd
import config
import diagnostics
import plotting


def main():
    total_start = time.time()

    print("--- PART 2 - SECTION 1: LOADING SERIALIZED INFRASTRUCTURE ---")
    sec_start = time.time()

    # Load raw preprocessed tables
    df_raw = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_raw.pkl"))
    df_long = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_long.pkl"))

    # Load structural indexes
    with open(os.path.join(config.MODEL_DIR, "struct_maps.pkl"), "rb") as f:
        struct_maps = pickle.load(f)

    # Load posteriors
    idata = az.from_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data.nc"))

    print("  Artifacts successfully retrieved from disk.")
    print(f"Elapsed time for SECTION 1: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2 - SECTION 2: COMPUTING CONVERGENCE DIAGNOSTICS ---")
    sec_start = time.time()
    warnings, summary = diagnostics.run_diagnostic_checks(idata, df_long, struct_maps)

    # Log warning files directly into diagnostics
    warning_path = os.path.join(config.DIAGNOSTICS_DIR, "diagnostic_warnings.log")
    with open(warning_path, "w") as f:
        if warnings:
            f.write("--- SYSTEM MODEL CONVERGENCE CRITICAL WARNINGS ---\n")
            for w in warnings:
                f.write(f"* {w}\n")
        else:
            f.write("All statistical convergence checks passed stably.\n")

    print(f"  Warning logs saved to '{warning_path}'")
    print(f"Elapsed time for SECTION 2: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2 - SECTION 3: EXPORTING TABULAR PARAMETER ESTIMATES ---")
    sec_start = time.time()
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    hdis = az.hdi(idata, hdi_prob=0.95)

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
    }).to_csv(os.path.join(config.CSV_DIR, "mu_parameter_estimates.csv"), index=False)

    # Export sigma values (fixed to 1 in this model)
    sigma_means = posterior_means["sigma"].values.flatten()
    pd.DataFrame({
        "Section":         sections_list,
        "sigma_mean":      sigma_means,
        "sigma_hdi_lower": np.ones_like(sigma_means),
        "sigma_hdi_upper": np.ones_like(sigma_means)
    }).to_csv(os.path.join(config.CSV_DIR, "sigma_parameter_estimates.csv"), index=False)

    export_df = pd.DataFrame()
    export_df["group_idx"] = np.arange(J)

    reverse_group_map = {v: k for k, v in struct_maps["group_idx_map"].items()}
    export_df["group"] = export_df["group_idx"].map(reverse_group_map)
    export_df[[config.ID_VAR, config.YEAR_COL]] = export_df["group"].str.split("|", expand=True)
    export_df[config.YEAR_COL] = export_df[config.YEAR_COL].astype(int)

    team_to_dept_map = df_raw[[config.ID_VAR, "dept"]].drop_duplicates().set_index(config.ID_VAR)["dept"].to_dict()
    export_df["dept"] = export_df[config.ID_VAR].map(team_to_dept_map)

    # Extract Section Z-Scores
    sec_means = posterior_means["theta_sec"].values
    sec_hdis = hdis["theta_sec"].values
    for s_idx, sec in enumerate(sections_list):
        export_df[f"section_{sec}"] = sec_means[:, s_idx]
        export_df[f"section_{sec}_hdi_lower"] = sec_hdis[:, s_idx, 0]
        export_df[f"section_{sec}_hdi_upper"] = sec_hdis[:, s_idx, 1]

    # Extract Category Z-Scores
    cat_means = posterior_means["theta_cat"].values
    cat_hdis = hdis["theta_cat"].values
    for c_idx, cat in enumerate(categories_list):
        export_df[f"category_{cat}"] = cat_means[:, c_idx]
        export_df[f"category_{cat}_hdi_lower"] = cat_hdis[:, c_idx, 0]
        export_df[f"category_{cat}_hdi_upper"] = cat_hdis[:, c_idx, 1]

    export_df.to_csv(os.path.join(config.CSV_DIR, "hierarchical_team_capability_estimates.csv"), index=False)

    # Export Section-level Correlations (Omega)
    Omega_mean = posterior_means["Omega"].values
    corr_df = pd.DataFrame(Omega_mean, index=sections_list, columns=sections_list)
    corr_df.to_csv(os.path.join(config.CSV_DIR, "capability_correlations.csv"))

    print("  tabular estimates saved inside: estimates_csv/")
    print(f"Elapsed time for SECTION 3: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2 - SECTION 4: RENDERING CAPABILITY VISUALIZATIONS ---")
    sec_start = time.time()
    sim_years = sorted(export_df[config.YEAR_COL].unique())

    # Generate Section slope and ridge plots
    for s_idx, sec in enumerate(sections_list):
        plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP)
        plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP,
                                               target_dept="Engineering")
        plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx)
        plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx, target_dept="Engineering")

    # Generate Category slope and ridge plots
    for c_idx, cat in enumerate(categories_list):
        plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP)
        plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP,
                                               target_dept="Engineering")
        plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx)
        plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx, target_dept="Engineering")

    # Clustermaps
    plotting.plot_omega_clustermap(corr_df)

    # Stacked Likert Response distributions
    for sec_name, cats in config.SURVEY_HIERARCHY.items():
        for cat, qs in cats.items():
            plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=sim_years[-1],
                                                        target_dept=None)
            plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=sim_years[-1],
                                                        target_dept="Engineering")

    print("  Structural and capability graphs saved inside: plots/")
    print(f"Elapsed time for SECTION 4: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2 - SECTION 5: RENDERING ITEM DIAGNOSTICS & TRACE CHECKS ---")
    sec_start = time.time()

    # Question-level CRF, Predictive Checks, IIF plots
    for q_name in struct_maps["questions"]:
        plotting.plot_category_response_functions(q_name, idata, struct_maps)
        plotting.plot_predicted_vs_empirical_dist(q_name, idata, df_long, struct_maps)
        plotting.plot_item_information_function(q_name, idata, struct_maps)

    # Test information curves (TIF)
    plotting.plot_test_information_function(idata, len(struct_maps["questions"]))

    # Save diagnostic tracing plots directly into diagnostics
    try:
        import matplotlib.pyplot as plt
        az.plot_trace(idata, var_names=["a"])
        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "trace_plots.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] Trace Graph failed: {e}")

    try:
        import matplotlib.pyplot as plt
        az.plot_ppc(idata, num_pp_samples=100)
        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "ppc_plot.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] PPC plot failed: {e}")

    print("  Trace, fit validation, and TIF graphs saved inside: diagnostics/ and plots/")
    print(f"Elapsed time for SECTION 5: {time.time() - sec_start:.2f} seconds")

    print(f"\n=======================================================")
    print(f"Part 2 Analysis Completed in {time.time() - total_start:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()