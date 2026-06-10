# -*- coding: utf-8 -*-
"""
main.py

The main orchestration module. Initializes simulations or reads files,
compiles parameters, initiates standard diagnostics, generates plots,
and prints overall interpretive insights [3].
"""

import datetime
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_utils
import diagnostics
import model_pymc
import plotting


def main():
    t_start = datetime.datetime.now()
    print("--- SECTION 1: SYSTEM INITIATION ---")

    # Run Simulation Generation (Toggle for CSV input)
    RUN_SIMULATION = True

    if RUN_SIMULATION:
        sim_years = [2023, 2025, 2026]
        df_raw, true_params = data_utils.simulate_dora_data_reorg(
            sim_years_reorg=sim_years,
            reorg_map_child_to_parents=config.REORG_MAPPING_Y_LAST_TO_Y1,
            sim_n_resp_per_team_year=25,
            category_map_sim=config.CATEGORY_MAPPING_INITIAL,
            response_options_sim=config.RESPONSE_OPTIONS
        )
    else:
        # Load CSV file if running on external datasets
        # df_raw = pd.read_csv("your_dora_survey_data.csv")
        pass

    # Preprocessing
    print("\n--- SECTION 2: EXECUTING DATA PREPROCESSING ---")
    df_long, category_mapping_final, cat_idx_to_name, question_idx_map, question_categories = data_utils.load_and_preprocess_data(
        df_raw)

    K = len(question_idx_map)
    L = len(category_mapping_final)
    J = len(df_long["group_idx"].unique())
    print(f"Data counts completed. Items K={K}, Dimensions L={L}, Groups J={J}")

    # Section 3/4: Run Model
    print("\n--- SECTION 3 & 4: COMPILING MODEL AND DRAWS ---")
    idata, model = model_pymc.build_and_run_model(df_long, question_idx_map, question_categories)

    # Section 5: Diagnostics
    warnings, summary = diagnostics.run_diagnostic_checks(idata, df_long, cat_idx_to_name, question_idx_map)

    # Section 6: Process and Export Parameters
    print("\n--- SECTION 6: PROCESSING PARAMETER EXPORT FILES ---")
    hdis = az.hdi(idata, hdi_prob=0.95)
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    category_names_final = list(cat_idx_to_name.values())

    # Export mu values
    mu_means = posterior_means["mu"].values.flatten()
    mu_lower = hdis["mu"].values[:, 0, 0]
    mu_upper = hdis["mu"].values[:, 0, 1]
    pd.DataFrame({
        "Category":     category_names_final,
        "mu_mean":      mu_means,
        "mu_hdi_lower": mu_lower,
        "mu_hdi_upper": mu_upper
    }).to_csv(os.path.join(config.OUTPUT_DIR, "mu_parameter_estimates.csv"), index=False)

    # Export sigma values
    sigma_means = posterior_means["sigma"].values.flatten()
    sigma_lower = hdis["sigma"].values[:, 0]
    sigma_upper = hdis["sigma"].values[:, 1]
    pd.DataFrame({
        "Category":        category_names_final,
        "sigma_mean":      sigma_means,
        "sigma_hdi_lower": sigma_lower,
        "sigma_hdi_upper": sigma_upper
    }).to_csv(os.path.join(config.OUTPUT_DIR, "sigma_parameter_estimates.csv"), index=False)

    # Export capability dimension tables
    theta_means = posterior_means["theta"].values
    theta_hdis = hdis["theta"].values
    theta_z_means = posterior_means["theta_z"].values
    theta_z_hdis = hdis["theta_z"].values

    theta_cols = [f"theta_{cat}" for cat in category_names_final]
    theta_df = pd.DataFrame(theta_means, columns=theta_cols)

    for idx, cat in enumerate(category_names_final):
        theta_df[f"theta_{cat}_hdi_lower"] = theta_hdis[:, idx, 0]
        theta_df[f"theta_{cat}_hdi_upper"] = theta_hdis[:, idx, 1]
        theta_df[f"theta_{cat}_z"] = theta_z_means[:, idx]
        theta_df[f"theta_{cat}_z_hdi_lower"] = theta_z_hdis[:, idx, 0]
        theta_df[f"theta_{cat}_z_hdi_upper"] = theta_z_hdis[:, idx, 1]

    # Map groups back
    reverse_group_map = {v: k for k, v in df_long.groupby("group")["group_idx"].first().to_dict().items()}
    theta_df["group_idx"] = np.arange(J)
    theta_df["group"] = theta_df["group_idx"].map(reverse_group_map)
    theta_df[[config.ID_VAR, config.YEAR_COL]] = theta_df["group"].str.split("|", expand=True)
    theta_df[config.YEAR_COL] = theta_df[config.YEAR_COL].astype(int)

    # Apply department metadata mappings
    team_to_dept_map = df_raw[[config.ID_VAR, "dept"]].drop_duplicates().set_index(config.ID_VAR)["dept"].to_dict()
    theta_df["dept"] = theta_df[config.ID_VAR].map(team_to_dept_map)

    # Re-order columns safely
    cap_order = [config.ID_VAR, config.YEAR_COL, "dept", "group", "group_idx"] + theta_cols
    for cat in category_names_final:
        cap_order.extend([
            f"theta_{cat}_hdi_lower", f"theta_{cat}_hdi_upper",
            f"theta_{cat}_z", f"theta_{cat}_z_hdi_lower", f"theta_{cat}_z_hdi_upper"
        ])
    theta_df = theta_df[cap_order]

    # Mask capability categories that had no responses for that group (e.g. expansion years)
    for group in theta_df["group"]:
        team, yr = group.split("|")
        yr = int(yr)
        for cat in category_names_final:
            questions_in_cat = category_mapping_final.get(cat, [])
            subset_obs = df_raw[(df_raw[config.ID_VAR] == team) & (df_raw[config.YEAR_COL] == yr)][questions_in_cat]
            if subset_obs.dropna(how="all").empty:
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}"] = np.nan
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}_hdi_lower"] = np.nan
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}_hdi_upper"] = np.nan
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}_z"] = np.nan
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}_z_hdi_lower"] = np.nan
                theta_df.loc[theta_df["group"] == group, f"theta_{cat}_z_hdi_upper"] = np.nan

    theta_df.to_csv(os.path.join(config.OUTPUT_DIR, "team_capability_estimates.csv"), index=False)

    # Export capability covariance correlations
    Omega_mean = posterior_means["Omega"].values
    corr_df = pd.DataFrame(Omega_mean, index=category_names_final, columns=category_names_final)
    corr_df.to_csv(os.path.join(config.OUTPUT_DIR, "capability_correlations.csv"))

    # Section 7: Generate Visualizations
    print("\n--- SECTION 7: RENDERING SYSTEM VISUALIZATIONS ---")
    mu_map = dict(zip(category_names_final, mu_means))
    sigma_map = dict(zip(category_names_final, sigma_means))
    years_to_evaluate = sorted(df_raw[config.YEAR_COL].unique())

    # Build lineage maps for plotting
    parent_to_child_map = {}
    all_parents = {p for parents in config.REORG_MAPPING_Y_LAST_TO_Y1.values() for p in parents}
    for parent in all_parents:
        parent_to_child_map[parent] = [child for child, parents in config.REORG_MAPPING_Y_LAST_TO_Y1.items() if
                                       parent in parents]
    for child, parents in config.REORG_MAPPING_Y_LAST_TO_Y1.items():
        if len(parents) == 1 and parents[0] == child and child not in parent_to_child_map:
            parent_to_child_map[child] = [child]

    # Generate graphs
    for cat in category_names_final:
        plotting.plot_reorg_slope_chart_standardized_internal(
            theta_df, cat, years_to_evaluate, mu_map, sigma_map, parent_to_child_map, target_dept=None
        )
        plotting.plot_reorg_slope_chart_standardized_internal(
            theta_df, cat, years_to_evaluate, mu_map, sigma_map, parent_to_child_map, target_dept="Engineering"
        )
        plotting.plot_likert_response_distributions(
            df_raw, theta_df, category_mapping_final, target_year=years_to_evaluate[-1], target_dept=None
        )
        plotting.plot_team_ridge_plots(
            idata, theta_df, cat, target_dept=None
        )
        plotting.plot_team_ridge_plots(
            idata, theta_df, cat, target_dept="Engineering"
        )

        for q in category_mapping_final[cat]:
            plotting.plot_category_response_functions(q, idata, question_idx_map)
            plotting.plot_predicted_vs_empirical_dist(q, idata, df_long, question_idx_map, question_categories)
            plotting.plot_item_information_function(q, idata, question_idx_map)

    plotting.plot_omega_clustermap(corr_df)
    plotting.plot_test_information_function(idata, K)

    # Diagnostic Traces and PPCs
    try:
        az.plot_ppc(idata, num_pp_samples=100)
        plt.savefig(os.path.join(config.OUTPUT_DIR, "ppc_plot.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] PPC Graph failed: {e}")

    try:
        az.plot_trace(idata, var_names=["mu", "sigma", "a"])
        plt.savefig(os.path.join(config.OUTPUT_DIR, "trace_plots.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [SKIPPED] Trace Graph failed: {e}")

    # Ground Truth Parameter Recovery Check (Simulation validation)
    if RUN_SIMULATION:
        print("\n--- GROUND TRUTH COHESION REPORT (PARAMETER RECOVERY) ---")
        mu_mae = np.mean(np.abs(mu_means - true_params["true_mu"]))
        sigma_mae = np.mean(np.abs(sigma_means - true_params["true_sigma"]))

        true_a_vals = np.array([true_params["true_a"][q] for q in question_idx_map.keys()])
        a_mae = np.mean(np.abs(posterior_means["a"].values - true_a_vals))

        print(f"  mu (Global Capability Location) MAE: {mu_mae:.4f}")
        print(f"  sigma (Global Capability Variability) MAE: {sigma_mae:.4f}")
        print(f"  a (Item Discrimination) MAE: {a_mae:.4f}")

    # Output Diagnostic Alerts
    if warnings:
        print("\n--- DIAGNOSTIC WARNING LOGS ---")
        for alert in warnings:
            print(f"  * {alert}")
    else:
        print("\nNo diagnostics issues flagged. Model convergence appears stable.")

    # Section 9: DORA Considerations and Actionability [3]
    print("\n--- SECTION 9: DORA SURVEY-SPECIFIC CONSIDERATIONS ---")
    print("""
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
    """)

    t_end = datetime.datetime.now()
    print(f"\nAnalysis complete. Run executed in {(t_end - t_start).total_seconds():.1f} seconds.")


if __name__ == "__main__":
    main()
