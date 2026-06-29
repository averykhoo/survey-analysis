# -*- coding: utf-8 -*-
"""
run_3b_full_res_charts.py

Orchestration pipeline for Part 3b: Generate high-resolution analytical plots
(using full available draws) inside the standard 'plots' subdirectory.
"""

import os
import pickle
import time

import arviz as az
import numpy as np
import pandas as pd

import config
import plotting


def main():
    total_start = time.time()

    print("--- PART 3b: RUNNING HIGH-RESOLUTION ANALYTICAL PLOTTING ---")
    sec_start = time.time()

    df_raw = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_raw.pkl"))
    df_long = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_long.pkl"))

    with open(os.path.join(config.MODEL_DIR, "struct_maps.pkl"), "rb") as f:
        struct_maps = pickle.load(f)

    idata = az.from_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data.nc"))
    print(f"Elapsed time for LOADING: {time.time() - sec_start:.2f} seconds")

    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    hdis = az.hdi(idata, prob=0.95)

    sections_list = struct_maps["sections"]
    categories_list = struct_maps["categories"]
    J = len(struct_maps["group_idx_map"])

    export_df = pd.DataFrame()
    export_df["group_idx"] = np.arange(J)
    reverse_group_map = {v: k for k, v in struct_maps["group_idx_map"].items()}
    export_df["group"] = export_df["group_idx"].map(reverse_group_map)
    export_df[[config.ID_VAR, config.YEAR_COL]] = export_df["group"].str.split("|", expand=True)
    export_df[config.YEAR_COL] = export_df[config.YEAR_COL].astype(int)

    # Sort raw data by year first to ensure the latest (2026) department assignment takes priority
    df_raw_sorted = df_raw.sort_values(by=config.YEAR_COL)
    team_to_dept_map = (df_raw_sorted
                        .drop_duplicates(subset=[config.ID_VAR], keep="last")
                        .set_index(config.ID_VAR)["dept"].to_dict())
    export_df["dept"] = export_df[config.ID_VAR].map(team_to_dept_map)

    for s_idx, sec in enumerate(sections_list):
        print(f'{sec=}')
        export_df[f"section_{sec}"] = posterior_means["theta_sec"].values[:, s_idx]
        export_df[f"section_{sec}_hdi_lower"] = hdis["theta_sec"].values[:, s_idx, 0]
        export_df[f"section_{sec}_hdi_upper"] = hdis["theta_sec"].values[:, s_idx, 1]

    for c_idx, cat in enumerate(categories_list):
        print(f'{cat=}')
        export_df[f"category_{cat}"] = posterior_means["theta_cat"].values[:, c_idx]
        export_df[f"category_{cat}_hdi_lower"] = hdis["theta_cat"].values[:, c_idx, 0]
        export_df[f"category_{cat}_hdi_upper"] = hdis["theta_cat"].values[:, c_idx, 1]

    export_df.to_csv(os.path.join(config.CSV_DIR, "hierarchical_team_capability_estimates.csv"), index=False)

    # Dynamic Current-Year Department Selection
    sim_years = sorted(export_df[config.YEAR_COL].unique())
    latest_year = sim_years[-1]
    depts = [None] + export_df[export_df[config.YEAR_COL] == latest_year]["dept"].dropna().unique().tolist()

    print(f"\n--- LOOPING OVER CURRENT ERA DEPARTMENTS: {depts} ---")
    for d in depts:
        print(f"  Rendering high-res plots for lineage of department: {d}")

        for s_idx, sec in enumerate(sections_list):
            plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP,
                                                   config.PLOTS_DIR, target_dept=d)
            plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx, draws_to_use=6000,
                                                   output_dir=config.PLOTS_DIR, target_dept=d)

        for c_idx, cat in enumerate(categories_list):
            plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP,
                                                   config.PLOTS_DIR, target_dept=d)
            plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx, draws_to_use=6000,
                                                   output_dir=config.PLOTS_DIR, target_dept=d)

        for sec_name, cats in config.SURVEY_HIERARCHY.items():
            for cat, qs in cats.items():
                plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=latest_year,
                                                            output_dir=config.PLOTS_DIR, target_dept=d)

    plotting.plot_omega_clustermap(
        pd.DataFrame(posterior_means["Omega"].values, index=sections_list, columns=sections_list),
        config.PLOTS_DIR)

    for q_name in struct_maps["questions"]:
        plotting.plot_category_response_functions(q_name, idata, struct_maps, config.PLOTS_DIR)
        plotting.plot_predicted_vs_empirical_dist(q_name, idata, df_long, struct_maps, config.PLOTS_DIR)
        plotting.plot_item_information_function(q_name, idata, struct_maps, config.PLOTS_DIR)

    plotting.plot_test_information_function(idata, len(struct_maps["questions"]), config.PLOTS_DIR)

    print(f"\n=======================================================")
    print(f"Part 3b Full Resolution Charts Completed in {time.time() - total_start:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
