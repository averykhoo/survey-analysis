# -*- coding: utf-8 -*-
"""
run_3a_fast_charts.py

Orchestration pipeline for Part 3a: Generate fast, low-resolution analytical plots
(thinned down to 300 draws) inside the 'lowres_plots' subdirectory.
"""

import time
import os
import pickle
import arviz as az
import pandas as pd
import numpy as np
import config
import plotting


def main():
    total_start = time.time()

    print("--- PART 3a: RUNNING COARSE ANALYTICAL PLOTTING (FAST) ---")
    sec_start = time.time()

    df_raw = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_raw.pkl"))
    df_long = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_long.pkl"))

    with open(os.path.join(config.MODEL_DIR, "struct_maps.pkl"), "rb") as f:
        struct_maps = pickle.load(f)

    idata = az.from_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data.nc"))
    print(f"Elapsed time for LOADING: {time.time() - sec_start:.2f} seconds")

    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    hdis = az.hdi(idata, hdi_prob=0.95)

    sections_list = struct_maps["sections"]
    categories_list = struct_maps["categories"]
    J = len(struct_maps["group_idx_map"])

    export_df = pd.DataFrame()
    export_df["group_idx"] = np.arange(J)
    reverse_group_map = {v: k for k, v in struct_maps["group_idx_map"].items()}
    export_df["group"] = export_df["group_idx"].map(reverse_group_map)
    export_df[[config.ID_VAR, config.YEAR_COL]] = export_df["group"].str.split("|", expand=True)
    export_df[config.YEAR_COL] = export_df[config.YEAR_COL].astype(int)

    team_to_dept_map = df_raw[[config.ID_VAR, "dept"]].drop_duplicates().set_index(config.ID_VAR)["dept"].to_dict()
    export_df["dept"] = export_df[config.ID_VAR].map(team_to_dept_map)

    for s_idx, sec in enumerate(sections_list):
        export_df[f"section_{sec}"] = posterior_means["theta_sec"].values[:, s_idx]
        export_df[f"section_{sec}_hdi_lower"] = hdis["theta_sec"].values[:, s_idx, 0]
        export_df[f"section_{sec}_hdi_upper"] = hdis["theta_sec"].values[:, s_idx, 1]

    for c_idx, cat in enumerate(categories_list):
        export_df[f"category_{cat}"] = posterior_means["theta_cat"].values[:, c_idx]
        export_df[f"category_{cat}_hdi_lower"] = hdis["theta_cat"].values[:, c_idx, 0]
        export_df[f"category_{cat}_hdi_upper"] = hdis["theta_cat"].values[:, c_idx, 1]

    depts = df_raw["dept"].dropna().unique().tolist() + [None]
    sim_years = sorted(export_df[config.YEAR_COL].unique())

    print(f"\n--- LOOPING OVER DEPARTMENTS: {depts} ---")
    for d in depts:
        print(f"  Rendering fast plots for department: {d}")

        for s_idx, sec in enumerate(sections_list):
            plotting.plot_slope_chart_hierarchical(export_df, sec, "section", sim_years, config.REORG_LINEAGE_MAP,
                                                   config.LOWRES_PLOTS_DIR, target_dept=d)
            plotting.plot_ridge_plots_hierarchical(idata, export_df, sec, "section", s_idx, draws_to_use=300,
                                                   output_dir=config.LOWRES_PLOTS_DIR, target_dept=d)

        for c_idx, cat in enumerate(categories_list):
            plotting.plot_slope_chart_hierarchical(export_df, cat, "category", sim_years, config.REORG_LINEAGE_MAP,
                                                   config.LOWRES_PLOTS_DIR, target_dept=d)
            plotting.plot_ridge_plots_hierarchical(idata, export_df, cat, "category", c_idx, draws_to_use=300,
                                                   output_dir=config.LOWRES_PLOTS_DIR, target_dept=d)

        for sec_name, cats in config.SURVEY_HIERARCHY.items():
            for cat, qs in cats.items():
                plotting.plot_likert_response_distributions(df_raw, export_df, cat, qs, target_year=sim_years[-1],
                                                            output_dir=config.LOWRES_PLOTS_DIR, target_dept=d)

    plotting.plot_omega_clustermap(
        pd.DataFrame(posterior_means["Omega"].values, index=sections_list, columns=sections_list),
        config.LOWRES_PLOTS_DIR)

    for q_name in struct_maps["questions"]:
        plotting.plot_category_response_functions(q_name, idata, struct_maps, config.LOWRES_PLOTS_DIR)
        plotting.plot_predicted_vs_empirical_dist(q_name, idata, df_long, struct_maps, config.LOWRES_PLOTS_DIR)
        plotting.plot_item_information_function(q_name, idata, struct_maps, config.LOWRES_PLOTS_DIR)

    plotting.plot_test_information_function(idata, len(struct_maps["questions"]), config.LOWRES_PLOTS_DIR)

    print(f"\n=======================================================")
    print(f"Part 3a Fast Charts Completed in {time.time() - total_start:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()