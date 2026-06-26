# -*- coding: utf-8 -*-
"""
run_sampling.py

Orchestration pipeline for Part 1: Preprocessing, PyMC compilation, sampling,
posterior predictive simulation, and saving everything to disk.
"""

import time
import os
import pickle
import config
import data_utils
import model_pymc


def main():
    total_start = time.time()

    print("--- PART 1 - SECTION 1: INITIATING SURVEY DATA GENERATION ---")
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

    print("\n--- PART 1 - SECTION 2: PARSING STRUCTURAL LAYERS & INDEX MAPPINGS ---")
    sec_start = time.time()
    df_long, struct_maps = data_utils.load_and_preprocess_data(df_raw, config.SURVEY_HIERARCHY)
    print(f"Elapsed time for SECTION 2: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 1 - SECTION 3: EXECUTING MCMC SAMPLING & PPC ---")
    sec_start = time.time()
    idata, model = model_pymc.build_and_run_model(df_long, struct_maps)
    print(f"Elapsed time for SECTION 3: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 1 - SECTION 4: SERIALIZING OUTPUT FILES TO DISK ---")
    sec_start = time.time()

    # Save InferenceData object natively (completely preserves chains and coords)
    idata.to_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data.nc"))

    # Save preprocessing DataFrames
    df_raw.to_pickle(os.path.join(config.MODEL_DIR, "df_raw.pkl"))
    df_long.to_pickle(os.path.join(config.MODEL_DIR, "df_long.pkl"))

    # Save structural maps
    with open(os.path.join(config.MODEL_DIR, "struct_maps.pkl"), "wb") as f:
        pickle.dump(struct_maps, f)

    print("  Sampling trace saved. Exiting process to clear VRAM pool.")
    print(f"Elapsed time for SECTION 4: {time.time() - sec_start:.2f} seconds")

    print(f"\n=======================================================")
    print(f"Part 1 Sampling Completed in {time.time() - total_start:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()