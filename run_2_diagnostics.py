# -*- coding: utf-8 -*-
"""
run_2_diagnostics.py

Orchestration pipeline for Part 2: Load serialized inference structures,
rebuild model context, run thinned posterior predictive checks on the CPU,
execute diagnostics, and generate diagnostic plots.
"""

import time
import os
import pickle
import pandas as pd
import arviz as az
import config
import diagnostics
import model_pymc


def main():
    total_start = time.time()

    print("--- PART 2, SECTION 1: LOADING TRACE FOR FIT DIAGNOSTICS ---")
    sec_start = time.time()

    df_long = pd.read_pickle(os.path.join(config.MODEL_DIR, "df_long.pkl"))
    with open(os.path.join(config.MODEL_DIR, "struct_maps.pkl"), "rb") as f:
        struct_maps = pickle.load(f)

    # Load posteriors, pull them fully into memory, and release the disk file lock
    idata = az.from_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data.nc"))
    idata.load()
    idata.close()
    print(f"Elapsed time for SECTION 1: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2, SECTION 2: COMPILING THINNED CPU POSTERIOR PREDICTIVE ---")
    sec_start = time.time()

    _, dummy_model = model_pymc.build_and_run_model(df_long, struct_maps, run_sampling=False)
    idata_diagnostic = diagnostics.generate_cpu_posterior_predictive(idata, dummy_model)
    # Save the diagnostic version to a separate file so it doesn't overwrite your 1,500-draw master trace!
    idata_diagnostic.to_netcdf(os.path.join(config.MODEL_DIR, "dora_inference_data_diagnostic.nc"))
    print(f"Elapsed time for SECTION 2: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2, SECTION 3: COMPUTING CONVERGENCE DIAGNOSTICS ---")
    sec_start = time.time()
    warnings, summary = diagnostics.run_diagnostic_checks(idata, df_long, struct_maps)

    warning_path = os.path.join(config.DIAGNOSTICS_DIR, "diagnostic_warnings.log")
    with open(warning_path, "w") as f:
        if warnings:
            f.write("--- SYSTEM MODEL CONVERGENCE CRITICAL WARNINGS ---\n")
            for w in warnings:
                f.write(f"* {w}\n")
        else:
            f.write("All statistical convergence checks passed stably.\n")

    print(f"  Warning logs saved to '{warning_path}'")
    print(f"Elapsed time for SECTION 3: {time.time() - sec_start:.2f} seconds")

    print("\n--- PART 2, SECTION 4: RENDERING STATISTICAL DIAGNOSTIC PLOTS ---")
    sec_start = time.time()
    diagnostics.plot_ppc_safely(idata)

    try:
        import matplotlib.pyplot as plt
        # 1. Raise ArviZ's subplot limit to protect other diagnostics from crashing
        az.rcParams["plot.max_subplots"] = 200

        # 2. Compress the multidimensional 'a' trace onto a single row of axes
        az.plot_trace(idata, var_names=["a"])

        plt.savefig(os.path.join(config.DIAGNOSTICS_DIR, "trace_plots.png"), bbox_inches="tight")
        plt.close()
        print("  Saved trace plots to diagnostics/trace_plots.png.")
    except Exception as e:
        print(f"  [SKIPPED] Trace Graph failed: {e}")

    print(f"Elapsed time for SECTION 4: {time.time() - sec_start:.2f} seconds")

    print(f"\n=======================================================")
    print(f"Part 2 Diagnostics Completed in {time.time() - total_start:.1f} total seconds.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()