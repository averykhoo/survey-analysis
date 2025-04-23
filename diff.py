
# -----------------------------------------
# Section 6: Results Extraction & Processing
# -----------------------------------------
# --- Correlation Matrix Estimate (Omega) ---

# --- Mu and Sigma Means (for plotting) ---
mu_map_for_plot = {}; sigma_map_for_plot = {}; cat_names_list_plot = []
if 'mu' in samples and 'sigma' in samples:
    mu_means_plot = samples['mu'].mean(axis=0); sigma_means_plot = samples['sigma'].mean(axis=0)
    cat_names_list_plot = [cat_idx_to_name_final[i] for i in range(1, L_final + 1)]
    mu_map_for_plot = dict(zip(cat_names_list_plot, mu_means_plot))
    sigma_map_for_plot = dict(zip(cat_names_list_plot, sigma_means_plot))
    print("\nCalculated posterior means for mu and sigma for plotting.")
else: print("WARNING: 'mu' or 'sigma' not found. Cannot create maps for standardized plots.")
# --- Ranking and Improvement Summary ---




