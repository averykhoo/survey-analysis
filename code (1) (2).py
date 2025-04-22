# -----------------------------------------
# Section 1: Simulate Sample Data (Self-Contained, Optional)
# -----------------------------------------
# This section creates dummy data for testing the script.
# Comment out the call to this function in Section 8 to use real data.
def simulate_dora_data_reorg(sim_years_reorg, # Should be exactly 2 years for this logic
                             initial_teams,   # Teams present in sim_years_reorg[0]
                             reorg_map_child_to_parents, # {'X': ['A','B'], ...}
                             sim_n_resp_per_team_year,
                             category_map_sim, response_options_sim):
    """
    Generates simulated DORA survey data reflecting a reorg between two years.
    Calculates Year 2 traits based on Year 1 parents + improvement.
    Returns df_raw and true parameters used for simulation.
    """
    print("--- Section 1: Simulating Sample DORA Data with Reorg (for testing) ---")
    
    if len(sim_years_reorg) != 2:
        raise ValueError("Reorg simulation logic currently requires exactly two years.")
    year1, year2 = sim_years_reorg[0], sim_years_reorg[1]

    np.random.seed(RANDOM_SEED)
    n_cats_response_sim = len(response_options_sim)
    n_latent_sim = len(category_map_sim)

    # Derive questions list for simulation
    questions_sim = [q for qs in category_map_sim.values() for q in qs]
    question_categories_sim = {}
    cat_idx_sim = 1
    for cat_name, cat_questions in category_map_sim.items():
        for q in cat_questions:
            question_categories_sim[q] = cat_idx_sim
        cat_idx_sim += 1

    # Simulate a correlation matrix between latent dimensions (same as before)
    base_corr = 0.4
    corr_matrix = np.ones((n_latent_sim, n_latent_sim)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = np.random.uniform(-0.2, 0.2, (n_latent_sim, n_latent_sim))
    noise = (noise + noise.T) / 2; np.fill_diagonal(noise, 0.0)
    true_Omega = np.clip(corr_matrix + noise, 0.05, 0.95); np.fill_diagonal(true_Omega, 1.0)
    print("Simulated True Correlation Matrix (Omega):")
    print(np.round(true_Omega, 2))

    # Simulate true means and SDs for latent traits
    true_mu = np.random.normal(0, 0.2, n_latent_sim)
    true_sigma = np.random.lognormal(-0.5, 0.2, n_latent_sim)
    cov_matrix = np.diag(true_sigma) @ true_Omega @ np.diag(true_sigma)

    def generate_correlated_traits(mean_vector, cov_matrix):
        return np.random.multivariate_normal(mean_vector, cov_matrix)

    # --- Simulate Year 1 Traits ---
    true_latent_y1 = {} # Store true thetas for Year 1: {team: [theta_l1, ...]}
    latent_for_resp = {} # Store for response generation: {(team, year, cat_idx): theta_val}

    print(f"Simulating traits for initial teams in {year1}: {initial_teams}")
    for team in initial_teams:
        team_base_mean_offset = np.random.normal(0, 0.4) # Team variation
        mean_vector = true_mu + team_base_mean_offset
        traits_y1 = generate_correlated_traits(mean_vector, cov_matrix)
        true_latent_y1[team] = traits_y1
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year1, cat_idx)] = traits_y1[cat_idx - 1]

    # --- Simulate Year 2 Traits based on Reorg ---
    true_latent_y2 = {} # Store true thetas for Year 2: {team: [theta_l1, ...]}
    year2_teams = list(reorg_map_child_to_parents.keys())
    print(f"Simulating traits for reorganized teams in {year2}: {year2_teams}")

    for child_team, parent_teams in reorg_map_child_to_parents.items():
        if not parent_teams: # New team in Year 2
            print(f"  Simulating new team: {child_team}")
            team_base_mean_offset = np.random.normal(0, 0.4)
            mean_vector = true_mu + team_base_mean_offset # Generate fresh traits
            traits_y2 = generate_correlated_traits(mean_vector, cov_matrix)
        else:
            # Combine parent traits (e.g., average) and add improvement/noise
            print(f"  Simulating reorg team: {child_team} from parents {parent_teams}")
            parent_thetas = [true_latent_y1.get(p_team) for p_team in parent_teams if p_team in true_latent_y1]
            if not parent_thetas:
                 print(f"    Warning: No valid parent thetas found for {child_team}. Simulating as new team.")
                 team_base_mean_offset = np.random.normal(0, 0.4)
                 mean_vector = true_mu + team_base_mean_offset
                 traits_y2 = generate_correlated_traits(mean_vector, cov_matrix)
            else:
                # Simple average of parent thetas
                avg_parent_theta = np.mean(np.array(parent_thetas), axis=0)
                # Simulate improvement applied to the average parent capability
                category_improvement_mean = 0.4
                improvements = np.random.normal(category_improvement_mean, 0.2, n_latent_sim) \
                               + np.random.normal(0, 0.15) # Noise/variation in resulting team capability
                traits_y2 = avg_parent_theta + improvements

        true_latent_y2[child_team] = traits_y2
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(child_team, year2, cat_idx)] = traits_y2[cat_idx - 1]

    # Combine true latent traits for storage
    true_latent_combined = {}
    for team, traits in true_latent_y1.items(): true_latent_combined[(team, year1)] = traits
    for team, traits in true_latent_y2.items(): true_latent_combined[(team, year2)] = traits

    # --- Simulate Item Parameters (same as before) ---
    true_a = {q: np.random.lognormal(-0.1, 0.4) for q in questions_sim}
    true_cutpoints = {}
    for q in questions_sim:
        item_difficulty = np.random.normal(0, 0.7)
        num_cutpoints = n_cats_response_sim - 1
        spacings = np.maximum(np.random.lognormal(0, 0.3, num_cutpoints), 0.3)
        raw_cutpoints = np.cumsum(spacings)
        centered_cutpoints = raw_cutpoints - np.mean(raw_cutpoints) + item_difficulty
        true_cutpoints[q] = np.sort(centered_cutpoints)

    # --- Simulate Responses (same function as before) ---
    def simulate_response_gr(theta_val, a, cutpoints, response_options_sim):
        eta = a*theta_val
        cum_prob_le = sps.expit(cutpoints - eta)
        n_cats_sim = len(response_options_sim)
        probs = np.zeros(n_cats_sim)
        probs[0] = cum_prob_le[0]
        for k in range(1, n_cats_sim - 1): probs[k] = cum_prob_le[k] - cum_prob_le[k-1]
        probs[n_cats_sim - 1] = 1.0 - cum_prob_le[n_cats_sim - 2]
        probs = np.maximum(probs, 1e-9); probs /= probs.sum()
        return np.random.choice(response_options_sim, p=probs)

    # --- Generate DataFrame ---
    data = []
    # Year 1 Data
    for team in initial_teams:
        for i in range(sim_n_resp_per_team_year):
            row = {YEAR_COL: year1, ID_VAR: team}
            for q in questions_sim:
                cat = question_categories_sim[q]
                theta_true = latent_for_resp[(team, year1, cat)]
                a_val = true_a[q]; cp_val = true_cutpoints[q]
                if np.random.rand() < 0.05: row[q] = np.nan
                else: row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
            data.append(row)
    # Year 2 Data
    for team in year2_teams:
         for i in range(sim_n_resp_per_team_year):
            row = {YEAR_COL: year2, ID_VAR: team}
            for q in questions_sim:
                cat = question_categories_sim[q]
                theta_true = latent_for_resp[(team, year2, cat)]
                a_val = true_a[q]; cp_val = true_cutpoints[q]
                if np.random.rand() < 0.05: row[q] = np.nan
                else: row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
            data.append(row)

    df = pd.DataFrame(data)
    print(f"Simulated data generated with {len(df)} rows reflecting reorg.")
    print("Simulated Sample Data (first 5 rows):")
    print(df.head())

    true_params = {
        'true_a': true_a,
        'true_cutpoints': true_cutpoints,
        'true_Omega': true_Omega,
        'true_latent': true_latent_combined, # {(team, year): [thetas]}
        'true_mu': true_mu,
        'true_sigma': true_sigma
    }

    return df, true_params


# --- How to call it in Section 8 (Main Execution Flow) ---
# Comment out this block if using real data

# --- Flag to control simulation ---
RUN_SIMULATION = True # Set to False to load real data

# --- Run Simulation (if flag is True) ---
df_raw = None
true_params = None
if RUN_SIMULATION:
    # Define simulation parameters including reorg
    sim_years_list_reorg = [2023, 2024] # Exactly 2 years for this sim
    # Initial teams in Year 1
    sim_initial_teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D', 'Team_E', 'Team_F']
    # Reorg mapping: Child (Y2) -> [Parents (Y1)]
    sim_reorg_map = {
        'Team_X': ['Team_A', 'Team_B'], 'Team_Y': ['Team_B', 'Team_C'],
        'Team_Z': ['Team_A', 'Team_C', 'Team_D'], # Reorged teams
        'Team_E': ['Team_E'], 'Team_F': ['Team_F'], # Unchanged teams
        'Team_G': [] # New team in Y2
    }
    sim_n_resp = 20 # responses per team-year

    df_raw, true_params = simulate_dora_data_reorg(
        sim_years_reorg=sim_years_list_reorg,
        initial_teams=sim_initial_teams,
        reorg_map_child_to_parents=sim_reorg_map,
        sim_n_resp_per_team_year=sim_n_resp,
        category_map_sim=category_mapping_initial, # Use initial map for sim
        response_options_sim=RESPONSE_OPTIONS
    )
else:
    print("--- Section 1: Skipped Data Simulation ---")
    # Real data loading happens in the next section

# --- Rest of the script (Sections 2 through 9) remains the same ---
# Make sure Section 8 defines the actual reorg mapping needed for plotting
# based on your real team names and structure.