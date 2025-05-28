# -*- coding: utf-8 -*-
"""
Comprehensive MGRM Analysis Script for DORA Survey Data

Purpose: Analyze Likert-scale DORA survey responses using a Bayesian 
         Multidimensional Graded Response Model (MGRM) with correlated latent traits.
         Estimates latent capabilities (theta) for engineering teams over time, 
         assesses improvement, estimates correlations between capabilities, 
         and performs rigorous diagnostic checks.

Structure:
    0. Imports & Global Settings
    1. Simulate Sample Data (Self-Contained, Optional)
    2. Data Loading & Preprocessing (Incl. Validation & Filtering)
    3. Stan Model Definition (MGRM, PyStan 2.19 Compatible)
    4. Smart Initial Values & Stan Execution
    5. Diagnostic Checks Function
    6. Results Extraction & Processing
    7. Visualization Functions (Incl. adjustText)
    8. Main Execution Flow & Reporting (Incl. Optional Parameter Recovery)
    9. DORA Specific Considerations (Commentary)
"""

import os

import arviz as az
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan  # Ensure v2.19 is installed for compatibility
import scipy.special as sps
import seaborn as sns
from adjustText import adjust_text  # For avoiding label overlap

# -----------------------------------------
# Section 0: Imports & Global Settings
# -----------------------------------------

# --- MCMC Settings ---
ITER_WARMUP = 1000
ITER_SAMPLING = 1500  # Increased sampling iterations for better n_eff
CHAINS = 4
THIN = 1
RANDOM_SEED = 42

# --- Diagnostic Thresholds ---
RHAT_THRESHOLD = 1.05
NEFF_RATIO_THRESHOLD = 0.1  # Min effective samples as fraction of post-warmup samples
PARETO_K_THRESHOLD = 0.7
HIGH_CORR_THRESHOLD = 0.85
LOW_DISCRIMINATION_THRESHOLD = 0.3
MIN_RESPONSES_PER_GROUP = 5  # Min responses per team-year to avoid warning

# --- File Paths & Output ---
OUTPUT_DIR = 'dora_analysis_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# We are recompiling each time now, so no need for STAN_MODEL_FILE

# --- Stan Compiler Flags ---
# Suppress specific warnings common in PyStan 2.19 interaction
STAN_COMPILE_FLAGS = [
    '-Wno-int-in-bool-context',
    '-Wno-unused-local-typedef',
    '-Wno-misleading-indentation',
    '-Wno-deprecated-declarations'
]

# --- Core Survey Definition ---
# These should be defined based on your actual survey structure
YEAR_COL = 'year'
ID_VAR = 'team_id'  # Use 'team_id' or similar for DORA context

# Define response options (Likert scale, e.g., 1 to 5 or 1 to 6) - MODIFY IF NEEDED
RESPONSE_OPTIONS = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE = len(RESPONSE_OPTIONS)

# Define categories (DORA domains/capabilities) and their associated questions 
# !!! MODIFY FOR YOUR ACTUAL SURVEY !!!
# This dictionary is a CORE ASSUMPTION of the model.
# Example categories relevant to DORA
category_mapping_initial = {
    'Deployment & Release':       ['q1', 'q2', 'q3'],
    'Monitoring & Observability': ['q4', 'q5'],
    'Technical Practices':        ['q6', 'q7', 'q8'],
    'Team Collaboration':         ['q9', 'q10'],
    'Process Efficiency':         ['q11'],
    'Learning & Development':     ['q12', 'q13'],
    'System Reliability':         ['q14', 'q15'],  # Example
    'Change Management':          ['q16']  # Example
}

# Derive basic helper maps (will be refined after data loading based on available questions)
question_to_cat_name_initial = {q: cat_name for cat_name, qs in category_mapping_initial.items() for q in qs}
N_LATENT_initial = len(category_mapping_initial)

print(f"Defined {N_LATENT_initial} categories with {sum(map(len, category_mapping_initial.values()))} total questions.")
print(f"Response scale: {RESPONSE_OPTIONS}")

# --- Define Reorg Mapping ---
# !!! IMPORTANT: DEFINE YOUR ACTUAL MAPPING HERE !!!
# Example: Child Team (Year 2) -> [List of Parent Teams (Year 1)]
reorg_mapping_y2_to_y1 = {
    # Year 2 Team: [List of constituent Year 1 teams]
    'Team_X': ['Team_A', 'Team_B'],
    'Team_Y': ['Team_B', 'Team_C'],
    'Team_Z': ['Team_A', 'Team_C', 'Team_D'],
    # Add teams that didn't change (map to themselves)
    'Team_E': ['Team_E'],
    'Team_F': ['Team_F'],
    # Add teams that only existed in Year 2 (map to empty list or special marker)
    'Team_G': []  # New team in Year 2
    # Note: Teams only in Year 1 won't appear as keys here.
}


# Identify teams only present in Year 1 (no children in Year 2 mapping)
# This requires knowing all teams present in Year 1
# year1_teams = list(theta_df[theta_df[year_col_viz] == year1][id_var_viz].unique()) # Assuming theta_df exists
# year1_only_teams = [t for t in year1_teams if t not in parent_to_child_mapping]

# -----------------------------------------
# Section 1: Simulate Sample Data (Self-Contained, Optional)
# -----------------------------------------
# This section creates dummy data for testing the script.
# Comment out the call to this function in Section 8 to use real data.
def simulate_dora_data(sim_years, sim_teams, sim_n_resp_per_team_year,
                       category_map_sim, response_options_sim):
    """
    Generates simulated DORA survey data with correlated traits and improvement.
    Returns df_raw and true parameters used for simulation.
    """
    print("--- Section 1: Simulating Sample DORA Data (for testing) ---")

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

    # Simulate a correlation matrix between latent dimensions
    base_corr = 0.4  # Slightly higher correlation for DORA domains maybe
    corr_matrix = np.ones((n_latent_sim, n_latent_sim)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = np.random.uniform(-0.2, 0.2, (n_latent_sim, n_latent_sim))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    true_Omega = np.clip(corr_matrix + noise, 0.05, 0.95)  # Keep correlations plausible
    np.fill_diagonal(true_Omega, 1.0)
    print("Simulated True Correlation Matrix (Omega):")
    print(np.round(true_Omega, 2))

    # Simulate true means and SDs for latent traits (could vary slightly)
    true_mu = np.random.normal(0, 0.2, n_latent_sim)
    true_sigma = np.random.lognormal(-0.5, 0.2, n_latent_sim)  # Ensure positive, typically < 1

    # Use Cholesky decomposition for generating correlated data
    cov_matrix = np.diag(true_sigma) @ true_Omega @ np.diag(true_sigma)

    def generate_correlated_traits(mean_vector, cov_matrix):
        return np.random.multivariate_normal(mean_vector, cov_matrix)

    # Simulate latent traits (theta) for each team-year
    true_latent = {}  # Store true thetas: {(team, year): [theta_l1, theta_l2,...]}
    latent_for_resp = {}  # Store for response generation: {(team, year, cat_idx): theta_val}

    for year in sim_years:
        for team in sim_teams:
            if year == sim_years[0]:
                # Initial latent traits - different baseline for each team
                team_base_mean_offset = np.random.normal(0, 0.4)  # Team variation
                mean_vector = true_mu + team_base_mean_offset
                traits = generate_correlated_traits(mean_vector, cov_matrix)
            else:
                # Simulate improvement from previous year, assuming comparison is Y0 -> Y1
                if year > sim_years[0] and (team, sim_years[0]) in true_latent:
                    previous_traits = true_latent[(team, sim_years[0])]
                    # Improvements vary by category and team slightly
                    category_improvement_mean = 0.4
                    improvements = np.random.normal(category_improvement_mean, 0.2, n_latent_sim) \
                                   + np.random.normal(0, 0.1)  # Team variation in improvement
                    traits = previous_traits + improvements
                else:  # Team only appears later, generate initial traits
                    team_base_mean_offset = np.random.normal(0, 0.4)
                    mean_vector = true_mu + team_base_mean_offset
                    traits = generate_correlated_traits(mean_vector, cov_matrix)

            true_latent[(team, year)] = traits
            for cat_idx in range(1, n_latent_sim + 1):
                latent_for_resp[(team, year, cat_idx)] = traits[cat_idx - 1]

    # Simulate true item parameters (discrimination 'a', cutpoints 'cut')
    true_a = {q: np.random.lognormal(-0.1, 0.4) for q in questions_sim}  # Allow slightly lower discrim on average
    true_cutpoints = {}
    for q in questions_sim:
        # Simulate variation in item difficulty (mean position of cutpoints)
        item_difficulty = np.random.normal(0, 0.7)
        # Generate ordered cutpoints with some spacing noise
        num_cutpoints = n_cats_response_sim - 1
        spacings = np.random.lognormal(0, 0.3, num_cutpoints)  # Spacing between cutpoints
        # Ensure minimum spacing
        spacings = np.maximum(spacings, 0.3)
        raw_cutpoints = np.cumsum(spacings)
        # Center the cutpoints around the item difficulty
        centered_cutpoints = raw_cutpoints - np.mean(raw_cutpoints) + item_difficulty
        true_cutpoints[q] = np.sort(centered_cutpoints)

    def simulate_response_gr(theta_val, a, cutpoints, response_options_sim):
        """Simulates ordinal response using Graded Response Model."""
        logits = a * (theta_val - cutpoints)
        # Cumulative probabilities P(Y > k | theta) = sigmoid(logits_k)
        # Or P(Y <= k | theta) = sigmoid(cut_k - eta) = sigmoid(cut_k - a*theta)
        # Stan's ordered_logistic uses P(Y <= k | eta) = sigmoid(cut_k - eta)
        # Let's simulate based on Stan's logic for consistency
        # P(y <= k) = sigmoid(cut_k - a*theta)
        # Need P(y=k) = P(y<=k) - P(y<=k-1)
        eta = a * theta_val
        cum_prob_le = sps.expit(cutpoints - eta)  # P(Y <= k) for k=1..C-1

        n_cats_sim = len(response_options_sim)
        probs = np.zeros(n_cats_sim)
        probs[0] = cum_prob_le[0]  # P(Y=1) = P(Y<=1)
        for k in range(1, n_cats_sim - 1):
            probs[k] = cum_prob_le[k] - cum_prob_le[k - 1]  # P(Y=k) = P(Y<=k) - P(Y<=k-1)
        probs[n_cats_sim - 1] = 1.0 - cum_prob_le[n_cats_sim - 2]  # P(Y=C) = 1 - P(Y<=C-1)

        # Handle potential numerical instability
        probs = np.maximum(probs, 1e-9)  # Ensure non-negative, add epsilon
        probs /= probs.sum()  # Ensure sums to 1
        return np.random.choice(response_options_sim, p=probs)

    # Generate response data frame
    data = []
    for year in sim_years:
        for team in sim_teams:
            # Simulate some teams appearing/disappearing
            if year == sim_years[0] and team == sim_teams[-1]: continue  # Last team appears in Y2
            if year == sim_years[-1] and team == sim_teams[-2]: continue  # Second last team disappears in Y2

            for i in range(sim_n_resp_per_team_year):
                row = {YEAR_COL: year, ID_VAR: team}
                for q in questions_sim:
                    cat = question_categories_sim[q]
                    theta_true = latent_for_resp[(team, year, cat)]
                    a_val = true_a[q]
                    cp_val = true_cutpoints[q]
                    # Simulate some NAs, more likely on less relevant categories? (Simplification here)
                    if np.random.rand() < 0.05:  # 5% random NA overall
                        row[q] = np.nan
                    else:
                        row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
                data.append(row)

    df = pd.DataFrame(data)
    print(f"Simulated data generated with {len(df)} rows.")
    print("Simulated Sample Data (first 5 rows):")
    print(df.head())

    true_params = {
        'true_a':         true_a,
        'true_cutpoints': true_cutpoints,
        'true_Omega':     true_Omega,
        'true_latent':    true_latent,  # {(team, year): [thetas]}
        'true_mu':        true_mu,
        'true_sigma':     true_sigma
    }

    return df, true_params


# --- Flag to control simulation ---
RUN_SIMULATION = True  # Set to False to load real data

# --- Run Simulation (if flag is True) ---
df_raw = None
true_params = None
if RUN_SIMULATION:
    # Define simulation parameters (independent of real data)
    sim_years_list = [2023, 2024]
    sim_teams_list = [f'Team_{chr(65 + i)}' for i in range(10)]  # 10 teams
    sim_n_resp = 20  # responses per team-year

    df_raw, true_params = simulate_dora_data(
        sim_years=sim_years_list,
        sim_teams=sim_teams_list,
        sim_n_resp_per_team_year=sim_n_resp,
        category_map_sim=category_mapping_initial,  # Use initial map for sim
        response_options_sim=RESPONSE_OPTIONS
    )
else:
    print("--- Section 1: Skipped Data Simulation ---")
    # Real data loading happens in the next section

# -----------------------------------------
# Section 2: Data Loading & Preprocessing
# -----------------------------------------
print("\n--- Section 2: Loading & Preprocessing Data ---")

if df_raw is None:
    # --- !!! LOAD YOUR REAL DATA HERE !!! ---
    # Example: df_raw = pd.read_csv('your_dora_survey_data.csv')
    try:
        # Replace with your actual loading mechanism
        # df_raw = pd.read_csv('your_dora_survey_data.csv') 
        print("ERROR: Real data loading not implemented. Please load data into df_raw.")
        exit()
    except FileNotFoundError:
        print("ERROR: Real data file not found. Cannot proceed.")
        exit()
    # --- Perform any initial cleaning specific to your real data ---
    # E.g., renaming columns, converting year format etc.

# --- Derive Data-Specific Variables ---
actual_years = sorted(df_raw[YEAR_COL].unique())
actual_teams = df_raw[ID_VAR].unique()
actual_questions_in_data = df_raw.columns.intersection(question_to_cat_name_initial.keys())

print(f"Data contains years: {actual_years}")
print(f"Data contains {len(actual_teams)} unique teams.")
print(f"Found {len(actual_questions_in_data)} questions from category mapping in the data.")

# --- Filter Questions: Keep only questions with valid responses across ALL analyzed years ---
print(f"Filtering questions: Keeping only those with responses in ALL years: {actual_years}")
questions_to_use = []
questions_dropped = []
question_responses_by_year = df_raw.groupby(YEAR_COL)  # Group for yearly checks

valid_questions_per_year = {}
for year in actual_years:
    valid_questions_per_year[year] = df_raw[df_raw[YEAR_COL] == year][actual_questions_in_data].dropna(axis=1,
                                                                                                       how='all').columns

# Find intersection: questions valid in ALL years
if actual_years:
    questions_to_use = set(valid_questions_per_year[actual_years[0]])
    for year in actual_years[1:]:
        questions_to_use.intersection_update(valid_questions_per_year[year])
    questions_to_use = sorted(list(questions_to_use))
else:
    questions_to_use = []  # No years found

questions_dropped = sorted(list(set(actual_questions_in_data) - set(questions_to_use)))

if questions_dropped:
    print(f"WARNING: Dropping {len(questions_dropped)} questions not present or without non-NA "
          f"responses across all analyzed years ({actual_years}):")
    print(f"  Dropped: {questions_dropped}")
if not questions_to_use:
    print("ERROR: No questions found with valid responses across all specified years. Cannot proceed.")
    exit()
print(f"Using {len(questions_to_use)} questions for analysis.")

# --- Refine Category Mapping based on questions_to_use ---
category_mapping_final = {}
question_categories_final = {}  # Map: question -> category index (1-based)
question_to_cat_name_final = {}  # Map: question -> category name
cat_idx_final = 1
final_cat_names = []

for cat_name_orig, qs_orig in category_mapping_initial.items():
    qs_in_cat_to_use = [q for q in qs_orig if q in questions_to_use]
    if qs_in_cat_to_use:  # Only keep category if it has questions left
        category_mapping_final[cat_name_orig] = qs_in_cat_to_use
        final_cat_names.append(cat_name_orig)
        for q in qs_in_cat_to_use:
            question_categories_final[q] = cat_idx_final
            question_to_cat_name_final[q] = cat_name_orig
        cat_idx_final += 1

if not category_mapping_final:
    print("ERROR: No categories remaining after filtering questions. Check data and mapping.")
    exit()

cat_idx_to_name_final = {idx: name for idx, name in enumerate(final_cat_names, 1)}
N_LATENT_final = len(category_mapping_final)

print(f"Using {N_LATENT_final} categories based on available questions:")
print(list(category_mapping_final.keys()))

# --- Melt Data using only questions_to_use ---
print("Reshaping data to long format...")
df_long = df_raw.melt(id_vars=[YEAR_COL, ID_VAR],
                      value_vars=questions_to_use,  # Use filtered list
                      var_name='question',
                      value_name='response')

# --- Handle Missing / NA Responses in 'response' column ---
initial_rows = len(df_long)
df_long.dropna(subset=['response'], inplace=True)
rows_after_na = len(df_long)
print(f"Removed {initial_rows - rows_after_na} rows with NA responses.")
print("IMPORTANT: If NA rates differ systematically (e.g., DS vs Eng teams), "
      "filtering NAs might bias results. Consider separate analyses "
      "for relevant subgroups and question sets (see Section 9).")

# --- Validate Response Values & Convert Type ---
try:
    # Attempt conversion, coercing errors to NaN
    df_long['response_numeric'] = pd.to_numeric(df_long['response'], errors='coerce')
    df_long.dropna(subset=['response_numeric'], inplace=True)  # Drop rows that couldn't be converted
    df_long['response'] = df_long['response_numeric'].astype(int)

    # Check if responses are within the allowed range
    is_valid_response = df_long['response'].isin(RESPONSE_OPTIONS)
    invalid_responses = df_long[~is_valid_response]['response'].unique()
    if not is_valid_response.all():
        print(
            f"WARNING: Found responses outside the expected range {RESPONSE_OPTIONS}: {invalid_responses}. Filtering these rows.")
        df_long = df_long[is_valid_response]
    print("Response column validated and converted to integer.")
    df_long.drop(columns=['response_numeric'], inplace=True)  # Remove temporary column
except Exception as e:
    print(f"ERROR during response validation/conversion: {e}. Check data quality.")
    exit()

if df_long.empty:
    print("ERROR: No valid observations remaining after NA removal and validation. Cannot proceed.")
    exit()

# --- Create Indices for Stan (1-based) ---
# Group index (unique team-year combination)
df_long['group'] = df_long[ID_VAR].astype(str) + "|" + df_long[YEAR_COL].astype(str)
groups_final = df_long['group'].unique()
group_idx_map = {g: i + 1 for i, g in enumerate(groups_final)}
df_long['group_idx'] = df_long['group'].map(group_idx_map)

group_idx_to_info = {idx: {"team": g.split('|')[0], "year": int(g.split('|')[1])}
                     for g, idx in group_idx_map.items()}

# Question index (based on questions_to_use)
question_idx_map = {q: i + 1 for i, q in enumerate(questions_to_use)}  # Use filtered list
df_long['question_idx'] = df_long['question'].map(question_idx_map)

# Map from question index (1..K) to dimension index (1..L) for Stan data block
# Create this map using the final indices and final category assignments
q_idx_to_dim_idx = {}
for q, q_idx_stan in question_idx_map.items():
    cat_name = question_to_cat_name_final[q]
    # Find the final index for this category name
    dim_idx_stan = next((idx for idx, name in cat_idx_to_name_final.items() if name == cat_name), None)
    if dim_idx_stan:
        q_idx_to_dim_idx[q_idx_stan] = dim_idx_stan
    else:
        print(f"ERROR: Could not find final category index for question {q} (category {cat_name})")
        exit()

# Create the array for Stan data block, ordered by question_idx_stan (1 to K)
question_to_dimension_stan = [q_idx_to_dim_idx[k] for k in range(1, len(question_idx_map) + 1)]

# --- Data Validation Checks ---
print("Performing data validation checks...")
# Check for teams with very few responses per year
responses_per_group = df_long.groupby('group').size()
low_response_groups = responses_per_group[responses_per_group < MIN_RESPONSES_PER_GROUP]
if not low_response_groups.empty:
    print(f"WARNING: {len(low_response_groups)} team-year groups have < {MIN_RESPONSES_PER_GROUP} responses.")
    print("  Theta estimates for these groups will have high uncertainty.")
    # print(low_response_groups) # Optionally print the groups

# Check for teams completely dropped
teams_in_processed_data = df_long[ID_VAR].unique()
teams_dropped_completely = sorted(list(set(actual_teams) - set(teams_in_processed_data)))
if teams_dropped_completely:
    print(f"WARNING: {len(teams_dropped_completely)} teams from the raw data have NO valid responses "
          f"after filtering for questions present across all years and removing NAs. "
          f"These teams are EXCLUDED from the analysis:")
    print(f"  Excluded Teams: {teams_dropped_completely}")

# --- Final Data Counts for Stan ---
N_final = df_long.shape[0]
J_final = len(groups_final)  # Number of unique team-year groups
K_final = len(questions_to_use)  # Number of unique questions included
C_final = N_CATEGORIES_RESPONSE  # Number of response categories (remains constant)
L_final = N_LATENT_final  # Number of latent dimensions included

print("\nFinal Data Counts for Stan:")
print(f"  N (observations): {N_final}")
print(f"  J (groups/team-years): {J_final}")
print(f"  K (unique questions): {K_final}")
print(f"  C (response categories): {C_final}")
print(f"  L (latent dimensions): {L_final}")

if N_final == 0 or J_final == 0 or K_final == 0 or L_final == 0:
    print("ERROR: One of the key dimensions (N, J, K, L) is zero after processing. Cannot run Stan.")
    exit()

print("\nPreprocessed Data Sample (first 5 rows):")
print(df_long.head())

# -----------------------------------------
# Section 3: Stan Model Definition (MGRM, PyStan 2.19 Compatible)
# -----------------------------------------
print("\n--- Section 3: Defining Stan Model ---")

# MGRM Model Code for Stan
# Includes prior sensitivity check alternatives as comments
# Tailored for PyStan 2.19 syntax where declarations must be at top of blocks
stan_model_code = """
data {
  // Declarations must be at the top in PyStan 2.19
  int<lower=1> N;           // number of observations (rows in long data)
  int<lower=1> J;           // number of groups (team-year combinations)
  int<lower=1> K;           // number of questions included in analysis
  int<lower=2> C;           // number of response categories (e.g., 6 for 1-6)
  int<lower=1> L;           // number of latent dimensions (categories) included

  int<lower=1, upper=J> group_idx[N];         // group id for each observation (1..J)
  int<lower=1, upper=K> q_idx[N];             // question id for each observation (1..K)
  int<lower=1, upper=C> y[N];                 // observed ordinal response (1..C)

  // Mapping questions (1..K) to latent dimensions (1..L)
  int<lower=1, upper=L> question_to_dimension[K];
}

parameters {
  // Declarations must be at the top
  // Latent traits (theta): Non-centered parameterization
  matrix[L, J] z; // Matrix of standard normal variables (L dimensions x J groups)

  // Question parameters
  vector<lower=0>[K] a;        // discrimination parameters (1 per question)
  ordered[C-1] cutpoints[K];   // ordered cutpoints (C-1 per question)

  // Hierarchical priors for latent trait distribution
  vector[L] mu;                // mean capability for each DORA dimension across groups
  vector<lower=0>[L] sigma;    // standard deviation for each dimension

  // Correlation structure between latent dimensions
  cholesky_factor_corr[L] L_Omega; // Cholesky factor of the correlation matrix
}

transformed parameters {
  // Declarations must be at the top
  matrix[L, J] theta_raw;     // Intermediate calculation
  matrix[J, L] theta;         // Actual latent traits (team capabilities)

  // Calculate the actual latent traits (theta) for each group and dimension
  // theta = mu + diag(sigma) * L_Omega * z
  theta_raw = diag_pre_multiply(sigma, L_Omega) * z;
  // Reconstruct theta[J, L] from theta_raw[L, J] and mu[L]
  // Loop is more explicit and compatible
  for (j in 1:J) {
    for (l in 1:L) {
      theta[j, l] = mu[l] + theta_raw[l, j];
    }
  }
}

model {
  // Priors (can be anywhere in model block for PyStan 2.19)
  // Non-centered latent traits
  to_vector(z) ~ std_normal(); // Prior on the standardized latent variables

  // Hierarchical priors for theta distribution
  mu ~ normal(0, 1);      // Prior for overall mean of each dimension
                          // Sensitivity Check: Could try normal(0, 0.5) or normal(0, 2) if scale might differ drastically.

  sigma ~ normal(0, 1);   // Prior for variability of each dimension (weakly regularizing, must be >0)
                          // Sensitivity Check 1: sigma ~ cauchy(0, 2.5); // Slightly heavier tails
                          // Sensitivity Check 2: sigma ~ exponential(1); // Different shape, mean 1

  // Prior for correlation structure
  L_Omega ~ lkj_corr_cholesky(2); // LKJ prior (eta=2 favors identity but allows correlations)
                                  // Sensitivity Check 1: L_Omega ~ lkj_corr_cholesky(1); // Uniform over correlation matrices
                                  // Sensitivity Check 2: L_Omega ~ lkj_corr_cholesky(4); // Stronger push towards zero correlation

  // Priors for question parameters
  a ~ lognormal(0, 0.5); // Prior for discrimination (positive, typically moderate)
                         // Sensitivity Check 1: a ~ lognormal(0, 1); // Allow more variability in discrimination
                         // Sensitivity Check 2: a ~ half_normal(0, 1); // Alternative positive distribution

  // Prior for cutpoints (ensure reasonable scale)
  for (k in 1:K) {
      cutpoints[k] ~ normal(0, 3); // Fairly weak prior on cutpoint locations
                                   // Sensitivity Check: cutpoints[k] ~ normal(0, 1.5); // Tighter prior if scale is expected near 0
  }

  // Likelihood: Graded Response Model using ordered_logistic
  // Loop through observations
  for (n in 1:N) {
    // Declare local variables used in the loop if needed by strict C++ rules Stan follows
    int current_q;
    int current_g;
    int current_dim;
    real eta;

    current_q = q_idx[n];       // Which question? (1..K)
    current_g = group_idx[n];   // Which group (team-year)? (1..J)
    current_dim = question_to_dimension[current_q]; // Which dimension does this question load on? (1..L)

    // Calculate linear predictor (eta) for ordered logistic
    eta = a[current_q] * theta[current_g, current_dim];

    // Calculate likelihood of the observed response y[n] (1..C)
    y[n] ~ ordered_logistic(eta, cutpoints[current_q]);
  }
}

generated quantities {
  // Declarations must be at the top
  matrix[L, L] Omega; // Estimated correlation matrix between DORA capabilities
  vector[N] log_lik;  // Log-likelihood for each observation (for LOOIC)

  // Recover the full correlation matrix from its Cholesky factor
  Omega = multiply_lower_tri_self_transpose(L_Omega);

  // Calculate log likelihood for each observation
  for (n in 1:N) {
    // Declare local variables
    int current_q;
    int current_g;
    int current_dim;
    real eta;

    current_q = q_idx[n];
    current_g = group_idx[n];
    current_dim = question_to_dimension[current_q];
    eta = a[current_q] * theta[current_g, current_dim];

    log_lik[n] = ordered_logistic_lpmf(y[n] | eta, cutpoints[current_q]);
  }
}
"""
print("Stan model code defined.")

# -----------------------------------------
# Section 4: Smart Initial Values & Stan Execution
# -----------------------------------------
print("\n--- Section 4: Defining Smart Inits & Running Stan ---")


def generate_smart_init(df_long_processed, K_val, L_val, J_val, C_val, question_idx_map_val, response_options_val):
    """
    Generate smarter initial values for Stan MCMC chains, using data statistics.
    """
    print("Generating smart initial values...")
    # Calculate empirical cutpoints based on response quantiles per question
    initial_cutpoints = []
    responses_per_question = df_long_processed.groupby('question_idx')['response']
    num_cutpoints = C_val - 1
    # Global quantiles as fallback
    global_probs = np.linspace(1 / C_val, 1 - 1 / C_val, num_cutpoints)
    global_quantiles = np.quantile(df_long_processed['response'], global_probs)
    # Adjust global quantiles to be midpoints between response options for better centering
    global_midpoints = (global_quantiles[:-1] + global_quantiles[1:]) / 2.0
    default_cuts = np.interp(np.linspace(0, 1, C_val + 1)[1:-1],  # Probabilities for C-1 cutoffs
                             np.linspace(0, 1, len(response_options_val) + 1)[1:-1],  # Midpoints of response options
                             np.array(response_options_val[:-1]) + 0.5)
    default_cuts_centered = default_cuts - np.mean(df_long_processed['response'])  # Center around mean response? Or 0?

    for k_idx_stan in range(1, K_val + 1):
        q_responses = responses_per_question.get_group(k_idx_stan)
        if len(q_responses) > 10:  # Use empirical quantiles if enough data
            probs = np.linspace(0, 1, C_val + 1)[1:-1]  # Probabilities for C-1 cutoffs
            q_quantiles = np.quantile(q_responses, probs)
            # Convert quantiles (response values) to latent scale (centered around 0)
            # A rough approximation: z-score the quantiles
            q_cuts = (q_quantiles - q_responses.mean()) / (q_responses.std() + 1e-6)
            # Ensure they are ordered (quantiles should be, but check)
            q_cuts = np.sort(q_cuts)
            # Ensure minimum difference
            min_diff = 0.1
            for i in range(len(q_cuts) - 1):
                if q_cuts[i + 1] - q_cuts[i] < min_diff:
                    q_cuts[i + 1] = q_cuts[i] + min_diff
            initial_cutpoints.append(q_cuts)
        else:  # Fallback to default centered values
            initial_cutpoints.append(default_cuts_centered + np.random.normal(0, 0.1, num_cutpoints))  # Add noise

    # Generate other initial values with random noise around defaults
    init = {
        'mu':        np.random.normal(0, 0.1, L_val),
        'sigma':     np.abs(np.random.normal(0.8, 0.1, L_val)),  # Start around 0.8
        'a':         np.abs(np.random.normal(1.0, 0.1, K_val)),  # Start around 1.0
        'z':         np.random.normal(0, 0.2, (L_val, J_val)),  # Slightly larger noise maybe
        'cutpoints': initial_cutpoints,
        # Start L_Omega near identity (no correlation) - Cholesky factor is Identity matrix
        'L_Omega':   np.eye(L_val)
    }
    print("Smart initial values generated.")
    return init


# Prepare data dictionary for Stan using final dimensions
stan_data = {
    'N':                     N_final,
    'J':                     J_final,
    'K':                     K_final,
    'C':                     C_final,
    'L':                     L_final,
    'group_idx':             df_long['group_idx'].values.astype(int),  # Ensure integer type
    'q_idx':                 df_long['question_idx'].values.astype(int),
    'y':                     df_long['response'].values.astype(int),
    'question_to_dimension': question_to_dimension_stan
}

# Compile Stan model (recompiles every time)
print("Compiling Stan model (PyStan 2.19)...")
try:
    sm = pystan.StanModel(
        model_code=stan_model_code,
        extra_compile_args=STAN_COMPILE_FLAGS  # Add flags here
    )
    print("Stan model compiled successfully.")
except Exception as e:
    print(f"ERROR during Stan model compilation: {e}")
    # Consider printing stan_model_code for debugging if compilation fails
    # print("\n--- Stan Model Code ---")
    # print(stan_model_code)
    # print("-----------------------\n")
    exit()

# Generate initial values list for chains
inits_list = [generate_smart_init(df_long, K_final, L_final, J_final, C_final, question_idx_map, RESPONSE_OPTIONS)
              for _ in range(CHAINS)]

# Run MCMC sampler
print(f"Running MCMC sampling ({CHAINS} chains, {ITER_WARMUP} warmup, {ITER_SAMPLING} sampling)...")
try:
    fit = sm.sampling(
        data=stan_data,
        iter=ITER_WARMUP + ITER_SAMPLING,
        warmup=ITER_WARMUP,
        chains=CHAINS,
        thin=THIN,
        seed=RANDOM_SEED,
        init=inits_list,
        # control={'adapt_delta': 0.9} # Uncomment/adjust if divergences occur
    )
    print("Sampling complete.")
except Exception as e:
    print(f"ERROR during Stan sampling: {e}")
    # Common issues: initialization problems, model misspecification, data issues
    # Check the Stan output/error messages for more details
    exit()

# -----------------------------------------
# Section 5: Diagnostic Checks Function
# -----------------------------------------
print("\n--- Section 5: Defining Diagnostic Checks Function ---")


def run_diagnostic_checks(fit, samples, stan_data, df_long,
                          cat_idx_to_name_map, question_idx_map,
                          n_chains, iter_sampling):
    """
    Runs comprehensive diagnostic checks on the Stan fit object and samples.
    Enhanced reporting included.
    """
    print("\n--- RUNNING DIAGNOSTIC CHECKS ---")
    warnings_found = []
    total_post_warmup_samples = n_chains * iter_sampling
    K_dims = stan_data['K']
    L_dims = stan_data['L']
    C_cats = stan_data['C']
    response_opts = list(range(1, C_cats + 1))

    # Convert fit object to ArviZ InferenceData
    try:
        # Use extract_dataset for better compatibility if from_pystan fails
        idata = az.from_pystan(
            posterior=fit,
            log_likelihood="log_lik",  # Explicitly map log_lik if named consistently
            posterior_predictive=None,  # We'll do basic PPC manually
            observed_data={"y": stan_data['y']},  # Link observed data
            coords={  # Define coordinates for dimensions
                "question":              list(question_idx_map.keys()),  # Use actual question IDs
                "category":              list(cat_idx_to_name_map.values()),
                "group":                 [group_idx_to_info[i]['team'] + "|" + str(group_idx_to_info[i]['year']) for i
                                          in range(1, stan_data['J'] + 1)],
                "response_category_dim": [f"cut_{i}" for i in range(C_cats - 1)],
                # Add more coords if needed (e.g., for observations)
            },
            dims={  # Map parameters to coordinates
                "a":         ["question"],
                "cutpoints": ["question", "response_category_dim"],
                "mu":        ["category"],
                "sigma":     ["category"],
                "L_Omega":   ["category", "category_"],  # ArviZ convention for matrix factors
                "Omega":     ["category", "category_"],
                "z":         ["category", "group"],  # Note dims are (L, J) in Stan code
                "theta":     ["group", "category"],
                "log_lik":   ["observation"]  # Need to add observation coord if not default
                # Check ArviZ docs for exact naming conventions
            }
        )
        print("Successfully converted fit object to ArviZ InferenceData.")
    except Exception as e:
        print(f"WARNING: Could not automatically convert fit to ArviZ InferenceData with full coords/dims: {e}")
        print("         Attempting checks using az.summary on fit object directly (less informative names).")
        try:
            idata = az.from_pystan(posterior=fit, log_likelihood="log_lik")  # Simpler conversion
        except Exception as e2:
            print(f"ERROR: Basic conversion to InferenceData also failed: {e2}")
            print("         Proceeding with checks on raw samples dict, diagnostics limited.")
            idata = None  # Fallback if all fails

    # --- 1. MCMC Convergence Checks ---
    print("\n--- Checking MCMC Convergence ---")
    summary = None
    if idata is not None:
        try:
            summary = az.summary(idata, round_to=3)
            print("ArviZ summary generated.")
        except Exception as e:
            print(f"ERROR generating ArviZ summary: {e}. Cannot perform summary-based checks.")
            warnings_found.append("ERROR: Could not generate ArviZ summary for MCMC checks.")
            summary = None  # Ensure summary is None if failed
    else:
        print("INFO: Cannot generate ArviZ summary without InferenceData.")
        warnings_found.append("WARNING: ArviZ InferenceData conversion failed, summary checks skipped.")

    # Check Divergences (using sampler parameters from fit object)
    try:
        sampler_params = fit.get_sampler_params(inc_warmup=False)
        # Ensure 'divergent__' key exists, handle potential KeyErrors
        divergences = sum(p.get('divergent__', np.array([0])).sum() for p in sampler_params)
        if divergences > 0:
            msg = (f"CRITICAL WARNING: {divergences} divergent transitions found! "
                   "Results are unreliable. Increase adapt_delta "
                   "(e.g., control={'adapt_delta': 0.95}) or reparameterize.")
            warnings_found.append(msg)
            print(msg)
        else:
            print("  Divergences: OK (0 found)")
    except Exception as e:
        print(f"  WARNING: Error checking divergences via get_sampler_params: {e}")

    if summary is not None:
        # Check R-hat
        max_rhat = summary['r_hat'].max()
        print(f"  Max R-hat observed: {max_rhat:.3f}")
        high_rhat_params = summary[summary['r_hat'] > RHAT_THRESHOLD]
        if not high_rhat_params.empty:
            msg = (f"WARNING: {len(high_rhat_params)} parameters have R-hat > {RHAT_THRESHOLD}. "
                   f"Chains may not have converged well. Check trace plots. "
                   f"Problematic parameters (first 10):\n{high_rhat_params.index.tolist()[:10]}")
            warnings_found.append(msg)
            print(msg)
        else:
            print(f"  R-hat Threshold Check: OK (All <= {RHAT_THRESHOLD})")

        # Check Effective Sample Size (n_eff)
        min_expected_neff = NEFF_RATIO_THRESHOLD * total_post_warmup_samples
        min_ess_bulk = summary['ess_bulk'].min()
        min_ess_tail = summary['ess_tail'].min()
        print(f"  Min ess_bulk observed: {min_ess_bulk:.0f}")
        print(f"  Min ess_tail observed: {min_ess_tail:.0f}")

        low_neff_bulk = summary[summary['ess_bulk'] < min_expected_neff]
        low_neff_tail = summary[summary['ess_tail'] < min_expected_neff]
        low_neff_params = pd.concat([low_neff_bulk, low_neff_tail]).index.unique()

        if len(low_neff_params) > 0:
            msg = (f"WARNING: {len(low_neff_params)} parameters have low n_eff (ess_bulk or ess_tail) < "
                   f"{min_expected_neff:.0f} ({NEFF_RATIO_THRESHOLD * 100:.0f}% of total samples). "
                   "Estimates are noisy. Consider running more iterations. "
                   f"Problematic parameters (first 10):\n{low_neff_params.tolist()[:10]}")
            warnings_found.append(msg)
            print(msg)
        else:
            print(f"  n_eff Threshold Check: OK (All >= {min_expected_neff:.0f})")
    else:
        print("  INFO: R-hat and n_eff checks skipped due to missing summary.")

    # --- 2. Trace Plot ("Fuzzy Caterpillar") Check ---
    print("\n--- Generating Trace Plots (Manual Check Required) ---")
    if idata is not None:
        try:
            # Get available var names from the InferenceData object
            available_vars = list(idata.posterior.data_vars.keys())

            # Select a subset of *available* parameters
            params_to_plot = []
            # Core parameters
            for p in ['mu', 'sigma']:
                if p in available_vars: params_to_plot.append(p)
            # Examples of indexed parameters (if available)
            indexed_examples = {'a': 3, 'theta': 1, 'Omega': 2}  # Param prefix -> num examples
            for prefix, num_examples in indexed_examples.items():
                # Find available params starting with the prefix
                matches = [v for v in available_vars if v.startswith(prefix + '[')]
                if matches:
                    params_to_plot.extend(matches[:num_examples])  # Add first few matches

            if not params_to_plot:
                print(
                    "  INFO: Could not find standard parameters (mu, sigma, a, etc.) in InferenceData for trace plots.")
            else:
                print(f"  Plotting traces for: {params_to_plot}")
                az.plot_trace(idata, var_names=params_to_plot)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'trace_plots_subset.png'))
                plt.show(block=False)  # Use block=False to continue execution
                plt.pause(1)  # Pause briefly to allow plot rendering
                plt.close()  # Close the plot window
                msg = ("INFO: Trace plots generated for a subset of parameters (see trace_plots_subset.png). "
                       "MANUALLY INSPECT them for good mixing (overlapping chains, 'fuzzy caterpillar' shape) "
                       "and no long-term trends.")
                print(msg)
        except Exception as e:
            msg = f"WARNING: Could not generate trace plots: {e}. Manual inspection via ArviZ is recommended."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: Trace plots skipped as ArviZ InferenceData object is not available.")

    # --- 3. Model Fit Checks ---
    print("\n--- Checking Model Fit ---")
    # Posterior Predictive Check (Histogram - Basic Overall Check)
    print("  Generating Overall Posterior Predictive Check histogram (Manual Check Required)...")
    if 'theta' in samples and 'a' in samples and 'cutpoints' in samples:
        try:
            # Reuse function defined earlier
            n_samples_total = len(samples['theta'])
            sample_idx = np.random.randint(0, n_samples_total)
            sample_theta = samples['theta'][sample_idx]
            sample_a = samples['a'][sample_idx]
            sample_cut = samples['cutpoints'][sample_idx]

            # Use the final map derived in preprocessing
            q_cat_map_series = df_long[['question_idx', 'q_cat_final']].drop_duplicates().set_index('question_idx')[
                'q_cat_final']

            pred_responses = predict_responses(
                sample_theta, sample_a, sample_cut,
                stan_data['group_idx'], stan_data['q_idx'], q_cat_map_series
            )

            plt.figure(figsize=(12, 6))
            bins = np.arange(min(response_opts) - 0.5, max(response_opts) + 1.5, 1)
            plt.subplot(1, 2, 1)
            plt.hist(stan_data['y'], bins=bins, alpha=0.7, label='Observed', density=True)
            plt.title('Observed Response Distribution (Overall)')
            plt.xlabel('Response Category')
            plt.ylabel('Density')
            plt.xticks(response_opts)

            plt.subplot(1, 2, 2)
            plt.hist(pred_responses, bins=bins, alpha=0.7, label='Predicted (1 Sample)', density=True)
            plt.title('Predicted Response Distribution (from 1 posterior sample)')
            plt.xlabel('Response Category')
            plt.ylabel('Density')
            plt.xticks(response_opts)

            plt.suptitle("Posterior Predictive Check (Overall Distribution)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_DIR, 'ppc_histogram_overall.png'))
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            msg = ("INFO: Basic PPC Histogram generated (see ppc_histogram_overall.png). "
                   "MANUALLY COMPARE observed vs. predicted. Major discrepancies indicate poor overall fit. "
                   "Consider more detailed PPCs (e.g., `az.plot_ppc`, checks per group/question) if needed.")
            print(msg)
        except Exception as e:
            msg = f"WARNING: Could not generate PPC histogram: {e}. Model fit check skipped."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: Required parameters (theta, a, cutpoints) not found in samples for PPC histogram.")

    # LOOIC Check (using PSIS-LOO)
    if idata is not None and 'log_lik' in idata.log_likelihood:
        print("  Calculating LOOIC using ArviZ (may take time)...")
        try:
            # Recompute LOO using the log_lik data within idata
            loo_result = az.loo(idata, pointwise=True, var_name='log_lik')
            print("\nLOOIC Results:")
            print(loo_result)

            # Check Pareto k values
            pareto_ks = loo_result.pareto_k.values
            max_pareto_k = np.max(pareto_ks)
            print(f"  Max Pareto k observed: {max_pareto_k:.3f}")

            problematic_k_count = np.sum(pareto_ks > PARETO_K_THRESHOLD)
            if problematic_k_count > 0:
                msg = (f"WARNING: {problematic_k_count} observations have Pareto k > {PARETO_K_THRESHOLD}. "
                       f"LOOIC estimate may be unreliable. These points are highly influential or poorly fit. "
                       "Inspect these data points and consider model refinement.")
                warnings_found.append(msg)
                print(msg)
                # Plot Pareto k values
                try:
                    az.plot_khat(loo_result, threshold=PARETO_K_THRESHOLD)
                    plt.savefig(os.path.join(OUTPUT_DIR, 'looic_pareto_k_diagnostic.png'))
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close()
                except Exception as plot_e:
                    print(f"  WARNING: Could not plot Pareto k values: {plot_e}")
            else:
                print(f"  Pareto k diagnostic: OK (All <= {PARETO_K_THRESHOLD})")

        except Exception as e:
            msg = f"WARNING: Could not calculate LOOIC: {e}. Ensure 'log_lik' is correctly computed and available."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: 'log_lik' not found in InferenceData or InferenceData unavailable. Skipping LOOIC calculation.")

    # --- 4. Parameter Estimate Checks ---
    print("\n--- Checking Parameter Estimates ---")

    # Check Correlations (Omega)
    if 'Omega' in samples:
        Omega_mean = samples['Omega'].mean(axis=0)
        # Mask diagonal for calculating max off-diagonal correlation
        np.fill_diagonal(Omega_mean, np.nan)
        max_correlation = np.nanmax(np.abs(Omega_mean))
        np.fill_diagonal(Omega_mean, 1.0)  # Put diagonal back for matrix
        print(f"  Max absolute off-diagonal correlation observed: {max_correlation:.3f}")

        merged_suggestions = []
        categories = list(cat_idx_to_name_map.values())
        for i in range(L_dims):
            for j in range(i + 1, L_dims):
                if Omega_mean[i, j] > HIGH_CORR_THRESHOLD:
                    merged_suggestions.append((categories[i], categories[j], Omega_mean[i, j]))

        if merged_suggestions:
            msg = (
                f"INFO: Found {len(merged_suggestions)} pairs of categories with mean correlation > {HIGH_CORR_THRESHOLD}. "
                "Consider if merging these categories makes theoretical sense:")
            print(msg)
            for cat1, cat2, corr in merged_suggestions:
                print(f"      - '{cat1}' and '{cat2}' (Corr: {corr:.3f})")
        # else: # No need to print OK if max is already printed
        # print(f"  High Correlations (> {HIGH_CORR_THRESHOLD}): None found.")

    else:
        print("  INFO: 'Omega' not found in samples. Skipping correlation check.")

    # Check Discrimination (a)
    if 'a' in samples:
        a_mean = samples['a'].mean(axis=0)
        min_discrimination = np.min(a_mean)
        print(f"  Min mean discrimination ('a') observed: {min_discrimination:.3f}")

        idx_to_question = {v: k for k, v in question_idx_map.items()}
        low_a_questions = []
        if len(a_mean) == K_dims:
            for k_idx_0based in range(K_dims):
                stan_idx = k_idx_0based + 1
                if a_mean[k_idx_0based] < LOW_DISCRIMINATION_THRESHOLD:
                    q_id = idx_to_question.get(stan_idx, f"Unknown Q (Index {stan_idx})")
                    # Need final question->cat mapping here
                    cat_name = question_to_cat_name_final.get(q_id, "Unknown Cat")
                    low_a_questions.append({'Question': q_id, 'Category': cat_name, 'Mean_a': a_mean[k_idx_0based]})

            if low_a_questions:
                msg = (
                    f"INFO: Found {len(low_a_questions)} questions with mean discrimination ('a') < {LOW_DISCRIMINATION_THRESHOLD}. "
                    "These questions may weakly distinguish capability levels. "
                    "Consider for revision/removal in future surveys:")
                print(msg)
                low_a_df = pd.DataFrame(low_a_questions).sort_values('Mean_a')
                print(low_a_df.round(3))
            # else: # No need to print OK if min is printed
            # print(f"  Low Discrimination (< {LOW_DISCRIMINATION_THRESHOLD}): None found.")
        else:
            print(
                f"  WARNING: Length mismatch between mean 'a' ({len(a_mean)}) and K ({K_dims}). Skipping low discrimination details.")
    else:
        print("  INFO: 'a' not found in samples. Skipping discrimination check.")

    # --- 5. Final Summary ---
    print("\n--- DIAGNOSTIC CHECKS COMPLETE ---")
    critical_warnings = [w for w in warnings_found if "CRITICAL" in w or "ERROR" in w]
    other_warnings = [w for w in warnings_found if "CRITICAL" not in w and "ERROR" not in w]

    if critical_warnings:
        print(
            f"CRITICAL ISSUES FOUND ({len(critical_warnings)}): Results likely unreliable. Please review warnings above.")
    elif other_warnings:
        print(f"Potential issues flagged ({len(other_warnings)}). Review warnings and perform manual checks.")
    else:
        print("Finished. No major potential issues flagged by automated checks.")
        print("Remember to perform manual visual checks (Trace Plots, PPC) and interpret results in context.")

    return warnings_found, summary  # Return summary for potential use


# -----------------------------------------
# Section 6: Results Extraction & Processing
# -----------------------------------------
print("\n--- Section 6: Extracting and Processing Results ---")

# Extract posterior samples (handle potential errors)
try:
    # Note: PyStan 2.19 extract is already permuted by default
    samples = fit.extract()
    print("Posterior samples extracted.")
except Exception as e:
    print(f"ERROR: Could not extract samples from fit object: {e}")
    print("Cannot proceed with results processing.")
    exit()

# --- Theta Estimates (Team Capabilities) ---
theta_df = None  # Initialize
if 'theta' in samples:
    theta_samples = samples['theta']  # Shape (n_samples, J, L)
    theta_means = theta_samples.mean(axis=0)  # shape (J, L)

    # Use final category names derived after question filtering
    theta_columns = [f"theta_{cat_idx_to_name_final[l + 1]}" for l in range(L_final)]
    theta_df = pd.DataFrame(theta_means, columns=theta_columns)

    # Add group info back using final group mapping
    theta_df['group_idx'] = list(group_idx_map.values())  # Ensure order matches J dimension
    theta_df['group'] = theta_df['group_idx'].map({v: k for k, v in group_idx_map.items()})
    # Handle potential splitting errors if group name format changes
    try:
        theta_df[[ID_VAR, YEAR_COL]] = theta_df['group'].str.split('|', expand=True)
        theta_df[YEAR_COL] = theta_df[YEAR_COL].astype(int)
    except Exception as e:
        print(f"Warning: Could not split 'group' column into {ID_VAR} and {YEAR_COL}. Error: {e}")
        # Keep group column, maybe add dummy ID/Year if needed later
        if ID_VAR not in theta_df: theta_df[ID_VAR] = 'Unknown'
        if YEAR_COL not in theta_df: theta_df[YEAR_COL] = 0

    # Reorder columns
    theta_df = theta_df[[ID_VAR, YEAR_COL, 'group', 'group_idx'] + theta_columns]

    print("\nEstimated Team Capabilities (Theta Posterior Means):")
    print(theta_df.head())
    theta_filepath = os.path.join(OUTPUT_DIR, 'team_capability_estimates.csv')
    theta_df.to_csv(theta_filepath, index=False)
    print(f"Team capability estimates saved to {theta_filepath}")
else:
    print("WARNING: 'theta' not found in samples. Cannot process capability estimates.")

# --- Correlation Matrix Estimate (Omega) ---
corr_df = None  # Initialize
if 'Omega' in samples:
    Omega_mean = samples['Omega'].mean(axis=0)
    category_names_final = list(cat_idx_to_name_final.values())
    corr_df = pd.DataFrame(Omega_mean, index=category_names_final, columns=category_names_final)
    print("\nEstimated Mean Correlation Matrix between Capabilities ('Omega'):")
    print(corr_df.round(2))
    corr_filepath = os.path.join(OUTPUT_DIR, 'capability_correlations.csv')
    corr_df.to_csv(corr_filepath)
    print(f"Capability correlation matrix saved to {corr_filepath}")
else:
    print("WARNING: 'Omega' not found in samples. Cannot process correlation matrix.")


# --- Ranking and Improvement Summary ---
# (Function definition moved here for clarity before use)
def create_ranking_summary(theta_df_in, id_var_in, year_col_in, cat_map_in, years_in):
    """Creates a summary DataFrame of rankings and improvements."""
    if theta_df_in is None or theta_df_in.empty:
        print("INFO: theta_df is empty, skipping ranking summary.")
        return pd.DataFrame()

    print(f"\nCalculating rankings and improvements between {min(years_in)} and {max(years_in)}...")
    summary_data = []
    unique_ids = theta_df_in[id_var_in].unique()
    years_sorted = sorted(years_in)

    # Compare first and last year provided
    year1, year_last = years_sorted[0], years_sorted[-1]

    df_y1 = theta_df_in[theta_df_in[year_col_in] == year1].set_index(id_var_in)
    df_y_last = theta_df_in[theta_df_in[year_col_in] == year_last].set_index(id_var_in)

    for cat_name in cat_map_in.keys():  # Use the final category map keys
        col_name = f"theta_{cat_name}"
        if col_name not in theta_df_in.columns:
            # print(f"Warning: Column {col_name} not found in theta_df. Skipping category {cat_name} for ranking.")
            continue  # Silently skip if column missing

        # Calculate ranks for the specific years being compared
        rank_y1 = df_y1[col_name].rank(ascending=False, method='min').astype(int)
        rank_y_last = df_y_last[col_name].rank(ascending=False, method='min').astype(int)

        for team_id in unique_ids:
            val_y1 = df_y1.loc[team_id, col_name] if team_id in df_y1.index else np.nan
            val_y_last = df_y_last.loc[team_id, col_name] if team_id in df_y_last.index else np.nan

            improvement = val_y_last - val_y1 if pd.notna(val_y1) and pd.notna(val_y_last) else np.nan

            r_y1 = rank_y1.get(team_id, np.nan)
            r_y_last = rank_y_last.get(team_id, np.nan)

            rank_change = r_y1 - r_y_last if pd.notna(r_y1) and pd.notna(r_y_last) else np.nan

            summary_data.append({
                'Team':                    team_id,
                'Category':                cat_name,
                f'Capability_{year1}':     val_y1,
                f'Capability_{year_last}': val_y_last,
                'Improvement':             improvement,
                f'Rank_{year1}':           r_y1,
                f'Rank_{year_last}':       r_y_last,
                'Rank_Change':             rank_change
            })

    summary_df = pd.DataFrame(summary_data)
    # Add overall rank average across categories (use with caution!)
    # summary_df['Avg_Rank_Last'] = summary_df.groupby('Team')[f'Rank_{year_last}'].transform('mean')
    # summary_df['Overall_Rank_Last'] = summary_df['Avg_Rank_Last'].rank(ascending=True, method='min').astype(int)
    return summary_df


ranking_summary = pd.DataFrame()  # Initialize
if theta_df is not None:
    actual_years_final = sorted(theta_df[YEAR_COL].unique())
    if len(actual_years_final) >= 2:
        ranking_summary = create_ranking_summary(
            theta_df_in=theta_df,
            id_var_in=ID_VAR,
            year_col_in=YEAR_COL,
            cat_map_in=category_mapping_final,  # Use final map
            years_in=actual_years_final  # Use actual years from data
        )
        print("\nTeam Rankings and Improvements by Category (Excerpt):")
        # Sort by last year's rank within category
        print(ranking_summary.sort_values(['Category', f'Rank_{actual_years_final[-1]}']).head(10))
        ranking_filepath = os.path.join(OUTPUT_DIR, 'team_rankings_by_category.csv')
        ranking_summary.to_csv(ranking_filepath, index=False, float_format='%.3f')
        print(f"Ranking summary saved to {ranking_filepath}")
    else:
        print(
            "INFO: Fewer than two years of data found in processed results. Cannot calculate improvement or rank change.")
else:
    print("INFO: Skipping ranking summary as theta_df could not be generated.")

# -----------------------------------------
# Section 7: Visualization Functions
# -----------------------------------------
print("\n--- Section 7: Defining Visualization Functions ---")


def plot_category_slope_chart(theta_df_viz, category_name_viz, id_var_viz, year_col_viz, years_viz, output_dir_viz):
    """Creates a slope chart for a specific category across two years, avoiding label overlap."""
    if theta_df_viz is None or theta_df_viz.empty:
        print(f"INFO: theta_df is empty, skipping slope chart for {category_name_viz}.")
        return None

    col_name = f"theta_{category_name_viz}"
    if col_name not in theta_df_viz.columns:
        print(f"INFO: Column {col_name} not found, skipping slope chart for {category_name_viz}.")
        return None

    if len(years_viz) != 2:
        print(f"INFO: Slope chart requires exactly two years. Skipping for {category_name_viz}.")
        return None
    year1, year2 = min(years_viz), max(years_viz)

    df_pivot = theta_df_viz[theta_df_viz[year_col_viz].isin([year1, year2])]
    df_pivot = df_pivot.pivot_table(index=id_var_viz, columns=year_col_viz, values=col_name)
    df_pivot.dropna(inplace=True)  # Only plot teams present in both years

    if df_pivot.empty:
        print(
            f"INFO: No teams present in both {year1} and {year2} for category {category_name_viz}. Skipping slope chart.")
        return None

    df_pivot = df_pivot.sort_values(by=year2)
    avg_improvement = (df_pivot[year2] - df_pivot[year1]).mean()

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_pivot) * 0.6)))  # Adjust height

    texts = []  # Store text objects for adjustText
    for team_id, row in df_pivot.iterrows():
        ax.plot([year1, year2], [row[year1], row[year2]], marker='o', linewidth=1.5, markersize=5, color='grey',
                alpha=0.7)
        texts.append(ax.text(year1 - 0.05, row[year1], team_id, horizontalalignment='right', verticalalignment='center',
                             fontsize=8))
        texts.append(ax.text(year2 + 0.05, row[year2], team_id, horizontalalignment='left', verticalalignment='center',
                             fontsize=8))

    # Use adjustText to prevent label overlap
    try:
        adjust_text(texts, ax=ax,  # Pass the axes object
                    # arrowprops=dict(arrowstyle='-', color='gray', lw=0.5) # Optional arrows
                    )
    except Exception as e:
        print(f"Warning: adjust_text failed for {category_name_viz}, labels might overlap. Error: {e}")

    ax.set_title(
        f'Team Capability: {category_name_viz} ({year1} vs {year2})\nAvg. Improvement (plotted teams): {avg_improvement:.3f}')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Estimated Capability ({category_name_viz})')
    ax.set_xticks([year1, year2])
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    # Adjust xlim slightly based on data range for labels
    ax.set_xlim(year1 - 0.3, year2 + 0.3)

    filepath = os.path.join(output_dir_viz, f'slope_chart_{category_name_viz}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(f"Slope chart saved for {category_name_viz} to {filepath}")
    return fig


def plot_reorg_slope_chart(theta_df_viz, category_name_viz, id_var_viz, year_col_viz, years_viz,
                           parent_to_child_map,  # Pass the mapping: Parent -> [Children]
                           output_dir_viz):
    """
    Creates a slope chart visualizing team evolution across a reorg.
    Plots all points and draws lines based on the parent->child mapping,
    coloring lines by the child (Year 2) team.
    """
    if theta_df_viz is None or theta_df_viz.empty:
        print(f"INFO: theta_df is empty, skipping reorg slope chart for {category_name_viz}.")
        return None

    col_name = f"theta_{category_name_viz}"
    if col_name not in theta_df_viz.columns:
        print(f"INFO: Column {col_name} not found, skipping reorg slope chart for {category_name_viz}.")
        return None

    if len(years_viz) != 2:
        print(f"INFO: Reorg slope chart requires exactly two years. Skipping for {category_name_viz}.")
        return None
    year1, year2 = min(years_viz), max(years_viz)

    # Get data for each year separately
    df_y1 = theta_df_viz[theta_df_viz[year_col_viz] == year1].set_index(id_var_viz)
    df_y2 = theta_df_viz[theta_df_viz[year_col_viz] == year2].set_index(id_var_viz)

    if df_y1.empty and df_y2.empty:
        print(f"INFO: No data found for either {year1} or {year2} for category {category_name_viz}. Skipping chart.")
        return None

    fig, ax = plt.subplots(figsize=(12, max(8, (len(df_y1) + len(df_y2)) * 0.4)))  # Adjust height

    texts = []  # Store text objects for adjustText
    plotted_points_y1 = {}  # Store coordinates for line drawing
    plotted_points_y2 = {}

    # --- Plot Points and Labels ---
    # Year 1 points
    for team_id, row in df_y1.iterrows():
        y1_val = row[col_name]
        if pd.notna(y1_val):
            ax.plot(year1, y1_val, 'o', color='black', markersize=6)  # Mark Y1 points
            texts.append(ax.text(year1 - 0.05, y1_val, team_id, horizontalalignment='right', verticalalignment='center',
                                 fontsize=8))
            plotted_points_y1[team_id] = y1_val

    # Year 2 points
    all_year2_teams = df_y2.index.unique().tolist()
    # Create a color map for Year 2 teams
    # Use a perceptually uniform colormap like 'viridis' or 'tab20' if many teams
    colors = cm.get_cmap('tab20', len(all_year2_teams))
    color_map_y2 = {team: colors(i) for i, team in enumerate(all_year2_teams)}

    for team_id, row in df_y2.iterrows():
        y2_val = row[col_name]
        if pd.notna(y2_val):
            point_color = color_map_y2.get(team_id, 'gray')  # Use mapped color
            ax.plot(year2, y2_val, 'o', color=point_color, markersize=6)  # Mark Y2 points with color
            texts.append(ax.text(year2 + 0.05, y2_val, team_id, horizontalalignment='left', verticalalignment='center',
                                 fontsize=8))
            plotted_points_y2[team_id] = y2_val

    # --- Plot Connecting Lines based on Mapping ---
    print(f"  Plotting reorg lines for {category_name_viz}...")
    lines_plotted = 0
    for parent_team, child_teams in parent_to_child_map.items():
        if parent_team in plotted_points_y1:
            y1_val = plotted_points_y1[parent_team]
            for child_team in child_teams:
                if child_team in plotted_points_y2:
                    y2_val = plotted_points_y2[child_team]
                    line_color = color_map_y2.get(child_team, 'gray')  # Color by child team
                    ax.plot([year1, year2], [y1_val, y2_val], linestyle='-', linewidth=1.0, color=line_color, alpha=0.6)
                    lines_plotted += 1

    print(f"    {lines_plotted} lineage lines plotted.")
    # Optional: Highlight teams only in Y1 or Y2 if needed (e.g., different marker)
    # --- Adjust Labels ---
    try:
        adjust_text(texts, ax=ax, force_points=(0.1, 0.2))  # Add some force to spread labels
    except Exception as e:
        print(f"Warning: adjust_text failed for {category_name_viz}, labels might overlap. Error: {e}")

    # --- Final Touches ---
    ax.set_title(
        f'Team Capability Evolution: {category_name_viz} ({year1} vs {year2})\n(Lines show team lineage, colored by Year 2 team)')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Estimated Capability ({category_name_viz})')
    ax.set_xticks([year1, year2])
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_xlim(year1 - 0.4, year2 + 0.4)  # Wider padding for labels

    # Create a simple legend (might get crowded if too many Y2 teams)
    if len(all_year2_teams) <= 20:  # Only show legend if manageable
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=team,
                              markerfacecolor=color_map_y2.get(team, 'gray'), markersize=6)
                   for team in all_year2_teams]
        ax.legend(handles=handles, title="Year 2 Teams", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.subplots_adjust(right=0.85)  # Make space for legend

    filepath = os.path.join(output_dir_viz, f'reorg_slope_chart_{category_name_viz}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print(f"Reorg slope chart saved for {category_name_viz} to {filepath}")
    return fig


def plot_omega_clustermap(corr_df_viz, output_dir_viz):
    """Visualizes the correlation matrix Omega as a clustered heatmap."""
    if corr_df_viz is None or corr_df_viz.empty:
        print("INFO: Correlation matrix DataFrame is empty. Skipping Omega clustermap.")
        return

    n_cats = corr_df_viz.shape[0]
    if n_cats < 2:
        print("INFO: Need at least 2 categories to create a clustermap. Skipping Omega plot.")
        return

    print("\n--- Generating Clustered Heatmap for Capability Correlations (Omega) ---")
    figsize = (max(8, n_cats * 0.8), max(7, n_cats * 0.7))
    annotate = n_cats <= 15

    try:
        cluster_grid = sns.clustermap(
            corr_df_viz,
            method='average',
            metric='euclidean',
            cmap='coolwarm',  # Blue=Positive, Red=Negative correlation
            vmin=-1, vmax=1,
            center=0,
            annot=annotate,
            fmt=".2f",
            linewidths=.5,
            linecolor='lightgray',
            figsize=figsize
        )
        plt.setp(cluster_grid.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
        cluster_grid.fig.suptitle('Clustered Heatmap of Capability Correlations (Mean Omega)', y=1.02)
        filepath = os.path.join(output_dir_viz, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print(f"Omega clustermap saved to {filepath}")

        print("\nInterpretation Notes for Omega Clustermap:")
        print("- Colors indicate correlation strength (Blue=positive, Red=negative).")
        print("- Dendrograms show hierarchical clustering of capabilities.")
        print("- Capabilities grouped closely together likely represent related concepts or influence each other.")
    except Exception as e:
        print(f"WARNING: Could not generate Omega clustermap: {e}")


# -----------------------------------------
# Section 8: Main Execution Flow & Reporting
# -----------------------------------------
print("\n--- Section 8: Running Main Analysis Flow ---")

# --- Perform Diagnostics ---
# Diagnostics are run *after* sampling is complete
diagnostic_warnings, arviz_summary = run_diagnostic_checks(
    fit=fit,
    samples=samples,
    stan_data=stan_data,
    df_long=df_long,  # Use the final processed df_long
    cat_idx_to_name_map=cat_idx_to_name_final,  # Use final map
    question_idx_map=question_idx_map,  # Use final map
    n_chains=CHAINS,
    iter_sampling=ITER_SAMPLING
)

# --- Generate Visualizations ---

# Generate reverse map (Parent -> Children) needed for plotting lines FROM parents
parent_to_child_mapping = {}
all_parent_teams = set(p for parents in reorg_mapping_y2_to_y1.values() for p in parents)
for parent in all_parent_teams:
    parent_to_child_mapping[parent] = [child for child, parents in reorg_mapping_y2_to_y1.items() if parent in parents]

# We also need the reverse mapping for plotting lines FROM parents
# Parent Team (Year 1) -> [List of Child Teams (Year 2)]
parent_to_child_mapping = {}
all_parent_teams = set(p for parents in reorg_mapping.values() for p in parents)
for parent in all_parent_teams:
    parent_to_child_mapping[parent] = [child for child, parents in reorg_mapping.items() if parent in parents]

print("\n--- Generating Visualizations ---")
if theta_df is not None:
    actual_years_final = sorted(theta_df[YEAR_COL].unique())
    if len(actual_years_final) >= 2:
        # Plot REORG slope charts for each category comparing first and last year
        years_to_plot = [actual_years_final[0], actual_years_final[-1]]
        print(f"\nGenerating slope charts comparing {years_to_plot[0]} and {years_to_plot[1]}...")
        for cat_name in category_mapping_final.keys():  # Use final keys
            # plot_category_slope_chart(theta_df, cat_name, ID_VAR, YEAR_COL, years_to_plot, OUTPUT_DIR)

            plot_reorg_slope_chart(
                theta_df_viz=theta_df,
                category_name_viz=cat_name,
                id_var_viz=ID_VAR,
                year_col_viz=YEAR_COL,
                years_viz=years_to_plot,
                parent_to_child_map=parent_to_child_mapping,  # Pass the map
                output_dir_viz=OUTPUT_DIR
            )
    else:
        print("INFO: Skipping slope charts as less than two years of data are available.")

    # Generate Omega clustermap
    if corr_df is not None:
        plot_omega_clustermap(corr_df, OUTPUT_DIR)
    else:
        print("INFO: Skipping Omega clustermap as corr_df was not generated.")
else:
    print("INFO: Skipping visualizations as theta_df could not be generated.")

# --- Optional: Compare Estimated Parameters to True Simulation Parameters ---
if RUN_SIMULATION and true_params is not None:
    print("\n--- Comparing Estimates to True Simulation Parameters (Optional) ---")
    # Compare Omega
    if 'Omega' in samples and 'true_Omega' in true_params:
        est_Omega_mean = samples['Omega'].mean(axis=0)
        omega_diff = np.mean(np.abs(est_Omega_mean - true_params['true_Omega']))
        print(f"  Mean Absolute Difference (Omega vs True Omega): {omega_diff:.3f}")

    # Compare mu
    if 'mu' in samples and 'true_mu' in true_params:
        est_mu_mean = samples['mu'].mean(axis=0)
        mu_diff = np.mean(np.abs(est_mu_mean - true_params['true_mu']))
        print(f"  Mean Absolute Difference (mu vs True mu): {mu_diff:.3f}")
        # print(f"    True mu: {np.round(true_params['true_mu'], 2)}")
        # print(f"    Est. mu: {np.round(est_mu_mean, 2)}")

    # Compare sigma
    if 'sigma' in samples and 'true_sigma' in true_params:
        est_sigma_mean = samples['sigma'].mean(axis=0)
        sigma_diff = np.mean(np.abs(est_sigma_mean - true_params['true_sigma']))
        print(f"  Mean Absolute Difference (sigma vs True sigma): {sigma_diff:.3f}")

    # Compare theta (more complex due to indexing and scale)
    # Can calculate correlation between true and estimated thetas per dimension

# --- Final Summary ---
print("\n--- Analysis Complete ---")
print(f"Results, diagnostics, and plots saved in: {OUTPUT_DIR}")

if any("CRITICAL" in w for w in diagnostic_warnings):
    print(
        "\n*** ACTION REQUIRED: Critical warnings detected during diagnostics. Review output carefully. Results may be unreliable. ***")
elif diagnostic_warnings:
    print("\nNOTE: Some potential issues were flagged during diagnostics. Review warnings and manual checks.")

print("\nReview Diagnostics Output and Manual Checks (Trace Plots, PPC) before interpreting results.")

# -----------------------------------------
# Section 9: DORA Specific Considerations (Commentary)
# -----------------------------------------
print("\n--- Section 9: DORA Survey Specific Considerations ---")
print("""
1.  Interpretation: The estimated 'theta' values represent the underlying capability 
    of each team in the defined DORA-related domains (e.g., Deployment, Reliability) 
    for a given year, based on the survey responses. Higher values indicate higher 
    perceived capability according to the model.

2.  Focus on Change: The 'Improvement' column in the ranking summary and the slope 
    charts are key outputs for tracking progress over time on these capabilities.

3.  NA Handling & Separate Analyses: As discussed, high NA rates for certain teams 
    (e.g., non-engineering roles) on specific questions likely require analyzing 
    subsets. Consider running:
    *   An 'Engineering Focused' analysis using only Eng teams and all relevant questions.
    *   A 'Core Practices' analysis using all teams but only questions applicable to everyone 
      (e.g., collaboration, process, learning). This script currently assumes one 
      combined analysis after filtering questions – adapt the data filtering in Section 2
      if separate analyses are needed.

4.  Linking to DORA Metrics: These survey-based capability scores are perceptions. 
    Correlate these `theta` estimates (especially changes) with objective DORA metrics 
    (Deployment Frequency, Lead Time, etc.) if available. Do teams reporting higher 
    'Deployment & Release' capability actually deploy more frequently?

5.  Actionability: Use the results to identify areas of strength and weakness across 
    teams or capability domains. Low scores or lack of improvement in critical areas 
    (e.g., System Reliability, Monitoring) should trigger deeper investigation and potential 
    interventions (training, tooling, process changes). Low discrimination questions 
    should be reviewed for clarity and relevance.

6.  Compositional Data (Work Breakdown): Analyze the percentage-based questions 
    separately using compositional data techniques (descriptives, potentially ILR 
    transformations). Explore correlations between work patterns (e.g., time spent on 
    toil vs. coding) and the estimated `theta` capabilities or objective DORA metrics 
    as an exploratory step.

7.  Global Ranking: Avoid simple averaging of ranks. Use weighted averages of 
    (standardized) `theta` scores based on strategic importance, or define performance 
    tiers based on multiple capabilities. The most value often lies in the detailed 
    category-specific insights.
""")
