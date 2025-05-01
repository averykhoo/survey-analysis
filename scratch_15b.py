# -*- coding: utf-8 -*-
"""
Comprehensive MGRM Analysis Script for DORA Survey Data (PyStan 2.19)

Purpose: Analyze Likert-scale DORA survey responses using a Bayesian
         Multidimensional Graded Response Model (MGRM) with correlated latent traits.
         Estimates latent capabilities (theta) for engineering teams over time,
         handles team reorganizations in visualization (using standardized scores),
         assesses improvement, estimates correlations between capabilities,
         plots item-level diagnostics (ICC/KDE, CRF, IIF), Test Information (TIF)
         and performs rigorous diagnostic checks.

Structure:
    0. Imports & Global Settings (incl. Reorg Map)
    1. Simulate Sample Data (Reorg Aware, Optional)
    2. Data Loading & Preprocessing (Incl. Validation & Filtering)
    3. Stan Model Definition (MGRM, PyStan 2.19 Compatible)
    4. Smart Initial Values & Stan Execution
    5. Diagnostic Checks Function
    6. Results Extraction & Processing (Incl. Mu/Sigma Map Creation)
    7. Visualization Functions (Standardized Slope, Omega, Item/Test Info, ICC/KDE, CRF)
    8. Main Execution Flow & Reporting (Incl. Calling New Plots & Optional Parameter Recovery)
    9. DORA Specific Considerations (Commentary)



TODO:
* Check if any sigma is much greater than the answer range (e.g., 1000+, given answers from 1-6)
* Check for data problems (excessive outliers) if nu turns out to be 1 when using a parameterized student's t distribution 
* Add a comment somewhere that pystan automatically truncates the distribution to a half normal, which is why there's no `half_normal` in pystan (at least not in 2.19)
* Visualize `arviz.plot_loo_pit()` too
"""

import os
import re

import arviz as az
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan  # Ensure v2.19 is installed for compatibility
import scipy.special as sps
import seaborn as sns

try:
    from adjustText import adjust_text  # pip install adjustText

    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    print("WARNING: 'adjustText' library not found. Install for better label placement (pip install adjustText).")
    ADJUST_TEXT_AVAILABLE = False

# -----------------------------------------
# Section 0: Imports & Global Settings
# -----------------------------------------
print("--- Section 0: Loading Libraries & Settings ---")

# --- Display versions ---
print(f"Using pystan version: {pystan.__version__}")
print(f"Using arviz version: {az.__version__}")
print(f"Using pandas version: {pd.__version__}")
print(f"Using numpy version: {np.__version__}")

# --- MCMC Settings ---
ITER_WARMUP = 1000
ITER_SAMPLING = 1500  # Consider increasing if n_eff is low
CHAINS = 4  # Number of parallel chains
THIN = 1  # Thinning factor (usually 1 is fine)
RANDOM_SEED = 42  # For reproducibility

# --- Diagnostic Thresholds ---
RHAT_THRESHOLD = 1.05  # Max acceptable R-hat for convergence
NEFF_RATIO_THRESHOLD = 0.1  # Min effective samples as fraction of post-warmup samples
PARETO_K_THRESHOLD = 0.7  # Threshold for problematic Pareto k in LOOIC
HIGH_CORR_THRESHOLD = 0.85  # Correlation threshold for suggesting category merge
LOW_DISCRIMINATION_THRESHOLD = 0.3  # Threshold for flagging weak questions
MIN_RESPONSES_PER_GROUP = 5  # Min non-NA responses per team-year to avoid warning

# --- File Paths & Output ---
OUTPUT_DIR = 'dora_analysis_output_reorg'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- Stan Compiler Flags for PyStan 2.19 ---
# Suppress specific common warnings during compilation
STAN_COMPILE_FLAGS = [
    '-Wno-int-in-bool-context',
    '-Wno-unused-local-typedef',
    '-Wno-misleading-indentation',
    '-Wno-deprecated-declarations'
]

# --- Core Survey Definition ---
# These should be defined based on your actual survey structure
YEAR_COL = 'year'  # Column name for the survey year
ID_VAR = 'team_id'  # Column name for the team identifier

# Define response options (Likert scale) - MODIFY IF YOUR SCALE IS DIFFERENT
RESPONSE_OPTIONS = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE = len(RESPONSE_OPTIONS)
print(f"Response scale set to: {RESPONSE_OPTIONS} ({N_CATEGORIES_RESPONSE} categories)")

# Define categories (DORA domains/capabilities) and their associated questions
# !!! MODIFY FOR YOUR ACTUAL SURVEY QUESTIONS AND CATEGORIES !!!
# This dictionary represents a key theoretical assumption of the model.
category_mapping_initial = {
    'Deployment & Release':       ['q1', 'q2', 'q3'],
    'Monitoring & Observability': ['q4', 'q5'],
    'Technical Practices':        ['q6', 'q7', 'q8'],
    'Team Collaboration':         ['q9', 'q10'],
    'Process Efficiency':         ['q11'],
    'Learning & Development':     ['q12', 'q13'],
    'System Reliability':         ['q14', 'q15'],
    'Change Management':          ['q16']
}

# --- Reorganization Mapping ---
# Defines how teams evolve between the *first* and *last* year being analyzed.
# Format: { Child_Team_Name (Last Year): [List of Parent_Team_Names (First Year)] }
# If no reorg, map each team to itself: {'TeamA': ['TeamA'], 'TeamB': ['TeamB'], ...}
# !!! MODIFY THIS WITH YOUR ACTUAL REORG MAPPING based on the first/last years in your data !!!
REORG_MAPPING_Y_LAST_TO_Y1 = {
    'Team_X': ['Team_A', 'Team_B'],
    'Team_Y': ['Team_B', 'Team_C'],
    'Team_Z': ['Team_A', 'Team_C', 'Team_D'],
    'Team_E': ['Team_E'],  # Unchanged team
    'Team_F': ['Team_F'],  # Unchanged team
    'Team_G': []  # New team in last year
}

# --- Derive Parent -> [Children] mapping (needed for plotting) ---
PARENT_TO_CHILD_MAPPING = {}
all_parent_teams_in_map = set(p for parents in REORG_MAPPING_Y_LAST_TO_Y1.values() for p in parents)
all_child_teams_in_map = set(REORG_MAPPING_Y_LAST_TO_Y1.keys())
# Map parents to children
for parent in all_parent_teams_in_map:
    PARENT_TO_CHILD_MAPPING[parent] = [child for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items() if
                                       parent in parents]
# Ensure self-mapping teams are included if they weren't parents
for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items():
    if len(parents) == 1 and parents[0] == child and child not in PARENT_TO_CHILD_MAPPING:
        PARENT_TO_CHILD_MAPPING[child] = [child]

# Determine the universe of teams involved based *only* on the mapping provided
# This ensures simulation uses the teams defined in the mapping structure
# all_teams_in_reorg_map = all_parent_teams_in_map.union(all_child_teams_in_map)
# print(f"Reorg map defined involving {len(all_teams_in_reorg_map)} unique team names.")
if not REORG_MAPPING_Y_LAST_TO_Y1:
    print("WARNING: Reorg mapping is empty. Visualization will assume no reorgs.")

# --- Derive initial helper maps (will be refined after data loading) ---
question_to_cat_name_initial = {q: cat_name for cat_name, qs in category_mapping_initial.items() for q in qs}
N_LATENT_initial = len(category_mapping_initial)


# -----------------------------------------
# Section 1: Simulate Sample Data (Reorg Aware, Optional)
# -----------------------------------------
def simulate_dora_data_reorg(sim_years_reorg,  # Should be exactly 2 years
                             reorg_map_child_to_parents,  # From global settings
                             sim_n_resp_per_team_year,
                             category_map_sim, response_options_sim):
    """
    Generates simulated DORA survey data reflecting a reorg between two years.
    Calculates Year 2 traits based on Year 1 parents + improvement.
    Returns df_raw and true parameters used for simulation.

    Args:
        sim_years_reorg (list): List containing exactly two years [year1, year2].
        reorg_map_child_to_parents (dict): Map {child_y2: [parent_y1,...]}.
        sim_n_resp_per_team_year (int): Number of responses per team-year to simulate.
        category_map_sim (dict): Category mapping to use for simulation.
        response_options_sim (list): List of possible response values.

    Returns:
        tuple: (pd.DataFrame, dict) containing the simulated data and true parameters.
    """
    print("--- Section 1: Simulating Sample DORA Data with Reorg (for testing) ---")

    if len(sim_years_reorg) != 2:
        raise ValueError("Reorg simulation logic currently requires exactly two years.")
    year1, year2 = sim_years_reorg[0], sim_years_reorg[1]

    # Determine team lists based ONLY on the provided reorg map
    year1_teams_sim = sorted(list(set(p for parents in reorg_map_child_to_parents.values() for p in parents)))
    year2_teams_sim = sorted(list(reorg_map_child_to_parents.keys()))
    print(f"  Simulating for Year 1 Teams: {year1_teams_sim}")
    print(f"  Simulating for Year 2 Teams: {year2_teams_sim}")

    np.random.seed(RANDOM_SEED)
    n_cats_response_sim = len(response_options_sim)
    n_latent_sim = len(category_map_sim)
    questions_sim = [q for qs in category_map_sim.values() for q in qs]
    question_categories_sim = {q: idx for idx, (cat, qs) in enumerate(category_map_sim.items(), 1) for q in qs}

    # Simulate correlation matrix (Omega)
    base_corr = 0.4
    corr_matrix = np.ones((n_latent_sim, n_latent_sim)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = np.random.uniform(-0.2, 0.2, (n_latent_sim, n_latent_sim))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    true_Omega = np.clip(corr_matrix + noise, 0.05, 0.95)
    np.fill_diagonal(true_Omega, 1.0)

    # Simulate true means (mu) and SDs (sigma)
    true_mu = np.random.normal(0, 0.2, n_latent_sim)
    true_sigma = np.random.lognormal(-0.5, 0.2, n_latent_sim)
    cov_matrix = np.diag(true_sigma) @ true_Omega @ np.diag(true_sigma)

    def generate_correlated_traits(mean_vector, cov_matrix):
        """Helper to generate MVN samples."""
        return np.random.multivariate_normal(mean_vector, cov_matrix)

    # Simulate Year 1 Traits
    true_latent_y1 = {}
    latent_for_resp = {}  # {(team, year, cat_idx): theta_val}
    for team in year1_teams_sim:
        team_base_mean_offset = np.random.normal(0, 0.4)
        mean_vector = true_mu + team_base_mean_offset
        traits_y1 = generate_correlated_traits(mean_vector, cov_matrix)
        true_latent_y1[team] = traits_y1
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year1, cat_idx)] = traits_y1[cat_idx - 1]

    # Simulate Year 2 Traits based on Reorg
    true_latent_y2 = {}
    for child_team in year2_teams_sim:
        parent_teams = reorg_map_child_to_parents.get(child_team, [])
        if not parent_teams:  # New team
            team_base_mean_offset = np.random.normal(0, 0.4)
            mean_vector = true_mu + team_base_mean_offset
            traits_y2 = generate_correlated_traits(mean_vector, cov_matrix)
        else:  # Reorged or unchanged team
            parent_thetas = [true_latent_y1.get(p_team) for p_team in parent_teams if p_team in true_latent_y1]
            if not parent_thetas:
                print(f"    Warning: No valid parent thetas found for {child_team}. Simulating as new.")
                team_base_mean_offset = np.random.normal(0, 0.4)
                mean_vector = true_mu + team_base_mean_offset
                traits_y2 = generate_correlated_traits(mean_vector, cov_matrix)
            else:
                avg_parent_theta = np.mean(np.array(parent_thetas), axis=0)
                category_improvement_mean = 0.4
                improvements = np.random.normal(category_improvement_mean, 0.2, n_latent_sim) + np.random.normal(0,
                                                                                                                 0.15)
                traits_y2 = avg_parent_theta + improvements
        true_latent_y2[child_team] = traits_y2
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(child_team, year2, cat_idx)] = traits_y2[cat_idx - 1]

    true_latent_combined = {(team, year1): traits for team, traits in true_latent_y1.items()}
    true_latent_combined.update({(team, year2): traits for team, traits in true_latent_y2.items()})

    # Simulate Item Parameters
    true_a = {q: np.random.lognormal(-0.1, 0.4) for q in questions_sim}
    true_cutpoints = {}
    for q in questions_sim:
        item_difficulty = np.random.normal(0, 0.7)
        num_cutpoints = n_cats_response_sim - 1
        spacings = np.maximum(np.random.lognormal(0, 0.3, num_cutpoints), 0.3)
        raw_cutpoints = np.cumsum(spacings)
        centered_cutpoints = raw_cutpoints - np.mean(raw_cutpoints) + item_difficulty
        true_cutpoints[q] = np.sort(centered_cutpoints)

    # Simulate Responses function (GRM logic)
    def simulate_response_gr(theta_val, a, cutpoints, response_options_sim):
        """Simulates ordinal response using Graded Response Model logic."""
        eta = a * theta_val
        cum_prob_le = sps.expit(cutpoints - eta)
        n_cats_sim = len(response_options_sim)
        probs = np.zeros(n_cats_sim)
        probs[0] = cum_prob_le[0]
        for k in range(1, n_cats_sim - 1):
            probs[k] = cum_prob_le[k] - cum_prob_le[k - 1]
        probs[n_cats_sim - 1] = 1.0 - cum_prob_le[n_cats_sim - 2]
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()
        return np.random.choice(response_options_sim, p=probs)

    # Generate DataFrame rows
    data = []
    for year in sim_years_reorg:
        teams_in_year = year1_teams_sim if year == year1 else year2_teams_sim
        for team in teams_in_year:
            if (team, year) not in true_latent_combined:
                continue  # Skip if team doesn't exist in this year

            for i in range(sim_n_resp_per_team_year):
                row = {YEAR_COL: year, ID_VAR: team}
                for q in questions_sim:
                    cat = question_categories_sim[q]
                    # Ensure theta exists for this team, year, cat before accessing
                    theta_key = (team, year, cat)
                    if theta_key in latent_for_resp:
                        theta_true = latent_for_resp[theta_key]
                        a_val = true_a[q]
                        cp_val = true_cutpoints[q]
                        # Simulate some NAs
                        if np.random.rand() < 0.05:
                            row[q] = np.nan
                        else:
                            row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
                    else:
                        # This case should ideally not happen with correct logic above
                        # print(f"Warning: Latent trait not found for {theta_key}")
                        row[q] = np.nan  # Assign NA if theta lookup fails
                data.append(row)

    df = pd.DataFrame(data)
    print(f"Simulated data generated with {len(df)} rows reflecting reorg.")
    print("Simulated Sample Data (first 5 rows):")
    print(df.head())

    true_params = {'true_a':      true_a, 'true_cutpoints': true_cutpoints, 'true_Omega': true_Omega,
                   'true_latent': true_latent_combined, 'true_mu': true_mu, 'true_sigma': true_sigma}
    return df, true_params


# --- Flag to control simulation ---
# Set RUN_SIMULATION to False when you want to load your real data in Section 2
RUN_SIMULATION = True

# --- Run Simulation (if flag is True) ---
df_raw = None
true_params = None
if RUN_SIMULATION:
    # Use global REORG_MAPPING_Y_LAST_TO_Y1 defined in Section 0 for simulation
    sim_years_list = [2023, 2024]  # Must be 2 years for reorg sim logic
    sim_n_resp = 20
    if not REORG_MAPPING_Y_LAST_TO_Y1:
        print("ERROR: Simulation requires a non-empty REORG_MAPPING_Y_LAST_TO_Y1 in Section 0.")
        exit()
    df_raw, true_params = simulate_dora_data_reorg(
        sim_years_reorg=sim_years_list,
        # all_teams_involved=all_teams_in_reorg_map, # Use teams derived from map
        reorg_map_child_to_parents=REORG_MAPPING_Y_LAST_TO_Y1,  # Pass the map
        sim_n_resp_per_team_year=sim_n_resp,
        category_map_sim=category_mapping_initial,  # Use initial map for sim consistency
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
    print("Loading real data...")
    try:
        # Replace with your actual loading mechanism for your DORA survey data
        # df_raw = pd.read_csv('your_dora_survey_data.csv')
        print("ERROR: Real data loading not implemented. Please load data into df_raw.")
        exit()
    except FileNotFoundError:
        print("ERROR: Real data file not found. Cannot proceed.")
        exit()
    except Exception as e:
        print(f"ERROR loading real data: {e}")
        exit()
    # --- Perform any initial cleaning specific to your real data ---
    # E.g., renaming columns, handling date formats, filtering irrelevant rows

# --- Derive Data-Specific Variables ---
try:
    actual_years = sorted(df_raw[YEAR_COL].unique())
    actual_teams = df_raw[ID_VAR].unique()
    actual_questions_in_data = df_raw.columns.intersection(question_to_cat_name_initial.keys())
except KeyError as e:
    print(f"ERROR: Missing expected column in data: {e}. Check YEAR_COL ('{YEAR_COL}') and ID_VAR ('{ID_VAR}').")
    exit()
except Exception as e:
    print(f"ERROR deriving initial variables from data: {e}")
    exit()

print(f"Data contains years: {actual_years}")
if not actual_years:
    print("ERROR: No year information found.")
    exit()
print(f"Data contains {len(actual_teams)} unique teams.")
print(f"Found {len(actual_questions_in_data)} potentially relevant questions in the data.")

# --- Filter Questions: Keep only questions with valid responses across ALL analyzed years ---
print(
    f"Filtering questions: Keeping only those consistently present with non-NA responses across ALL years: {actual_years}")
questions_to_use = set(actual_questions_in_data)
if len(actual_years) > 1:
    for year in actual_years:
        df_year = df_raw[df_raw[YEAR_COL] == year]
        if df_year.empty:
            print(f"WARNING: No data found for year {year}. This year will be ignored in filtering.")
            continue
        valid_in_year = df_year[actual_questions_in_data].dropna(axis=1, how='all').columns
        questions_to_use.intersection_update(valid_in_year)
questions_to_use = sorted(list(questions_to_use))
questions_dropped = sorted(list(set(actual_questions_in_data) - set(questions_to_use)))

if questions_dropped:
    print(
        f"WARNING: Dropping {len(questions_dropped)} questions not present/valid across all analyzed years ({actual_years}):")
    print(f"  Dropped: {questions_dropped}")
if not questions_to_use:
    print("ERROR: No questions found valid across all specified years.")
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
    print("ERROR: No categories remaining after filtering questions.")
    exit()
cat_idx_to_name_final = {idx: name for idx, name in enumerate(final_cat_names, 1)}
N_LATENT_final = len(category_mapping_final)
print(f"Using {N_LATENT_final} categories based on available questions: {final_cat_names}")

# --- Melt Data using only questions_to_use ---
print("Reshaping data to long format...")
df_long = df_raw.melt(id_vars=[YEAR_COL, ID_VAR], value_vars=questions_to_use,
                      var_name='question', value_name='response')

# --- Handle Missing / NA Responses ---
initial_rows = len(df_long)
df_long.dropna(subset=['response'], inplace=True)
rows_after_na = len(df_long)
print(f"Removed {initial_rows - rows_after_na} rows with NA responses in the 'response' column.")
print("INFO: If NA rates differ systematically (e.g., non-eng teams on eng questions), "
      "filtering NAs might bias results. Consider separate analyses (Section 9).")

# --- Validate Response Values & Convert Type ---
try:
    # Attempt conversion, coercing errors to NaN
    df_long['response_numeric'] = pd.to_numeric(df_long['response'], errors='coerce')
    rows_before_num_drop = len(df_long)
    df_long.dropna(subset=['response_numeric'], inplace=True)
    if len(df_long) < rows_before_num_drop:
        print(f"WARNING: Removed {rows_before_num_drop - len(df_long)} non-numeric response rows.")
    df_long['response'] = df_long['response_numeric'].astype(int)

    # Check if responses are within the allowed range
    is_valid_response = df_long['response'].isin(RESPONSE_OPTIONS)
    invalid_responses = df_long[~is_valid_response]['response'].unique()
    if not is_valid_response.all():
        print(f"WARNING: Responses outside range {RESPONSE_OPTIONS} found: {invalid_responses}. Filtering.")
        df_long = df_long[is_valid_response]
    print("Response column validated and converted to integer.")
    df_long.drop(columns=['response_numeric'], inplace=True)
except Exception as e:
    print(f"ERROR during response validation: {e}")
    exit()
if df_long.empty:
    print("ERROR: No valid observations remaining.")
    exit()

# --- Add final category index ---
df_long['q_cat_final'] = df_long['question'].map(question_categories_final)

# --- Create Indices for Stan (1-based) ---
df_long['group'] = df_long[ID_VAR].astype(str) + "|" + df_long[YEAR_COL].astype(str)
groups_final = sorted(df_long['group'].unique())
group_idx_map = {g: i + 1 for i, g in enumerate(groups_final)}
df_long['group_idx'] = df_long['group'].map(group_idx_map)
group_idx_to_info = {idx: {"team": g.split('|')[0], "year": int(g.split('|')[1])} for g, idx in group_idx_map.items()}

question_idx_map = {q: i + 1 for i, q in enumerate(questions_to_use)}
df_long['question_idx'] = df_long['question'].map(question_idx_map)

# Map from Stan question index (1..K) to Stan dimension index (1..L)
q_idx_to_dim_idx = {q_idx_stan: question_categories_final[q] for q, q_idx_stan in question_idx_map.items()}
question_to_dimension_stan = [q_idx_to_dim_idx[k] for k in range(1, len(question_idx_map) + 1)]

# --- Data Validation Checks ---
print("Performing data validation checks...")
# Check for teams with very few responses per year
responses_per_group = df_long.groupby('group').size()
low_response_groups = responses_per_group[responses_per_group < MIN_RESPONSES_PER_GROUP]
if not low_response_groups.empty:
    print(f"WARNING: {len(low_response_groups)} team-year groups have < {MIN_RESPONSES_PER_GROUP} valid responses.")
    print("  Theta estimates for these groups will have high uncertainty.")

# Check for teams completely dropped
teams_in_processed_data = df_long[ID_VAR].unique()
teams_dropped_completely = sorted(list(set(actual_teams) - set(teams_in_processed_data)))
if teams_dropped_completely:
    print(f"WARNING: {len(teams_dropped_completely)} teams from raw data have NO valid responses "
          f"after filtering/NA removal. EXCLUDED Teams: {teams_dropped_completely}")

# --- Final Data Counts for Stan ---
N_final = df_long.shape[0]
J_final = len(groups_final)  # Number of unique team-year groups
K_final = len(questions_to_use)  # Number of unique questions included
C_final = N_CATEGORIES_RESPONSE  # Number of response categories (remains constant)
L_final = N_LATENT_final  # Number of latent dimensions included
print(f"\nFinal Data Counts for Stan: N={N_final}, J={J_final}, K={K_final}, C={C_final}, L={L_final}")
if N_final * J_final * K_final * L_final == 0:
    print("ERROR: Zero dimension detected.")
    exit()
print("\nPreprocessed Data Sample (first 5 rows):")
print(df_long.head())

# -----------------------------------------
# Section 3: Stan Model Definition (MGRM, PyStan 2.19 Compatible)
# -----------------------------------------
print("\n--- Section 3: Defining Stan Model ---")
# MGRM Model Code for Stan (PyStan 2.19 syntax)
# Includes prior sensitivity check alternatives as comments
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
  matrix[J, L] theta;         // Actual latent traits (team capabilities) [J x L]

  // Calculate the actual latent traits (theta) for each group and dimension
  // theta = mu + diag(sigma) * L_Omega * z
  theta_raw = diag_pre_multiply(sigma, L_Omega) * z;
  // Reconstruct theta[J, L] from theta_raw[L, J] and mu[L]
  for (j in 1:J) {
    for (l in 1:L) {
      theta[j, l] = mu[l] + theta_raw[l, j];
    }
  }
}

model {
  // --- Priors ---
  // These specify our beliefs about the parameters before seeing the data.
  // They help regularize the model and ensure proper posterior distributions.

  // Non-centered latent traits (standard normal prior is standard practice)
  to_vector(z) ~ std_normal();

  // Hierarchical priors for theta distribution (means and standard deviations of capabilities)
  mu ~ normal(0, 1);        // Prior for overall mean of each capability dimension.
                          // Assumes capabilities are centered around 0 on latent scale.
                          // Sensitivity Check: normal(0, 0.5) or normal(0, 2)
                          // Consider also student_t(7, 0, 1), and perhaps parameterize nu>=1 as an integer

  sigma ~ normal(0, 1);     // Prior for variability of each capability across teams/years.
                          // Constrained positive. normal(0,1) is weakly informative.
                          // Sensitivity Check 1: sigma ~ cauchy(0, 2.5); // Heavier tails
                          // Sensitivity Check 2: sigma ~ exponential(1); // Mean 1

  // Prior for correlation structure between capabilities
  L_Omega ~ lkj_corr_cholesky(2); // LKJ prior on Cholesky factor. eta=2 weakly favors
                                  // lower correlations (identity matrix) but allows strong ones.
                                  // Sensitivity Check 1: L_Omega ~ lkj_corr_cholesky(1); // Uniform over correlation matrices
                                  // Sensitivity Check 2: L_Omega ~ lkj_corr_cholesky(4); // Stronger push towards zero correlation

  // Priors for question parameters
  a ~ lognormal(0, 0.5);    // Prior for discrimination 'a'. Must be positive.
                          // lognormal(0, 0.5) keeps most values moderate (e.g., 0.5-2.0).
                          // Sensitivity Check 1: a ~ lognormal(0, 1);   // More variable 'a' allowed
                          // Sensitivity Check 2: a ~ normal(0, 1); // Actually half-normal - alternative positive prior

  // Prior for cutpoints (thresholds between response categories on latent scale)
  for (k in 1:K) {
      cutpoints[k] ~ normal(0, 3); // Prior for location of cutpoints for question k.
                                   // Fairly weak prior. Assumes average difficulty is near 0.
                                   // Sensitivity Check: cutpoints[k] ~ normal(0, 1.5); // Tighter prior
  }

  // --- Likelihood ---
  // This defines how the observed data 'y' depends on the parameters.
  // It uses the Graded Response Model logic via Stan's ordered_logistic distribution.
  for (n in 1:N) {
    // Declare local variables used in the loop (good practice in Stan)
    int current_q;       // Index for the question (1..K)
    int current_g;       // Index for the group (team-year) (1..J)
    int current_dim;     // Index for the latent dimension this question measures (1..L)
    real eta;             // Linear predictor for the ordered logistic model

    current_q = q_idx[n];       // Which question? (1..K)
    current_g = group_idx[n];   // Which group (team-year)? (1..J)
    current_dim = question_to_dimension[current_q]; // Which dimension does this question load on? (1..L)

    // Calculate linear predictor: eta = discrimination * team_capability
    // The capability score is indexed by the group and the relevant dimension.
    eta = a[current_q] * theta[current_g, current_dim];

    // Calculate likelihood of the observed response y[n] (1..C) given eta
    // and the specific cutpoints for this question.
    y[n] ~ ordered_logistic(eta, cutpoints[current_q]);
  }
}

generated quantities {
  // --- Post-processing Calculations ---
  // These are calculated for each posterior sample after fitting the model.

  // Declarations must be at the top
  matrix[L, L] Omega; // Estimated correlation matrix between DORA capabilities
  vector[N] log_lik;  // Log-likelihood for each observation (for LOOIC/WAIC)

  // Recover the full correlation matrix from its Cholesky factor
  Omega = multiply_lower_tri_self_transpose(L_Omega);

  // Calculate log likelihood for each observation
  // This is needed for model comparison criteria like LOOIC.
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

    // Calculate the log probability mass function (lpmf) for the observed response
    log_lik[n] = ordered_logistic_lpmf(y[n] | eta, cutpoints[current_q]);
  }
}
"""
print("Stan model code defined.")

# -----------------------------------------
# Section 4: Smart Initial Values & Stan Execution
# -----------------------------------------
print("\n--- Section 4: Defining Smart Inits & Running Stan ---")


def generate_smart_init(df_long_processed, K_val, L_val, J_val, C_val, question_idx_map_val, response_options_val,
                        chain_id):
    """
    Generates smarter initial values for Stan MCMC chains, using data statistics for cutpoints.

    Args:
        df_long_processed (pd.DataFrame): Processed long-format data.
        K_val (int): Number of questions.
        L_val (int): Number of latent dimensions.
        J_val (int): Number of groups.
        C_val (int): Number of response categories.
        question_idx_map_val (dict): Map from question ID to Stan index (1-based).
        response_options_val (list): List of possible response values.
        chain_id (int): Identifier for the current chain (used for random seed).

    Returns:
        dict: Dictionary of initial values for one Stan chain.
    """
    print(f"  Generating smart initial values for chain {chain_id}...")
    # Use chain_id to vary random seed for initialization noise
    np.random.seed(RANDOM_SEED + chain_id)

    initial_cutpoints = np.zeros((K_val, C_val - 1))
    # Estimate empirical cutpoints (latent scale) based on response quantiles per question
    responses_per_question = df_long_processed.groupby('question_idx')['response']
    num_cutpoints = C_val - 1
    probs = np.linspace(0, 1, C_val + 1)[1:-1]  # Probabilities for C-1 cutoffs (e.g., P(y<=1), P(y<=2),...)

    for k_idx_stan in range(1, K_val + 1):
        # Use median response as a rough center for this question's latent difficulty
        q_median = np.median(df_long_processed['response'])  # Default global median
        if k_idx_stan in responses_per_question.groups:
            q_responses = responses_per_question.get_group(k_idx_stan)
            if len(q_responses) > 1:
                q_median = np.median(q_responses)

        # Estimate empirical cutpoints centered around 0 on latent scale
        try:
            q_quantiles = np.quantile(df_long_processed['response'], probs)  # Use global quantiles for stability
            # Map response quantiles to latent scale centered approx 0
            latent_approx = np.interp(q_quantiles,
                                      [min(response_options_val), max(response_options_val)],
                                      [-2.0, 2.0])  # Map response range to latent range
            # Shift relative to question median mapped to latent scale (approx 0)
            q_cuts = latent_approx - np.interp(q_median, [min(response_options_val), max(response_options_val)],
                                               [-2.0, 2.0])
            q_cuts = np.sort(q_cuts)  # Ensure order

            # Ensure minimum spacing
            min_diff = 0.15
            for i in range(num_cutpoints - 1):
                q_cuts[i + 1] = max(q_cuts[i + 1], q_cuts[i] + min_diff)
            initial_cutpoints[k_idx_stan - 1, :] = q_cuts + np.random.normal(0, 0.1)  # Add noise
        except Exception:  # Fallback if quantile calculation fails
            default_cuts = np.linspace(-1.5, 1.5, num_cutpoints)
            initial_cutpoints[k_idx_stan - 1, :] = default_cuts + np.random.normal(0, 0.1, num_cutpoints)

    # Other parameters initialized around plausible defaults with noise
    # init = {
    #     'mu':        np.random.normal(0, 0.1, L_val),
    #     'sigma':     np.abs(np.random.normal(0.8, 0.1, L_val)),  # Start around 0.8
    #     'a':         np.abs(np.random.normal(1.0, 0.1, K_val)),  # Start around 1.0
    #     'z':         np.random.normal(0, 0.2, (L_val, J_val)),  # Slightly larger noise maybe
    #     'cutpoints': initial_cutpoints,
    #     # Start L_Omega near identity (no correlation) - Cholesky factor is Identity matrix
    #     'L_Omega':   np.eye(L_val)
    # }
    init = {
        'mu':        np.random.normal(0, 0.2, L_val),
        'sigma':     np.abs(np.random.normal(0.8, 0.2, L_val)),  # Start constrained positive
        'a':         np.abs(np.random.normal(1.0, 0.2, K_val)),  # Start constrained positive
        'z':         np.random.normal(0, 0.3, (L_val, J_val)),  # Init non-centered params
        'cutpoints': initial_cutpoints,  # Use data-informed cutpoints
        'L_Omega':   np.eye(L_val)  # Start with no correlation
    }
    # Small check for NaN/Inf
    for key, val in init.items():
        if isinstance(val, np.ndarray) and not np.all(np.isfinite(val)):
            print(f"WARNING: Non-finite values found in initial '{key}' for chain {chain_id}. Check data/init logic.")
            # Replace with default if problematic (example for 'a')
            if key == 'a':
                init[key] = np.ones(K_val)
            if key == 'cutpoints':
                init[key] = np.array(
                    [np.sort(np.linspace(-1.5, 1.5, C_val - 1)) for _ in range(K_val)])
    return init


# --- Prepare Stan data dictionary ---
# Ensure all data passed to Stan uses the final dimensions (N_final, J_final, etc.)
stan_data = {
    'N':                     N_final,
    'J':                     J_final,
    'K':                     K_final,
    'C':                     C_final,
    'L':                     L_final,
    'group_idx':             df_long['group_idx'].values.astype(int),  # Ensure integer type
    'q_idx':                 df_long['question_idx'].values.astype(int),
    'y':                     df_long['response'].values.astype(int),
    'question_to_dimension': question_to_dimension_stan  # K-length array mapping K->L
}

# --- Compile Stan model (recompiles every time) ---
print("Compiling Stan model (PyStan 2.19)...")
try:
    # This can take several minutes, especially the first time
    sm = pystan.StanModel(
        model_code=stan_model_code,
        extra_compile_args=STAN_COMPILE_FLAGS  # Pass the flags here
    )
    print("Stan model compiled successfully.")
except Exception as e:
    print(f"ERROR during Stan model compilation: {e}")
    exit()

# --- Generate initial values list for chains ---
print("Generating initial values for MCMC chains...")
inits_list = [generate_smart_init(df_long, K_final, L_final, J_final, C_final,
                                  question_idx_map, RESPONSE_OPTIONS, chain_id=c + 1)
              for c in range(CHAINS)]

# --- Run MCMC sampler ---
print(f"\nRunning MCMC sampling ({CHAINS} chains, {ITER_WARMUP} warmup, {ITER_SAMPLING} sampling)...")
fit = None  # Initialize fit object
try:
    # Suppress specific transient warnings if necessary using warnings context manager
    # Example: with warnings.catch_warnings(): warnings.simplefilter("ignore", category=SomeWarningType)
    fit = sm.sampling(data=stan_data, iter=ITER_WARMUP + ITER_SAMPLING, warmup=ITER_WARMUP,
                      chains=CHAINS, thin=THIN, seed=RANDOM_SEED, init=inits_list,
                      control={'adapt_delta': 0.85})  # Slightly higher adapt_delta might help prevent divergences
    print("\nSampling complete.")
except (RuntimeError, Exception) as e:
    # Catch potential runtime errors during sampling
    print(f"\nERROR during Stan sampling: {e}")
    print("Sampling failed. Check Stan model, data, initial values, and sampler output for details.")
    # If fit exists but failed, it might contain partial info or error messages
    if fit:
        print(fit)  # Print partial fit info if available
    exit()

if fit is None:
    print("ERROR: Stan fit object was not created successfully.")
    exit()

# -----------------------------------------
# Section 5: Diagnostic Checks Function
# -----------------------------------------
print("\n--- Section 5: Defining Diagnostic Checks Function ---")


def predict_responses(theta, a, cutpoints_samples, group_idx, q_idx, q_idx_to_dim_map):
    """
    Generate predicted responses based on model parameters from ONE posterior sample.
    Used for Posterior Predictive Checks.

    Args:
        theta (array): Theta estimates (J x L) for ONE sample.
        a (array): Discrimination estimates (K,) for ONE sample.
        cutpoints_samples (array): Cutpoint estimates (K x C-1) for ONE sample.
        group_idx (array): Group index for each observation (N,). 1-based.
        q_idx (array): Question index for each observation (N,). 1-based.
        q_idx_to_dim_map (dict): Map from Stan question index (1..K) to Stan dimension index (1..L).

    Returns:
        array: Predicted responses (N,).
    """
    N_obs = len(q_idx)
    preds = np.zeros(N_obs, dtype=int)
    if cutpoints_samples.shape[0] != len(a) or theta.shape[1] != len(np.unique(list(q_idx_to_dim_map.values()))):
        raise ValueError("Dimension mismatch in predict_responses inputs.")
    n_categories_resp = cutpoints_samples.shape[1] + 1
    response_opts = list(range(1, n_categories_resp + 1))

    for i in range(N_obs):
        g = group_idx[i] - 1  # 0-based for Python
        q = q_idx[i] - 1  # 0-based for Python
        stan_k = q + 1  # 1-based for map lookup

        # Check if question index is valid before looking up dimension
        if stan_k not in q_idx_to_dim_map:
            # This shouldn't happen if data prep is correct
            # print(f"Warning: Question index {stan_k} not found in q->dim map during PPC.")
            preds[i] = -1  # Indicate error or skip? Assign mean/median?
            continue

        dim = q_idx_to_dim_map[stan_k] - 1  # Get L index (0-based)

        # Ensure indices are within bounds
        if g >= theta.shape[0] or dim >= theta.shape[1] or q >= len(a):
            # print(f"Warning: Index out of bounds during PPC. Obs {i}, g={g}, q={q}, dim={dim}")
            preds[i] = -1
            continue

        eta = a[q] * theta[g, dim]
        current_cutpoints = cutpoints_samples[q]
        cum_prob_le = sps.expit(current_cutpoints - eta)  # P(Y <= k)

        probs = np.zeros(n_categories_resp)
        probs[0] = cum_prob_le[0]
        for k in range(1, n_categories_resp - 1):
            probs[k] = cum_prob_le[k] - cum_prob_le[k - 1]
        probs[n_categories_resp - 1] = 1.0 - cum_prob_le[n_categories_resp - 2]
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()  # Normalize safely

        try:
            preds[i] = np.random.choice(response_opts, p=probs)
        except ValueError:  # Fallback if normalization failed somehow
            try:
                probs /= probs.sum()
                preds[i] = np.random.choice(response_opts, p=probs)
            except:
                preds[i] = int(np.round(np.median(response_opts)))  # Last resort: predict median
    return preds


def run_diagnostic_checks(fit, samples, stan_data, df_long,
                          cat_idx_to_name_map, question_idx_map, group_idx_to_info,
                          n_chains, iter_sampling):
    """
    Runs comprehensive diagnostic checks on the Stan fit object and samples.
    Enhanced reporting included. Requires ArviZ.

    Args:
        fit: PyStan fit object.
        samples (dict): Dictionary of posterior samples from fit.extract().
        stan_data (dict): Data dictionary passed to Stan.
        df_long (pd.DataFrame): Processed long-format DataFrame.
        cat_idx_to_name_map (dict): Map from category index (1..L) to name.
        question_idx_map (dict): Map from question ID to Stan index (1..K).
        group_idx_to_info (dict): Map from group index (1..J) to team/year info.
        n_chains (int): Number of MCMC chains run.
        iter_sampling (int): Number of post-warmup samples per chain.

    Returns:
        tuple: (list of warning messages, ArviZ summary DataFrame or None)
    """
    print("\n--- RUNNING DIAGNOSTIC CHECKS ---")
    warnings_found = []
    total_post_warmup_samples = n_chains * iter_sampling
    K_dims = stan_data['K']
    L_dims = stan_data['L']
    C_cats = stan_data['C']
    response_opts = list(range(1, C_cats + 1))

    # Convert fit object to ArviZ InferenceData
    idata = None
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
            idata = az.from_pystan(posterior=fit, log_likelihood="log_lik")
            print("Successfully converted fit object to ArviZ InferenceData (basic).")
        except Exception as e2:
            print(f"WARNING: Could not convert fit to ArviZ InferenceData: {e2}")
            warnings_found.append("WARNING: ArviZ InferenceData conversion failed, diagnostics limited.")

    # --- 1. MCMC Convergence Checks ---
    print("\n--- Checking MCMC Convergence ---")
    summary = None
    if idata is not None:
        try:
            summary = az.summary(idata, round_to=3, kind='diagnostics', hdi_prob=0.94)  # Use 'stats' kind
            print("ArviZ summary generated.")
        except Exception as e:
            print(f"ERROR generating ArviZ summary: {e}")
            warnings_found.append("ERROR: ArviZ summary failed.")
    else:
        print("INFO: Cannot generate ArviZ summary without InferenceData.")
        warnings_found.append(
            "WARNING: Summary checks skipped.")

    # Check Divergences (using sampler parameters from fit object - more reliable)
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
        if 'r_hat' in summary.columns:
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
        ess_cols = [col for col in ['ess_bulk', 'ess_tail'] if col in summary.columns]
        if ess_cols:
            min_ess = summary[ess_cols].min().min()
            print(f"  Min n_eff ({'/'.join(ess_cols)}) observed: {min_ess:.0f}")
            low_neff_params = summary[summary[ess_cols].lt(min_expected_neff).any(axis=1)].index
            if len(low_neff_params) > 0:
                msg = (
                    f"WARNING: {len(low_neff_params)} params have low n_eff < {min_expected_neff:.0f}. Estimates noisy. "
                    f"(First 10: {low_neff_params.tolist()[:10]})")
                warnings_found.append(msg)
                print(msg)
            else:
                print(f"  n_eff Threshold Check: OK (All >= {min_expected_neff:.0f})")
        else:
            print("  WARNING: 'ess_bulk'/'ess_tail' columns not found in ArviZ summary.")
    else:
        print("  INFO: R-hat and n_eff checks skipped (no summary).")

    # --- 2. Trace Plot ("Fuzzy Caterpillar") Check ---
    print("\n--- Generating Trace Plots (Manual Check Required) ---")
    if idata is not None:
        try:
            # Dynamically get available variables and select subset
            available_vars = list(idata.posterior.data_vars.keys())

            # Select a subset of *available* parameters
            params_to_plot = []
            for p in ['mu', 'sigma']:  # Core scale/location params
                if p in available_vars:
                    params_to_plot.append(p)
            # Examples of indexed params
            indexed_prefixes = ['a', 'theta', 'Omega']  # Check these prefixes
            for prefix in indexed_prefixes:
                matches = sorted([v for v in available_vars if v.startswith(prefix + '[')])
                if matches:
                    params_to_plot.extend(matches[:min(len(matches), 3)])  # Plot first few

            if not params_to_plot:
                print("  INFO: Could not find standard model parameters in InferenceData for trace plots.")
            else:
                print(f"  Plotting traces for a subset: {params_to_plot}")
                az.plot_trace(idata, var_names=params_to_plot)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'trace_plots_subset.png'))
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                msg = (
                    "INFO: Trace plots generated (see trace_plots_subset.png). MANUALLY INSPECT mixing and convergence.")
                print(msg)
        except Exception as e:
            msg = f"WARNING: Could not generate trace plots: {e}. Manual inspection recommended."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: Trace plots skipped (InferenceData unavailable).")

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
            # Create the K->L map needed by predict_responses using final maps
            q_map_pred = {q_idx_stan: q_idx_to_dim_idx[q_idx_stan] for q_idx_stan in range(1, K_dims + 1)}

            pred_responses = predict_responses(sample_theta, sample_a, sample_cut,
                                               stan_data['group_idx'], stan_data['q_idx'], q_map_pred)

            # Filter out any potential error codes (-1) from predictions
            pred_responses_valid = pred_responses[pred_responses != -1]

            plt.figure(figsize=(12, 6))
            bins = np.arange(min(response_opts) - 0.5, max(response_opts) + 1.5, 1)
            plt.subplot(1, 2, 1)
            plt.hist(stan_data['y'], bins=bins, alpha=0.7, label='Observed', density=True)
            plt.title('Observed Response Distribution (Overall)')
            plt.xlabel('Response Category')
            plt.ylabel('Density')
            plt.xticks(response_opts)

            plt.subplot(1, 2, 2)
            plt.hist(pred_responses_valid, bins=bins, alpha=0.7, label='Predicted (1 Sample)', density=True)
            plt.title('Predicted Response Distribution (1 sample)')
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
            msg = f"WARNING: Could not generate PPC histogram: {e}."
            warnings_found.append(msg)
        print(msg)
    else:
        print("  INFO: Required parameters not found for PPC histogram.")

    # LOOIC Check
    # Check if 'log_lik' exists directly in samples dict (common for PyStan 2.19 extract)
    if 'log_lik' in samples and samples['log_lik'] is not None and samples['log_lik'].size > 0:
        print("  Calculating LOOIC using ArviZ from extracted 'log_lik' (may take time)...")
        try:
            # ArviZ expects log_lik shape (chains, draws, N) or (draws, N)
            # Samples dict usually has shape (draws, N) or (draws*chains, N)
            log_lik_array = samples['log_lik']
            if log_lik_array.ndim == 1 and log_lik_array.shape[0] == stan_data['N']:
                print(
                    "  WARNING: log_lik appears to be single value per obs, not samples. Cannot calculate LOOIC/WAIC reliably.")
            else:
                # Attempt LOO calculation directly from array
                loo_result = az.loo(idata, pointwise=True)  # Might need r_eff specified if using array
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
                    try:  # Plotting might fail if idata wasn't created properly
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
            msg = f"WARNING: Could not calculate LOOIC from log_lik array: {e}."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: 'log_lik' not found in samples dict. Skipping LOOIC calculation.")

    # --- 4. Parameter Estimate Checks ---
    print("\n--- Checking Parameter Estimates ---")

    # Check Correlations (Omega)
    if 'Omega' in samples:
        Omega_mean = samples['Omega'].mean(axis=0)
        Omega_flat = Omega_mean[np.triu_indices_from(Omega_mean, k=1)]
        max_correlation = np.max(np.abs(Omega_flat)) if len(Omega_flat) > 0 else np.nan
        print(f"  Max absolute off-diagonal correlation observed: {max_correlation:.3f}")
        merged_suggestions = []
        categories = list(cat_idx_to_name_map.values())
        if L_dims > 1:  # Only check if more than 1 dimension
            for i in range(L_dims):
                for j in range(i + 1, L_dims):
                    if np.abs(Omega_mean[i, j]) > HIGH_CORR_THRESHOLD:
                        merged_suggestions.append(
                            (categories[i], categories[j], Omega_mean[i, j]))
            if merged_suggestions:
                msg = (
                    f"INFO: Found {len(merged_suggestions)} pairs with |correlation| > {HIGH_CORR_THRESHOLD}. Consider merging/relationships:")
                print(msg)
                [print(f"      - '{c1}' & '{c2}' (Corr: {co:.3f})") for c1, c2, co in merged_suggestions]
    else:
        print("  INFO: 'Omega' not found. Skipping correlation check.")

    # Check Discrimination (a)
    if 'a' in samples:
        a_mean = samples['a'].mean(axis=0)
        min_discrimination = np.min(a_mean) if len(a_mean) > 0 else np.nan
        print(f"  Min mean discrimination ('a') observed: {min_discrimination:.3f}")
        idx_to_question = {v: k for k, v in question_idx_map.items()}
        low_a_questions = []
        if len(a_mean) == K_dims:
            for k_idx_0based in range(K_dims):
                if a_mean[k_idx_0based] < LOW_DISCRIMINATION_THRESHOLD:
                    stan_idx = k_idx_0based + 1
                    q_id = idx_to_question.get(stan_idx, f"Idx {stan_idx}")
                    cat_name = question_to_cat_name_final.get(q_id, "Unknown")
                    low_a_questions.append({'Question': q_id, 'Category': cat_name, 'Mean_a': a_mean[k_idx_0based]})
            if low_a_questions:
                msg = (
                    f"INFO: Found {len(low_a_questions)} questions with mean discrimination ('a') < {LOW_DISCRIMINATION_THRESHOLD}. "
                    "These questions may weakly distinguish capability levels. "
                    "Consider for revision/removal in future surveys:")
                print(msg)
                print(pd.DataFrame(low_a_questions).round(3).sort_values('Mean_a'))
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
    samples = fit.extract()  # Already permuted in PyStan 2.19
    print("Posterior samples extracted.")
except Exception as e:
    print(f"ERROR: Could not extract samples from fit object: {e}")
    print("Cannot proceed with results processing.")
    exit()

# --- Theta Estimates (Team Capabilities) ---
theta_df = None
corr_df = None
ranking_summary = pd.DataFrame()  # Initialize
if 'theta' in samples:
    theta_means = samples['theta'].mean(axis=0)
    # Use final category names derived after question filtering
    theta_columns = [f"theta_{cat_idx_to_name_final[_l + 1]}" for _l in range(L_final)]
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
        if ID_VAR not in theta_df:
            theta_df[ID_VAR] = 'Unknown'
        if YEAR_COL not in theta_df:
            theta_df[YEAR_COL] = 0

    # Reorder columns
    theta_df = theta_df[[ID_VAR, YEAR_COL, 'group', 'group_idx'] + theta_columns]

    print("\nEstimated Team Capabilities (Theta Posterior Means):")
    print(theta_df.head())
    theta_filepath = os.path.join(OUTPUT_DIR, 'team_capability_estimates.csv')
    theta_df.to_csv(theta_filepath, index=False)
    print(f"Team capability estimates saved to {theta_filepath}")
else:
    print("WARNING: 'theta' not found in samples.")

# --- Correlation Matrix Estimate (Omega) ---
if 'Omega' in samples:
    Omega_mean = samples['Omega'].mean(axis=0)
    category_names_final = list(cat_idx_to_name_final.values())
    corr_df = pd.DataFrame(Omega_mean, index=category_names_final, columns=category_names_final)
    print("\nEstimated Mean Correlation Matrix ('Omega'):")
    print(corr_df.round(2))
    corr_filepath = os.path.join(OUTPUT_DIR, 'capability_correlations.csv')
    corr_df.to_csv(corr_filepath)
    print(f"Capability correlation matrix saved to {corr_filepath}")
else:
    print("WARNING: 'Omega' not found in samples.")

# --- Mu and Sigma Means (for plotting) ---
mu_map_for_plot = {}
sigma_map_for_plot = {}
cat_names_list_plot = []
if 'mu' in samples and 'sigma' in samples:
    mu_means_plot = samples['mu'].mean(axis=0)
    sigma_means_plot = samples['sigma'].mean(axis=0)
    cat_names_list_plot = [cat_idx_to_name_final[i] for i in range(1, L_final + 1)]
    mu_map_for_plot = dict(zip(cat_names_list_plot, mu_means_plot))
    sigma_map_for_plot = dict(zip(cat_names_list_plot, sigma_means_plot))
    print("\nCalculated posterior means for mu and sigma for plotting.")
else:
    print("WARNING: 'mu' or 'sigma' not found. Cannot create maps for standardized plots.")


# --- Ranking and Improvement Summary ---
# (Function definition moved here for clarity before use)
def create_ranking_summary(theta_df_in, id_var_in, year_col_in, cat_map_in, years_in):
    """Creates a summary DataFrame of rankings and improvements between first and last year."""
    if theta_df_in is None or theta_df_in.empty:
        return pd.DataFrame()
    years_sorted = sorted(years_in)
    if len(years_sorted) < 2:
        print("INFO: Need >= 2 years for ranking summary.")
        return pd.DataFrame()
    year1, year_last = years_sorted[0], years_sorted[-1]
    print(f"\nCalculating rankings/improvements between {year1} and {year_last}...")
    summary_data = []
    unique_ids = theta_df_in[id_var_in].unique()
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

        # new check?
        if rank_y1.isnull().all() or rank_y_last.isnull().all():
            continue  # Skip if no ranks possible
        rank_y1 = rank_y1.astype(int)
        rank_y_last = rank_y_last.astype(int)

        # Convert after check
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
        print("INFO: Fewer than two years processed. Cannot calculate improvement.")
else:
    print("INFO: Skipping ranking summary (theta_df unavailable).")

# -----------------------------------------
# Section 7: Visualization Functions
# -----------------------------------------
print("\n--- Section 7: Defining Visualization Functions ---")


def filename_escape(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


# --- Slope Chart (Internal Standardization) ---
def plot_reorg_slope_chart_standardized_internal(theta_df_viz, category_name_viz, id_var_viz, year_col_viz, years_viz,
                                                 mu_map_viz, sigma_map_viz, parent_to_child_map, output_dir_viz):
    """Creates slope chart using theta scores standardized INTERNALLY."""
    print(f"  Generating STANDARDIZED (internal) reorg slope chart for: {category_name_viz}")
    if theta_df_viz is None or theta_df_viz.empty:
        print(f"INFO: theta_df is empty, skipping reorg slope chart for {category_name_viz}.")
        return None

    pop_mean = mu_map_viz.get(category_name_viz)
    pop_sd = sigma_map_viz.get(category_name_viz)

    if pop_mean is None or pop_sd is None:
        print(f"INFO: Missing mu/sigma for {category_name_viz}. Skipping chart.")
        return None

    if pop_sd <= 1e-6:
        print(f"INFO: Sigma near zero for {category_name_viz}. Skipping chart.")
        return None

    raw_col_name = f"theta_{category_name_viz}"
    if theta_df_viz is None or raw_col_name not in theta_df_viz.columns:
        print(f"INFO: theta_df or raw column {raw_col_name} missing. Skipping chart.")
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
    # Plot Year 1 points & labels
    for team_id, row in df_y1.iterrows():
        raw_val = row[raw_col_name]
        if pd.notna(raw_val):
            y1_val_z = (raw_val - pop_mean) / pop_sd
            ax.plot(year1, y1_val_z, 'o', color='black', markersize=5)
            texts.append(ax.text(year1 - 0.05, y1_val_z, team_id, ha='right', va='center', fontsize=8))
            plotted_points_y1[team_id] = y1_val_z
    all_year2_teams = sorted(df_y2.index.unique().tolist())
    num_colors = max(1, len(all_year2_teams))
    cmap_func = cm.get_cmap('tab20', num_colors)  # Use tab20 for distinct colors
    color_map_y2 = {team: cmap_func(i % num_colors) for i, team in enumerate(all_year2_teams)}  # Cycle colors if > 20
    for team_id, row in df_y2.iterrows():
        raw_val = row[raw_col_name]
        if pd.notna(raw_val):
            y2_val_z = (raw_val - pop_mean) / pop_sd
            point_color = color_map_y2.get(team_id, 'gray')  # Use mapped color
            ax.plot(year2, y2_val_z, 'o', color=point_color, markersize=6)  # Mark Y2 points with color
            texts.append(ax.text(year2 + 0.05, y2_val_z, team_id, ha='left', va='center', fontsize=8))
            plotted_points_y2[team_id] = y2_val_z

    # --- Plot Connecting Lines based on Mapping ---
    print(f"  Plotting reorg lines for {category_name_viz}...")
    lines_plotted = 0
    for parent_team, child_teams in parent_to_child_map.items():
        if parent_team in plotted_points_y1:
            y1_val_z = plotted_points_y1[parent_team]
            for child_team in child_teams:
                if child_team in plotted_points_y2:
                    y2_val_z = plotted_points_y2[child_team]
                    line_color = color_map_y2.get(child_team, 'gray')  # Color by child team
                    ax.plot([year1, year2], [y1_val_z, y2_val_z], linestyle='-', lw=2.5, color=line_color, alpha=0.6)
                    lines_plotted += 1
    print(f"    {lines_plotted} lineage lines plotted for {category_name_viz}.")

    # Adjust labels to prevent overlap
    if ADJUST_TEXT_AVAILABLE and texts:
        try:
            adjust_text(texts, ax=ax, force_points=(0.2, 0.3), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except Exception as e:
            print(f"    Warning: adjust_text failed: {e}")
    elif not ADJUST_TEXT_AVAILABLE:
        print("    Info: Skipping adjust_text (library not installed).")

    # Final touches
    ax.set_title(f'Standardized Capability Evolution: {category_name_viz} ({year1} vs {year2})', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'Capability Z-Score (Std. Devs from Mean)', fontsize=12)
    ax.set_xticks([year1, year2])
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.0, alpha=0.8, zorder=0)  # center line
    ax.set_xlim(year1 - 0.4, year2 + 0.4)  # Wider padding for labels

    # Create a simple legend (might get crowded if too many Y2 teams)
    if len(all_year2_teams) <= 20 and all_year2_teams:  # Only show legend if manageable
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=team,
                              markerfacecolor=color_map_y2.get(team, 'gray'), markersize=6)
                   for team in all_year2_teams if team in plotted_points_y2]
        if handles:
            ax.legend(handles=handles, title=f"{year2} Teams", bbox_to_anchor=(1.05, 1), loc='upper left',
                      fontsize='small')
            plt.subplots_adjust(right=0.85)
    filepath = os.path.join(output_dir_viz,
                            f'standardized_internal_reorg_slope_chart_{filename_escape(category_name_viz)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close(fig)
    print(f"Standardized slope chart saved for {category_name_viz} to {filepath}")
    return fig


def plot_omega_clustermap(corr_df_viz, output_dir_viz):
    """Visualizes the correlation matrix Omega as a clustered heatmap using coolwarm."""
    if corr_df_viz is None or corr_df_viz.empty:
        return
    n_cats = corr_df_viz.shape[0]
    print("\n--- Generating Omega Clustermap ---")
    if n_cats < 2:
        print("INFO: Need >= 2 categories.")
        return
    figsize = (max(8, n_cats * 0.9), max(7, n_cats * 0.8))
    annotate = n_cats <= 12  # Annotate fewer cells
    try:
        cluster_grid = sns.clustermap(
            corr_df_viz, method='average', metric='euclidean', cmap='coolwarm',  # Blue=Positive
            vmin=-1, vmax=1, center=0, annot=annotate, fmt=".2f",
            linewidths=.5, linecolor='lightgray', figsize=figsize
        )
        plt.setp(cluster_grid.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
        cluster_grid.fig.suptitle('Clustered Heatmap of Capability Correlations (Mean Omega)', y=1.02)
        filepath = os.path.join(output_dir_viz, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        print(f"Saved Omega clustermap to {filepath}")
        print("Interpretation Notes: Blue=positive, Red=negative correlation. Dendrograms show clustering.")
    except Exception as e:
        print(f"WARNING: Could not generate Omega clustermap: {e}")


# --- Helper: Calculate GRM Probabilities ---
def calculate_grm_probabilities(theta_val, a_val, cutpoints_val, n_categories):
    """Calculates GRM probabilities for each category."""
    cutpoints_val = np.sort(cutpoints_val)
    eta = a_val * theta_val
    cum_prob_le = np.zeros(n_categories - 1)
    try:
        cum_prob_le = sps.expit(cutpoints_val - eta)
    except FloatingPointError:
        if eta > 30:
            cum_prob_le = np.zeros(n_categories - 1)
        elif eta < -30:
            cum_prob_le = np.ones(n_categories - 1)
        else:
            cum_prob_le = sps.expit(cutpoints_val)  # Fallback

    probs = np.zeros(n_categories)
    try:
        probs[0] = cum_prob_le[0]
        for c in range(1, n_categories - 1):
            probs[c] = max(0.0, cum_prob_le[c] - cum_prob_le[c - 1])
        probs[n_categories - 1] = max(0.0, 1.0 - cum_prob_le[n_categories - 2])
        probs_sum = probs.sum()
        if probs_sum > 1e-8:
            probs /= probs_sum
        else:
            probs = np.ones(n_categories) / n_categories
    except IndexError:
        probs = np.ones(n_categories) / n_categories
    return probs


# --- Plot: ICC/KDE Comparison ---
def plot_predicted_vs_empirical_dist(question_id, samples_dict, stan_data_dict, question_idx_map_val,
                                     q_idx_to_dim_idx_val, response_options_val, n_categories_val, output_dir_val,
                                     ci_level=0.94, n_posterior_samples=500):
    """Plots model-predicted response distribution vs. empirical KDE for a question."""
    print(f"--- Generating Predicted vs Empirical Plot for: {question_id} ---")
    if question_id not in question_idx_map_val:
        print(f"ERROR: Q '{question_id}' not found. Skipping.")
        return
    k_stan = question_idx_map_val[question_id]
    if k_stan not in q_idx_to_dim_idx_val:
        print(f"ERROR: Stan index {k_stan} not in q->dim map. Skipping.")
        return
    l_stan = q_idx_to_dim_idx_val[k_stan]
    required_keys = ['a', 'cutpoints', 'theta']
    required_data = ['y', 'q_idx', 'group_idx']
    if not all(key in samples_dict for key in required_keys) or not all(key in stan_data_dict for key in required_data):
        print("ERROR: Missing required data/samples. Skipping.")
        return
    k_idx_0based = k_stan - 1
    l_idx_0based = l_stan - 1

    obs_indices = np.where(stan_data_dict['q_idx'] == k_stan)[0]
    if len(obs_indices) == 0:
        print(f"INFO: No observed responses for {question_id}. Skipping.")
        return
    empirical_responses = stan_data_dict['y'][obs_indices]
    groups_for_q = stan_data_dict['group_idx'][obs_indices]

    total_samples = samples_dict['a'].shape[0]
    sample_indices = np.random.choice(total_samples, min(n_posterior_samples, total_samples), replace=False)
    predicted_proportions_samples = np.zeros((len(sample_indices), n_categories_val))

    print(f"  Calculating predicted proportions using {len(sample_indices)} samples for {question_id}...")
    for i, s_idx in enumerate(sample_indices):  # Can wrap sample_indices with tqdm()
        if k_idx_0based >= samples_dict['a'].shape[1] or k_idx_0based >= samples_dict['cutpoints'].shape[1]:
            continue  # Skip if item index invalid
        a_s = samples_dict['a'][s_idx, k_idx_0based]
        cutpoints_s = samples_dict['cutpoints'][s_idx, k_idx_0based, :]
        # Ensure group indices and latent dim index are valid
        valid_group_indices = groups_for_q - 1 < samples_dict['theta'].shape[1]
        valid_dim_index = l_idx_0based < samples_dict['theta'].shape[2]
        if not valid_dim_index or not np.all(valid_group_indices):
            # print(f"Warning: Invalid indices for theta lookup sample {s_idx}, item {k_idx_0based}. Skipping sample.")
            predicted_proportions_samples[i, :] = np.nan  # Mark as invalid
            continue
        theta_samples_for_q = samples_dict['theta'][
            s_idx, (groups_for_q - 1)[valid_group_indices], l_idx_0based]  # Get theta for relevant groups
        if theta_samples_for_q.size == 0:  # Handle case where no valid groups remain
            predicted_proportions_samples[i, :] = np.nan
            continue

        probs_per_obs = np.array(
            [calculate_grm_probabilities(th, a_s, cutpoints_s, n_categories_val) for th in theta_samples_for_q])
        predicted_proportions_samples[i, :] = np.nanmean(probs_per_obs, axis=0)  # Use nanmean

    # Filter out NaN rows if any samples failed
    predicted_proportions_samples = predicted_proportions_samples[~np.isnan(predicted_proportions_samples).any(axis=1)]
    if predicted_proportions_samples.shape[0] == 0:
        print(f"ERROR: Could not calculate any valid predicted proportions for {question_id}. Skipping plot.")
        return

    pred_mean = predicted_proportions_samples.mean(axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    pred_lower = np.percentile(predicted_proportions_samples, lower_perc, axis=0)
    pred_upper = np.percentile(predicted_proportions_samples, upper_perc, axis=0)
    a_mean = samples_dict['a'][:, k_idx_0based].mean()

    print(f"  Generating plot for {question_id}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = [str(r) for r in response_options_val]
    x_ticks = np.array(response_options_val)
    try:
        sns.kdeplot(empirical_responses, ax=ax, color='black', linestyle='--', linewidth=2, label='Empirical KDE',
                    bw_adjust=1.5)
        sns.rugplot(empirical_responses, ax=ax, color='black', alpha=0.5, height=0.03)
    except Exception as e:
        print(f"  Warning: KDE plot failed: {e}. Using histogram.")
        counts, _ = np.histogram(empirical_responses,
                                 bins=np.arange(
                                     min(response_options_val) - 0.5,
                                     max(response_options_val) + 1.5,
                                     1),
                                 density=True)
        ax.bar(
            x_ticks, counts, width=0.6, color='gray', alpha=0.5, label='Empirical Hist')
    ax.plot(x_ticks, pred_mean, marker='o', linestyle='-', color='dodgerblue', label='Predicted Mean', zorder=10)
    ax.fill_between(x_ticks, pred_lower, pred_upper, color='dodgerblue', alpha=0.2,
                    label=f'Predicted {ci_level * 100:.0f}% CI')
    ax.set_xlabel("Response Category", fontsize=12)
    ax.set_ylabel("Density / Proportion", fontsize=12)
    ax.set_title(f"Item Analysis: Question '{question_id}' (Mean a ≈ {a_mean:.2f})", fontsize=14, pad=15)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    filepath = os.path.join(output_dir_val, f'item_fit_{filename_escape(question_id)}.png')
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")


# --- Plot: Category Response Functions (CRF/ICC) ---
def plot_category_response_functions(question_id, samples_dict, question_idx_map_val, n_categories_val,
                                     response_options_val, output_dir_val, theta_range=(-3.5, 3.5), n_theta_points=100,
                                     ci_level=0.94, n_posterior_samples=500):
    """Plots the Category Response Functions P(Y=c|theta) for a question."""
    print(f"--- Generating Category Response Functions for: {question_id} ---")
    if question_id not in question_idx_map_val:
        print(f"ERROR: Q '{question_id}' not found. Skipping.")
        return
    k_stan = question_idx_map_val[question_id]
    k_idx_0based = k_stan - 1
    required_keys = ['a', 'cutpoints']
    if not all(key in samples_dict for key in required_keys):
        print(f"ERROR: Missing required keys. Skipping.")
        return
    if k_idx_0based >= samples_dict['a'].shape[1] or k_idx_0based >= samples_dict['cutpoints'].shape[1]:
        print(f"ERROR: Index {k_idx_0based} out of bounds. Skipping {question_id}.")
        return

    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta_points)
    total_samples = samples_dict['a'].shape[0]
    if total_samples == 0:
        print("ERROR: No posterior samples. Skipping.")
        return
    sample_indices = np.random.choice(total_samples, min(n_posterior_samples, total_samples), replace=False)
    # Store probabilities [sample, theta_point, category]
    crf_probs_samples = np.zeros((len(sample_indices), n_theta_points, n_categories_val))

    print(f"  Calculating CRF probabilities using {len(sample_indices)} samples for {question_id}...")
    for i, s_idx in enumerate(sample_indices):  # Can wrap sample_indices with tqdm()
        # Add index validation similar to plot_predicted_vs_empirical_dist if needed
        a_s = samples_dict['a'][s_idx, k_idx_0based]
        cutpoints_s = samples_dict['cutpoints'][s_idx, k_idx_0based, :]
        for j, th in enumerate(theta_vals):
            crf_probs_samples[i, j, :] = calculate_grm_probabilities(th, a_s, cutpoints_s, n_categories_val)

    # Calculate mean probabilities for each category across theta
    crf_mean = crf_probs_samples.mean(axis=0)  # Shape: [n_theta_points, n_categories]

    a_mean = samples_dict['a'][:, k_idx_0based].mean()

    print(f"  Generating plot for {question_id}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_categories_val))
    for c in range(n_categories_val):
        ax.plot(theta_vals, crf_mean[:, c], color=colors[c], linewidth=2, label=f'P(Y={response_options_val[c]})')

    ax.set_xlabel("Latent Trait (Theta)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Category Response Functions: Question '{question_id}' (Mean a ≈ {a_mean:.2f})", fontsize=14)
    ax.legend(title="Response")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(theta_range[0], theta_range[1])
    plt.tight_layout()
    filepath = os.path.join(output_dir_val, f'crf_{filename_escape(question_id)}.png')
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")


# --- Helper: Calculate Approximate GRM Item Information ---
def calculate_grm_item_information_approx(theta_val, a_val, cutpoints_val, n_categories):
    """Calculates GRM Item Information using approximate formula."""
    if n_categories < 2:
        return 0.0
    num_cutpoints = n_categories - 1
    if len(cutpoints_val) != num_cutpoints:
        return 0.0  # Handle error
    cutpoints_val = np.sort(cutpoints_val)
    information = 0.0
    try:
        for c in range(num_cutpoints):
            logit = cutpoints_val[c] - a_val * theta_val
            P_star = sps.expit(logit)
            p_times_1_minus_p = max(0.0, P_star * (1.0 - P_star))
            information += p_times_1_minus_p
        information *= (a_val ** 2)
    except FloatingPointError:
        return 0.0
    return max(0.0, information)


# --- Plot: Item Information Function (IIF) ---
def plot_item_information_function(question_id, samples_dict, question_idx_map_val, n_categories_val, output_dir_val,
                                   theta_range=(-3.5, 3.5), n_theta_points=100, ci_level=0.94, n_posterior_samples=500,
                                   return_mean_info=False):
    """Plots the Item Information Function (IIF) with CI."""
    print(f"--- Generating Item Information Function for: {question_id} ---")
    if question_id not in question_idx_map_val:
        print(f"ERROR: Q '{question_id}' not found. Skipping.")
        return None if return_mean_info else False
    k_stan = question_idx_map_val[question_id]
    k_idx_0based = k_stan - 1
    required_keys = ['a', 'cutpoints']
    if not all(key in samples_dict for key in required_keys):
        print(f"ERROR: Missing required keys. Skipping.")
        return None if return_mean_info else False
    if k_idx_0based >= samples_dict['a'].shape[1] or k_idx_0based >= samples_dict['cutpoints'].shape[1]:
        print(
            f"ERROR: Index {k_idx_0based} out of bounds. Skipping {question_id}.")
        return None if return_mean_info else False

    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta_points)
    total_samples = samples_dict['a'].shape[0]
    if total_samples == 0:
        print("ERROR: No posterior samples. Skipping.")
        return None if return_mean_info else False
    num_samples_to_use = min(n_posterior_samples, total_samples)
    sample_indices = np.random.choice(total_samples, num_samples_to_use, replace=False)
    information_samples = np.zeros((num_samples_to_use, n_theta_points))

    print(f"  Calculating IIF (approx) using {len(sample_indices)} samples for {question_id}...")
    for i, s_idx in enumerate(sample_indices):  # Can wrap sample_indices with tqdm()
        if s_idx >= samples_dict['a'].shape[0] or s_idx >= samples_dict['cutpoints'].shape[0]:
            continue
        if k_idx_0based >= samples_dict['a'].shape[1] or k_idx_0based >= samples_dict['cutpoints'].shape[1]:
            information_samples[i, :] = 0
            continue
        a_s = samples_dict['a'][s_idx, k_idx_0based]
        cutpoints_s = samples_dict['cutpoints'][s_idx, k_idx_0based, :]
        for j, th in enumerate(theta_vals):
            information_samples[i, j] = calculate_grm_item_information_approx(th, a_s, cutpoints_s, n_categories_val)

    info_mean = information_samples.mean(axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    info_lower = np.percentile(information_samples, lower_perc, axis=0)
    info_upper = np.percentile(information_samples, upper_perc, axis=0)
    a_mean = samples_dict['a'][:, k_idx_0based].mean()

    print(f"  Generating plot for {question_id}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_vals, info_mean, color='navy', linewidth=2, label='Mean Information')
    ax.fill_between(theta_vals, info_lower, info_upper, color='skyblue', alpha=0.4, label=f'{ci_level * 100:.0f}% CI')
    ax.set_xlabel("Latent Trait (Theta)", fontsize=12)
    ax.set_ylabel("Item Information (Approx.)", fontsize=12)
    ax.set_title(f"Item Information Function: Question '{question_id}' (Mean a ≈ {a_mean:.2f})", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_xlim(theta_range[0], theta_range[1])
    plt.tight_layout()
    filepath = os.path.join(output_dir_val, f'iif_{filename_escape(question_id)}.png')
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")
    if return_mean_info:
        return theta_vals, info_mean
    else:
        return True


# --- Plot: Test Information Function (TIF) ---
def plot_test_information_function(samples_dict, question_idx_map_val, n_categories_val, output_dir_val,
                                   theta_range=(-3.5, 3.5), n_theta_points=100, ci_level=0.94, n_posterior_samples=500):
    """Plots the Test Information Function (TIF) by summing IIFs."""
    print(f"\n--- Generating Test Information Function (Approx.) ---")
    K_final = len(question_idx_map_val)
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta_points)
    required_keys = ['a', 'cutpoints']
    if not all(key in samples_dict for key in required_keys):
        print(f"ERROR: Missing required keys. Skipping TIF.")
        return

    total_samples = samples_dict['a'].shape[0]
    if total_samples == 0:
        print("ERROR: No posterior samples. Skipping TIF.")
        return
    num_samples_to_use = min(n_posterior_samples, total_samples)
    sample_indices = np.random.choice(total_samples, num_samples_to_use, replace=False)
    tif_samples = np.zeros((num_samples_to_use, n_theta_points))
    idx_to_question = {v: k for k, v in question_idx_map_val.items()}

    print(f"  Calculating TIF using {K_final} items and {len(sample_indices)} samples...")
    for i, s_idx in enumerate(sample_indices):  # Can wrap sample_indices with tqdm()
        for k_stan, q_id in idx_to_question.items():
            k_idx_0based = k_stan - 1
            if s_idx >= samples_dict['a'].shape[0] or s_idx >= samples_dict['cutpoints'].shape[0]:
                continue
            if k_idx_0based >= samples_dict['a'].shape[1] or k_idx_0based >= samples_dict['cutpoints'].shape[1]:
                continue
            a_s = samples_dict['a'][s_idx, k_idx_0based]
            cutpoints_s = samples_dict['cutpoints'][s_idx, k_idx_0based, :]
            iif_s_k = np.array(
                [calculate_grm_item_information_approx(th, a_s, cutpoints_s, n_categories_val) for th in theta_vals])
            tif_samples[i, :] += iif_s_k

    tif_mean = tif_samples.mean(axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    tif_lower = np.percentile(tif_samples, lower_perc, axis=0)
    tif_upper = np.percentile(tif_samples, upper_perc, axis=0)
    sem_mean = np.sqrt(1.0 / np.maximum(tif_mean, 1e-6))
    sem_upper = np.sqrt(1.0 / np.maximum(tif_lower, 1e-6))
    sem_lower = np.sqrt(1.0 / np.maximum(tif_upper, 1e-6))

    print("  Generating plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'darkblue'
    color2 = 'firebrick'
    ax1.set_xlabel("Latent Trait (Theta)", fontsize=12)
    ax1.set_ylabel("Test Information (Approx.)", fontsize=12, color=color1)
    ax1.plot(theta_vals, tif_mean, color=color1, linewidth=2.5, label='Mean Test Information')
    ax1.fill_between(theta_vals, tif_lower, tif_upper, color='lightblue', alpha=0.5,
                     label=f'TIF {ci_level * 100:.0f}% CI')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle=':', alpha=0.7, axis='both')
    ax1.legend(loc='upper left')
    ax1.set_xlim(theta_range[0], theta_range[1])
    ax2 = ax1.twinx()
    ax2.set_ylabel("Standard Error of Measurement (SEM)", fontsize=12, color=color2)
    ax2.plot(theta_vals, sem_mean, color=color2, linewidth=2, linestyle='--', label='Mean SEM')
    ax2.fill_between(theta_vals, sem_lower, sem_upper, color='lightcoral', alpha=0.3,
                     label=f'SEM {ci_level * 100:.0f}% CI')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    plt.title(f"Test Information Function (TIF) and SEM ({K_final} Items - Approx. Info)", fontsize=14)
    fig.tight_layout()
    filepath = os.path.join(output_dir_val, 'tif_sem_approx.png')
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  Plot saved to {filepath}")


# -----------------------------------------
# Section 8: Main Execution Flow & Reporting
# -----------------------------------------
print("\n--- Section 8: Running Main Analysis Flow ---")

# --- Perform Diagnostics ---
diagnostic_warnings, arviz_summary = run_diagnostic_checks(
    fit=fit, samples=samples, stan_data=stan_data, df_long=df_long,
    cat_idx_to_name_map=cat_idx_to_name_final, question_idx_map=question_idx_map,
    group_idx_to_info=group_idx_to_info, n_chains=CHAINS, iter_sampling=ITER_SAMPLING)

# --- Generate Visualizations ---
print("\n--- Generating Core Visualizations ---")
if theta_df is not None:
    actual_years_final = sorted(theta_df[YEAR_COL].unique())
    if len(actual_years_final) >= 2:
        years_to_plot = [actual_years_final[0], actual_years_final[-1]]  # Compare first and last

        print(f"\nGenerating STANDARDIZED (internal) REORG slope charts "
              f"comparing {years_to_plot[0]} and {years_to_plot[1]}...")
        for cat_name in cat_names_list_plot:  # Use list derived when creating maps
            plot_reorg_slope_chart_standardized_internal(theta_df_viz=theta_df, category_name_viz=cat_name,
                                                         id_var_viz=ID_VAR, year_col_viz=YEAR_COL,
                                                         years_viz=years_to_plot, mu_map_viz=mu_map_for_plot,
                                                         sigma_map_viz=sigma_map_for_plot,
                                                         parent_to_child_map=PARENT_TO_CHILD_MAPPING,
                                                         output_dir_viz=OUTPUT_DIR)
    else:
        print("INFO: Skipping slope charts (< 2 years processed).")
    if corr_df is not None:
        plot_omega_clustermap(corr_df, OUTPUT_DIR)
    else:
        print("INFO: Skipping Omega clustermap (corr_df unavailable).")
elif theta_df is None:
    print("INFO: Skipping core visualizations (theta_df unavailable).")
else:
    print("INFO: Skipping visualizations (theta_df unavailable).")

# --- Generate Item-Level Diagnostic Plots ---
print("\n--- Generating Item-Level Diagnostic Plots ---")
questions_to_plot = []
if 'a' in samples:
    a_means = samples['a'].mean(axis=0)
    if K_final > 0:  # Check if there are any questions
        idx_to_question = {v: k for k, v in question_idx_map.items()}
        num_select = min(3, K_final // 2 if K_final > 1 else K_final)  # Ensure num_select <= K

        if num_select > 0:
            sorted_indices = np.argsort(a_means)
            bottom_indices_0based = sorted_indices[:num_select]
            top_indices_0based = sorted_indices[-num_select:]
            selected_indices_0based = np.unique(np.concatenate([bottom_indices_0based, top_indices_0based]))
            print(f"\nSelecting questions with {num_select} lowest and {num_select} highest mean 'a' for item plots...")
            for idx_0based in selected_indices_0based:
                k_stan = idx_0based + 1
                if k_stan in idx_to_question:
                    questions_to_plot.append(idx_to_question[k_stan])
                else:
                    print(f"Warning: Stan index {k_stan} not found in idx_to_question map.")
            questions_to_plot = sorted(list(set(questions_to_plot)))
            print(f"Will generate item plots for: {questions_to_plot}")

            for q_id in questions_to_plot:
                # Plot ICC/KDE Comparison
                plot_predicted_vs_empirical_dist(question_id=q_id, samples_dict=samples, stan_data_dict=stan_data,
                                                 question_idx_map_val=question_idx_map,
                                                 q_idx_to_dim_idx_val=q_idx_to_dim_idx,
                                                 response_options_val=RESPONSE_OPTIONS, n_categories_val=C_final,
                                                 output_dir_val=OUTPUT_DIR)
                # Plot Category Response Functions (CRF)
                plot_category_response_functions(question_id=q_id, samples_dict=samples,
                                                 question_idx_map_val=question_idx_map, n_categories_val=C_final,
                                                 response_options_val=RESPONSE_OPTIONS, output_dir_val=OUTPUT_DIR)
                # Plot Item Information Function (IIF - Approx)
                plot_item_information_function(question_id=q_id, samples_dict=samples,
                                               question_idx_map_val=question_idx_map, n_categories_val=C_final,
                                               output_dir_val=OUTPUT_DIR)
        else:
            print("INFO: Not enough questions (K <= 1) to select top/bottom 'a'.")
    else:
        print("INFO: K_final is 0. No questions to plot.")

    # --- Generate Test Information Function Plot ---
    # (Call this AFTER the loop for individual items)
    if K_final > 0:
        plot_test_information_function(samples_dict=samples, question_idx_map_val=question_idx_map,
                                       n_categories_val=C_final, output_dir_val=OUTPUT_DIR)
    else:
        print("INFO: Skipping TIF plot (K=0).")

else:
    print("INFO: 'a' not found in samples. Skipping all item-level plots based on discrimination.")

# --- Optional: Compare Estimated Parameters to True Simulation Parameters ---
if RUN_SIMULATION and true_params is not None:
    print("\n--- Comparing Estimates to True Simulation Parameters (Optional) ---")
    param_mae = {}
    for param_name in ['Omega', 'mu', 'sigma']:
        if param_name in samples and f'true_{param_name}' in true_params:
            est_mean = samples[param_name].mean(axis=0)
            true_val = true_params[f'true_{param_name}']
            # Ensure shapes match before calculating difference
            if hasattr(est_mean, 'shape') and hasattr(true_val, 'shape') and est_mean.shape == true_val.shape:
                mae = np.mean(np.abs(est_mean - true_val))
                param_mae[param_name] = mae
                print(f"  Mean Absolute Error ({param_name}): {mae:.3f}")
            else:
                print(f"  Skipping {param_name} comparison due to shape mismatch.")
        else:
            print(f"  Skipping {param_name} comparison (missing estimates or true values).")
    # Can add more sophisticated comparisons (e.g., correlation for theta) if needed

# --- Final Summary ---
print("\n--- Analysis Complete ---")
print(f"Results, diagnostics, and plots saved in: {OUTPUT_DIR}")
if any("CRITICAL" in w for w in diagnostic_warnings):
    print("\n*** ACTION REQUIRED: Critical warnings detected. Results may be unreliable. ***")
elif diagnostic_warnings:
    print("\nNOTE: Potential issues flagged. Review warnings and manual checks.")
print("\nReview Diagnostics Output and Manual Checks (Trace Plots, PPCs, Item Plots) before interpreting results.")
print("NOTE: IIF and TIF plots use an *approximate* formula for GRM information.")

# -----------------------------------------
# Section 9: DORA Specific Considerations (Commentary)
# -----------------------------------------
print("\n--- Section 9: DORA Survey Specific Considerations ---")
print("""
1.  Interpretation: 'theta' / Z-scores = team capability. Compare relative values & changes. Standardized scores show deviation from avg.
2.  Focus on Change: Use standardized slope charts for evolution. Reorgs complicate direct 'improvement' metrics.
3.  Item Diagnostics: Use ICC/KDE, CRF, IIF plots to check if questions behave as expected. Low 'a' or poor fit -> review question. TIF shows where survey is most precise.
4.  NA Handling & Separate Analyses: High NAs may require separate analyses. Adapt Section 2 filtering if needed.
5.  Linking to DORA Metrics: Crucial next step! Correlate the estimated capabilities (`theta` or Z-scores) with objective DORA metrics.
6.  Actionability: Use results (low scores/improvement, low info items, high correlations) to trigger investigations & refine surveys.
7.  Compositional Data: Analyze separately. Explore correlations with theta/metrics cautiously.
8.  Global Ranking: Avoid averaging ranks. Use capability scores. Focus on category-specific insights.
9.  Reorg Impact: Slope charts visualize transitions. Interpret capability change considering parent->child structure.
10. IIF/TIF Approximation: The plots use a simplified formula. For high-stakes decisions, verify information values using specialized IRT software.
""")

print("--- Script Finished ---")
