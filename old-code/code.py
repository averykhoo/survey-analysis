# -*- coding: utf-8 -*-
"""
Comprehensive MGRM Analysis Script for Survey Data

Purpose: Analyze Likert-scale survey responses using a Bayesian 
         Multidimensional Graded Response Model (MGRM) with correlated latent traits.
         Estimates latent traits (e.g., Usability, Performance) for different 
         products/teams over time, assesses improvement, and performs diagnostic checks.

Structure:
    0. Imports & Global Settings
    1. Survey Structure Definition
    2. Simulate Sample Data (Optional, for testing)
    3. Data Loading & Preprocessing
    4. Stan Model Definition (MGRM)
    5. Stan Helper Functions (Initial Values, PPC Simulation)
    6. Run Stan Model
    7. Diagnostic Checks Function
    8. Results Extraction & Processing
    9. Visualization Functions
   10. Main Execution Flow & Reporting
"""

import os  # For checking file existence

import arviz as az  # For diagnostics and visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan
import scipy.special as sps
import seaborn as sns  # Add seaborn import

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

# --- File Paths ---
OUTPUT_DIR = 'survey_analysis_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# STAN_MODEL_FILE = os.path.join(OUTPUT_DIR, 'mgr_model.pkl')  # To save compiled model

# -----------------------------------------
# Section 1: Survey Structure Definition
# -----------------------------------------
print("--- Section 1: Defining Survey Structure ---")
YEAR_COL = 'year'
ID_VAR = 'team_id'  # Changed from 'product' to 'team_id' as per context

# Define categories and their associated questions (MODIFY FOR YOUR ACTUAL SURVEY)
# This dictionary is a CORE ASSUMPTION of the model.
category_mapping = {
    'Deployment & Release':       ['q1', 'q2', 'q3'],  # Example DORA-related
    'Monitoring & Observability': ['q4', 'q5'],  # Example DORA-related
    'Technical Practices':        ['q6', 'q7', 'q8'],  # Example Eng Practices
    'Team Collaboration':         ['q9', 'q10'],  # Example Core/Shared
    'Process Efficiency':         ['q11'],  # Example Core/Shared
    'Learning & Development':     ['q12', 'q13'],  # Example Core/Shared
    # Add other relevant categories based on your survey
}

# --- Derive Helper Mappings ---
questions = []
for category_questions in category_mapping.values():
    questions.extend(category_questions)

question_categories = {}  # Map: question -> category index (1-based)
question_to_cat_name = {}  # Map: question -> category name
cat_idx = 1
for cat_name, cat_questions in category_mapping.items():
    for q in cat_questions:
        question_categories[q] = cat_idx
        question_to_cat_name[q] = cat_name
    cat_idx += 1

cat_idx_to_name = {idx: name for idx, name in enumerate(category_mapping.keys(), 1)}

# Ordered response options (Likert scale, e.g., 1 to 6) - MODIFY IF NEEDED
RESPONSE_OPTIONS = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE = len(RESPONSE_OPTIONS)

# Number of latent dimensions (categories)
N_LATENT = len(category_mapping)

print(f"Defined {N_LATENT} categories with {len(questions)} total questions.")
print(f"Response scale: {RESPONSE_OPTIONS}")


# -----------------------------------------
# Section 2: Simulate Sample Data
# -----------------------------------------
# This section creates dummy data for testing the script.
# Replace with your actual data loading in Section 3.
def simulate_data(years, teams, n_resp_per_team_year,
                  category_mapping, questions, question_categories,
                  response_options, n_latent):
    """Generates simulated survey data with correlated traits and improvement."""
    print("--- Section 2: Simulating Sample Data (for testing) ---")

    np.random.seed(RANDOM_SEED)
    n_categories_response = len(response_options)

    # Simulate a correlation matrix between latent dimensions
    base_corr = 0.3
    corr_matrix = np.ones((n_latent, n_latent)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = np.random.uniform(-0.15, 0.15, (n_latent, n_latent))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    corr_matrix = np.clip(corr_matrix + noise, 0.1, 0.9)
    np.fill_diagonal(corr_matrix, 1.0)
    print("Simulated True Correlation Matrix (for testing):")
    print(np.round(corr_matrix, 2))

    def generate_correlated_traits(mean_vector, cov_matrix):
        return np.random.multivariate_normal(mean_vector, cov_matrix)

    # Simulate latent traits for each team-year
    latent = {}
    for year in years:
        for team in teams:
            if year == years[0]:
                base_mean = np.random.normal(0, 0.5)  # Team-specific base level
                mean_vector = np.ones(n_latent) * base_mean
                traits = generate_correlated_traits(mean_vector, corr_matrix)
            else:
                # Simulate improvement from previous year
                previous_traits = np.array([latent[(team, years[0], cat_idx)] for cat_idx in range(1, n_latent + 1)])
                # Improvements vary by category and team slightly
                improvements = np.random.normal(0.5, 0.3, n_latent) + np.random.normal(0, 0.1)
                traits = previous_traits + improvements

            for cat_idx in range(1, n_latent + 1):
                latent[(team, year, cat_idx)] = traits[cat_idx - 1]

    # Simulate true item parameters
    true_a = {q: np.random.lognormal(0, 0.3) for q in questions}
    true_cutpoints = {}
    for q in questions:
        base = np.random.normal(0, 0.5)
        # Ensure cutpoints are reasonably spaced and cover typical range
        cp = np.sort(np.random.normal(base, 1.0, n_categories_response - 1))
        # Add buffer between cutpoints
        min_diff = 0.3
        for i in range(len(cp) - 1):
            if cp[i + 1] - cp[i] < min_diff:
                cp[i + 1] = cp[i] + min_diff
        true_cutpoints[q] = cp

    def simulate_response_gr(theta_val, a, cutpoints):
        """Simulates ordinal response using Graded Response Model."""
        logits = a * (theta_val - cutpoints)
        p_ge = np.concatenate(([1.0], sps.expit(logits), [0.0]))
        probs = p_ge[:-1] - p_ge[1:]
        # Handle potential numerical instability
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs /= probs.sum()  # Ensure sums to 1
        return np.random.choice(response_options, p=probs)

    # Generate response data
    data = []
    for year in years:
        for team in teams:
            for i in range(n_resp_per_team_year):
                row = {YEAR_COL: year, ID_VAR: team}
                for q in questions:
                    cat = question_categories[q]
                    theta_true = latent[(team, year, cat)]
                    a_val = true_a[q]
                    cp_val = true_cutpoints[q]
                    row[q] = simulate_response_gr(theta_true, a_val, cp_val)
                data.append(row)

    df = pd.DataFrame(data)
    print("\nSimulated Sample Data (first 5 rows):")
    print(df.head())
    return df


# --- Simulation Parameters (modify or remove when using real data) ---
SIMULATE = True  # Set to False to load real data
if SIMULATE:
    sim_years = [2023, 2024]
    sim_teams = [f'Team_{chr(65 + i)}' for i in range(10)]  # 10 teams
    sim_n_resp = 15  # responses per team-year

    df_raw = simulate_data(sim_years, sim_teams, sim_n_resp,
                           category_mapping, questions, question_categories,
                           RESPONSE_OPTIONS, N_LATENT)
else:
    # Placeholder for loading real data
    print("--- Section 2: Skipped Data Simulation ---")
    df_raw = None  # Assign your loaded DataFrame here

# -----------------------------------------
# Section 3: Data Loading & Preprocessing
# -----------------------------------------
print("\n--- Section 3: Loading & Preprocessing Data ---")

if df_raw is None:
    # --- !!! LOAD YOUR REAL DATA HERE !!! ---
    # Example: df_raw = pd.read_csv('your_survey_data.csv')
    # Ensure it has columns for YEAR_COL, ID_VAR, and all question IDs in `questions`
    print("ERROR: Real data loading not implemented. Please load data.")
    exit()
    # --- Perform any initial cleaning specific to your real data ---
    # E.g., renaming columns, converting types

# --- Reshape Data to Long Format ---
print("Reshaping data to long format...")
df_long = df_raw.melt(id_vars=[YEAR_COL, ID_VAR],
                      value_vars=questions,
                      var_name='question',
                      value_name='response')

# --- Handle Missing / NA Responses ---
# MGRM requires valid responses. Common strategies:
# 1. Filter out NA rows (simplest, used here). Assumes NA is Missing Completely At Random (MCAR)
#    or Missing At Random (MAR) - may introduce bias if NA pattern is informative.
# 2. Imputation (more complex, requires careful consideration of method).
# 3. Model 'NA' explicitly (most complex, requires changing Stan model, see earlier discussion).
# 4. Separate Analyses (Recommended if NA patterns differ systematically by group,
#    e.g., analyze Eng teams with Eng Qs, analyze All teams with Core Qs).

initial_rows = len(df_long)
df_long.dropna(subset=['response'], inplace=True)
# Convert response to integer AFTER dropping NAs
try:
    df_long['response'] = df_long['response'].astype(int)
    # Validate responses are within expected range
    is_valid_response = df_long['response'].isin(RESPONSE_OPTIONS)
    if not is_valid_response.all():
        print("WARNING: Some responses are outside the expected range after NA drop.")
        print(df_long[~is_valid_response])
        # Decide how to handle invalid responses (e.g., filter them out)
        df_long = df_long[is_valid_response]
except ValueError as e:
    print(f"ERROR converting 'response' to int: {e}. Check for non-numeric values.")
    exit()

rows_after_na = len(df_long)
print(f"Removed {initial_rows - rows_after_na} rows with NA responses.")
print("IMPORTANT: If NA rates differ systematically (e.g., DS vs Eng teams), "
      "filtering NAs might bias results. Consider separate analyses "
      "for relevant subgroups and question sets.")

# --- Add Category Information ---
df_long['q_cat'] = df_long['question'].map(question_categories)
df_long['category_name'] = df_long['question'].map(question_to_cat_name)

# --- Create Indices for Stan (1-based) ---
# Group index (unique team-year combination)
df_long['group'] = df_long[ID_VAR].astype(str) + "_" + df_long[YEAR_COL].astype(str)
groups = df_long['group'].unique()
group_idx_map = {g: i + 1 for i, g in enumerate(groups)}
df_long['group_idx'] = df_long['group'].map(group_idx_map)

group_idx_to_info = {idx: {"team": g.split('_')[0], "year": int(g.split('_')[1])}
                     for g, idx in group_idx_map.items()}

# Question index
questions_unique = df_long['question'].unique()  # Use questions present in the data
question_idx_map = {q: i + 1 for i, q in enumerate(questions_unique)}
df_long['question_idx'] = df_long['question'].map(question_idx_map)
# Re-map question_categories to use only questions present in the data if necessary
# This ensures q_cat corresponds to the filtered questions
q_cat_final = df_long[['question_idx', 'q_cat']].drop_duplicates().set_index('question_idx')['q_cat']

# --- Final Data Counts for Stan ---
N = df_long.shape[0]
J = len(groups)  # Number of unique team-year groups
K = len(questions_unique)  # Number of unique questions *present in data*
C = N_CATEGORIES_RESPONSE  # Number of response categories
L = N_LATENT  # Number of latent dimensions/categories

print("\nData prepared for Stan:")
print(f"  N (observations): {N}")
print(f"  J (groups/team-years): {J}")
print(f"  K (unique questions): {K}")
print(f"  C (response categories): {C}")
print(f"  L (latent dimensions): {L}")

print("\nPreprocessed Data Sample (first 5 rows):")
print(df_long.head())

# -----------------------------------------
# Section 4: Stan Model Definition (MGRM)
# -----------------------------------------
print("\n--- Section 4: Defining Stan Model ---")

stan_model_code = """
data {
  int<lower=1> N;           // number of observations (rows in long data)
  int<lower=1> J;           // number of groups (team-year combinations)
  int<lower=1> K;           // number of questions
  int<lower=2> C;           // number of response categories (e.g., 6 for 1-6)
  int<lower=1> L;           // number of latent dimensions (categories)
  
  // Data mapping
  array[N] int<lower=1, upper=J> group_idx; // group id for each observation
  array[N] int<lower=1, upper=K> q_idx;     // question id for each observation
  array[N] int<lower=1, upper=C> y;         // observed ordinal response
  
  // Mapping questions to latent dimensions
  // Ensure this array is indexed correctly (1 to K) in the data block
  array[K] int<lower=1, upper=L> question_to_dimension; 
}

parameters {
  // Latent traits (theta): Non-centered parameterization
  // Matrix of standard normal variables (J groups x L dimensions)
  matrix[L, J] z; // Transposed relative to some formulations for efficiency
  
  // Question parameters
  vector<lower=0>[K] a;        // discrimination parameters (1 per question)
  ordered[C-1] cutpoints[K];   // ordered cutpoints (C-1 per question)
  
  // Hierarchical priors for latent trait distribution
  vector[L] mu;                // mean for each latent dimension across groups
  vector<lower=0>[L] sigma;    // standard deviation for each dimension
  
  // Correlation structure between latent dimensions
  cholesky_factor_corr[L] L_Omega; // Cholesky factor of the correlation matrix
}

transformed parameters {
  // Calculate the actual latent traits (theta) for each group and dimension
  // theta = mu + diag(sigma) * L_Omega * z 
  // Need careful matrix multiplication order
  matrix[L, J] theta_raw = diag_pre_multiply(sigma, L_Omega) * z;
  matrix[J, L] theta; // Store as (groups x dimensions)
  for (j in 1:J) {
    for (l in 1:L) {
      theta[j, l] = mu[l] + theta_raw[l, j];
    }
  }
}

model {
  // Priors
  // Non-centered latent traits
  to_vector(z) ~ std_normal(); // Prior on the standardized latent variables
  
  // Hierarchical priors for theta distribution
  mu ~ normal(0, 1);      // Prior for overall mean of each dimension
  sigma ~ normal(0, 1);   // Prior for variability of each dimension (weakly regularizing)
                          // Consider half-Cauchy(0, 2.5) or exponential(1) as alternatives
  
  // Prior for correlation structure
  L_Omega ~ lkj_corr_cholesky(2); // LKJ prior (eta=2 favors identity but allows correlations)
                                  // eta=1 is uniform over correlation matrices
  
  // Priors for question parameters
  a ~ lognormal(0, 0.5); // Prior for discrimination (positive, typically moderate)
                         // Adjust mean/SD based on expected discrimination range
                         
  // Prior for cutpoints (ensure reasonable scale)
  // Centering cutpoints around 0 for each question can help
  for (k in 1:K) {
      cutpoints[k] ~ normal(0, 3); 
  }
  
  // Likelihood: Graded Response Model using ordered_logistic
  // Loop through observations
  for (n in 1:N) {
    int current_q = q_idx[n];       // Which question?
    int current_g = group_idx[n];   // Which group (team-year)?
    int current_dim = question_to_dimension[current_q]; // Which dimension does this question load on?
    
    // Calculate linear predictor (eta) for ordered logistic
    // eta = discrimination * (latent_trait - effective_difficulty)
    // ordered_logistic parameterization: eta = discrimination * latent_trait
    // cutpoints effectively handle the difficulty component
    real eta = a[current_q] * theta[current_g, current_dim];
    
    // Calculate likelihood of the observed response y[n]
    y[n] ~ ordered_logistic(eta, cutpoints[current_q]);
  }
}

generated quantities {
  // Recover the full correlation matrix
  matrix[L, L] Omega = multiply_lower_tri_self_transpose(L_Omega);
  
  // Calculate log likelihood for each observation (for LOOIC/WAIC)
  vector[N] log_lik;
  for (n in 1:N) {
    int current_q = q_idx[n];
    int current_g = group_idx[n];
    int current_dim = question_to_dimension[current_q];
    real eta = a[current_q] * theta[current_g, current_dim];
    log_lik[n] = ordered_logistic_lpmf(y[n] | eta, cutpoints[current_q]);
  }
}
"""

# -----------------------------------------
# Section 5: Stan Helper Functions
# -----------------------------------------
print("--- Section 5: Defining Stan Helper Functions ---")


def generate_init(num_chains, K_dims, L_dims, J_dims, C_cats):
    """Generate initial values for Stan MCMC chains."""
    list_of_inits = []
    for c in range(num_chains):
        # Generate reasonable starting points to aid convergence
        init = {
            'mu':        np.random.normal(0, 0.1, L_dims),
            'sigma':     np.abs(np.random.normal(0.5, 0.1, L_dims)),  # Ensure positive
            'a':         np.abs(np.random.normal(1, 0.1, K_dims)),  # Ensure positive
            'z':         np.random.normal(0, 0.1, (L_dims, J_dims)),
            # Initialize cutpoints ordered and reasonably spaced
            'cutpoints': [np.sort(np.random.normal(0, 1.5, C_cats - 1)) for _ in range(K_dims)],
            # Start L_Omega near identity (no correlation) - Cholesky factor is Identity matrix
            'L_Omega':   np.eye(L_dims)
        }
        list_of_inits.append(init)
    print(f"Generated initial values for {num_chains} chains.")
    return list_of_inits


def predict_responses(theta, a, cutpoints_samples, group_idx, q_idx, q_cat_mapping):
    """
    Generate predicted responses based on model parameters from ONE posterior sample.

    Args:
        theta (array): Theta estimates (J x L) for ONE sample.
        a (array): Discrimination estimates (K,) for ONE sample.
        cutpoints_samples (array): Cutpoint estimates (K x C-1) for ONE sample.
        group_idx (array): Group index for each observation (N,). 1-based.
        q_idx (array): Question index for each observation (N,). 1-based.
        q_cat_mapping (dict or pd.Series): Map from question index (K) to category index (L). 1-based.

    Returns:
        array: Predicted responses (N,).
    """
    N_obs = len(q_idx)
    preds = np.zeros(N_obs, dtype=int)
    n_categories_resp = cutpoints_samples.shape[1] + 1
    response_opts = list(range(1, n_categories_resp + 1))

    for i in range(N_obs):
        g = group_idx[i] - 1  # Convert to 0-based index for Python
        q = q_idx[i] - 1  # Convert to 0-based index for Python
        # Get the latent dimension for this question (using 1-based question index)
        dim = q_cat_mapping[q + 1] - 1  # Convert to 0-based index

        # Calculate eta using parameters from the single posterior sample
        eta = a[q] * theta[g, dim]

        # Calculate category probabilities using ordered logistic logic
        current_cutpoints = cutpoints_samples[q]

        # Cumulative probabilities P(y <= k)
        cumulative_probs = sps.expit(current_cutpoints - eta)  # Uses scipy.special.expit (logistic sigmoid)

        # P(y = 1) = P(y <= 1)
        # P(y = k) = P(y <= k) - P(y <= k-1) for k > 1
        # P(y = C) = 1 - P(y <= C-1)

        probs = np.zeros(n_categories_resp)
        probs[0] = cumulative_probs[0]
        for k in range(1, n_categories_resp - 1):
            probs[k] = cumulative_probs[k] - cumulative_probs[k - 1]
        probs[n_categories_resp - 1] = 1.0 - cumulative_probs[n_categories_resp - 2]

        # Handle potential numerical inaccuracies
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs /= probs.sum()  # Ensure sums to 1

        # Sample predicted response
        try:
            preds[i] = np.random.choice(response_opts, p=probs)
        except ValueError:
            # If sum(probs) is not exactly 1 due to float issues
            # print(f"Warning: Probability sum issue for obs {i}. Probs: {probs}. Sum: {probs.sum()}. Error: {e}")
            # Fallback: normalize again or assign most likely category
            probs /= probs.sum()
            preds[i] = np.random.choice(response_opts, p=probs)

    return preds


# -----------------------------------------
# Section 6: Run Stan Model
# -----------------------------------------
print("\n--- Section 6: Preparing Data and Running Stan Model ---")

# Prepare data dictionary for Stan
stan_data = {
    'N':                     N,
    'J':                     J,
    'K':                     K,
    'C':                     C,
    'L':                     L,
    'group_idx':             df_long['group_idx'].values,
    'q_idx':                 df_long['question_idx'].values,
    'y':                     df_long['response'].values,
    # Map from question index (k=1..K) to dimension index (l=1..L)
    'question_to_dimension': q_cat_final.values.astype(int)
}

# # Check if a compiled model exists
# if os.path.exists(STAN_MODEL_FILE):
#     print(f"Loading compiled Stan model from: {STAN_MODEL_FILE}")
#     try:
#         sm = pystan.load(STAN_MODEL_FILE)
#         if sm.model_code != stan_model_code:
#             print("WARNING: Saved model code differs from current code. Recompiling.")
#             sm = pystan.StanModel(model_code=stan_model_code)
#             pystan.dump(sm, STAN_MODEL_FILE)  # Save the new compiled model
#     except Exception as e:
#         print(f"Error loading compiled model: {e}. Recompiling.")
#         sm = pystan.StanModel(model_code=stan_model_code)
#         pystan.dump(sm, STAN_MODEL_FILE)  # Save the new compiled model
# else:
print("Compiling Stan model (this may take several minutes)...")
sm = pystan.StanModel(model_code=stan_model_code)
# pystan.dump(sm, STAN_MODEL_FILE)  # Save the compiled model
# print(f"Compiled model saved to: {STAN_MODEL_FILE}")

# Generate initial values
print("Generating initial values...")
inits = generate_init(CHAINS, K, L, J, C)

# Run MCMC sampler
print(f"Running MCMC sampling ({CHAINS} chains, {ITER_WARMUP} warmup, {ITER_SAMPLING} sampling)...")

fit = sm.sampling(
    data=stan_data,
    iter=ITER_WARMUP + ITER_SAMPLING,
    warmup=ITER_WARMUP,
    chains=CHAINS,
    thin=THIN,
    seed=RANDOM_SEED,
    init=inits,
    # Add control parameters if divergences occur
    # control={'adapt_delta': 0.9} # Default is 0.8, increase if divergences
)

print("Sampling complete.")

# Basic Stan fit summary (can be very verbose)
# print("\nRaw Stan Fit Summary (Excerpt):")
# print(fit.stansummary(pars=['mu', 'sigma', 'Omega'], probs=[0.025, 0.5, 0.975]))


# -----------------------------------------
# Section 7: Diagnostic Checks Function
# -----------------------------------------
print("\n--- Section 7: Defining Diagnostic Checks Function ---")


def run_diagnostic_checks(fit, samples, stan_data, df_long,
                          category_mapping, question_to_cat_name, cat_idx_to_name,
                          question_idx_map, group_idx_to_info,
                          n_chains, iter_sampling):
    """
    Runs comprehensive diagnostic checks on the Stan fit object and samples.
    """
    print("\n--- RUNNING DIAGNOSTIC CHECKS ---")
    warnings_found = []
    total_post_warmup_samples = n_chains * iter_sampling
    K_dims = stan_data['K']
    C_cats = stan_data['C']
    response_opts = list(range(1, C_cats + 1))

    # Convert fit object to ArviZ InferenceData
    try:
        idata = az.from_pystan(posterior=fit)
        print("Successfully converted fit object to ArviZ InferenceData.")
    except Exception as e:
        print(f"WARNING: Could not automatically convert fit to ArviZ InferenceData: {e}")
        print("         Attempting checks using az.summary on fit object directly.")
        idata = fit  # Use fit object directly if conversion fails

    # --- 1. MCMC Convergence Checks ---
    print("\n--- Checking MCMC Convergence ---")
    try:
        summary = az.summary(idata, round_to=3)
        print("ArviZ summary generated.")

        # Check for Divergences (Extract from sample_stats if possible)
        try:
            # Need to ensure group 'sample_stats' exists in idata
            if hasattr(idata, 'sample_stats') and 'diverging' in idata.sample_stats:
                divergences = idata.sample_stats.diverging.sum().item()
                if divergences > 0:
                    msg = (f"CRITICAL WARNING: {divergences} divergent transitions found! "
                           "Results are unreliable. Increase adapt_delta "
                           "(e.g., control={'adapt_delta': 0.95}) or reparameterize.")
                    warnings_found.append(msg)
                    print(msg)
                else:
                    print("  Divergences: OK (0 found)")
            else:
                print("  INFO: Could not access divergences in sample_stats.")
                # Check stanfit directly for divergences
                sampler_params = fit.get_sampler_params(inc_warmup=False)
                divergences = sum(p['divergent__'].sum() for p in sampler_params)
                if divergences > 0:
                    msg = (
                        f"CRITICAL WARNING: {divergences} divergent transitions found (checked via get_sampler_params)! "
                        "Results are unreliable. Increase adapt_delta (e.g., control={'adapt_delta': 0.95}) or reparameterize.")
                    warnings_found.append(msg)
                    print(msg)
                else:
                    print("  Divergences: OK (0 found via get_sampler_params)")


        except Exception as e:
            print(f"  WARNING: Error checking divergences: {e}")

        # Check R-hat
        high_rhat_params = summary[summary['r_hat'] > RHAT_THRESHOLD]
        if not high_rhat_params.empty:
            msg = (f"WARNING: {len(high_rhat_params)} parameters have R-hat > {RHAT_THRESHOLD}. "
                   "Chains have not converged well for these. Check trace plots. "
                   f"Problematic parameters (first 10):\n{high_rhat_params.index.tolist()[:10]}")
            warnings_found.append(msg)
            print(msg)
        else:
            print(f"  R-hat: OK (All <= {RHAT_THRESHOLD})")

        # Check Effective Sample Size (n_eff)
        min_expected_neff = NEFF_RATIO_THRESHOLD * total_post_warmup_samples
        # Check both bulk and tail ESS
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
            print(f"  n_eff (ess_bulk & ess_tail): OK (All >= {min_expected_neff:.0f})")

    except Exception as e:
        print(f"ERROR during MCMC summary/checks: {e}. Cannot proceed with detailed diagnostics.")
        warnings_found.append("ERROR: Could not generate ArviZ summary for MCMC checks.")
        return warnings_found  # Exit diagnostics early

    # --- 2. Trace Plot ("Fuzzy Caterpillar") Check ---
    print("\n--- Generating Trace Plots (Manual Check Required) ---")
    try:
        # Plot a manageable subset of parameters
        params_to_plot = ['mu', 'sigma']
        k_q = K_dims
        j_g = len(group_idx_to_info)
        l_c = len(cat_idx_to_name)
        # Add examples of 'a', 'theta', 'Omega' if they exist in summary
        if 'a[1]' in summary.index: params_to_plot.extend([f'a[{i + 1}]' for i in range(min(k_q, 3))])
        if 'theta[1,1]' in summary.index: params_to_plot.extend(
            [f'theta[{j + 1},{l + 1}]' for j in range(min(j_g, 1)) for l in range(min(l_c, 2))])
        if 'Omega[1,1]' in summary.index: params_to_plot.extend(
            [f'Omega[{i + 1},{j + 1}]' for i in range(min(l_c, 2)) for j in range(i, min(l_c, 2))])

        az.plot_trace(idata, var_names=params_to_plot)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'trace_plots_subset.png'))
        plt.show()
        msg = ("INFO: Trace plots generated for a subset of parameters (see trace_plots_subset.png). "
               "MANUALLY INSPECT them for good mixing (overlapping chains, 'fuzzy caterpillar' shape) "
               "and no long-term trends.")
        print(msg)
    except Exception as e:
        msg = f"WARNING: Could not generate trace plots: {e}. Manual inspection via ArviZ is recommended."
        warnings_found.append(msg)
        print(msg)

    # --- 3. Model Fit Checks ---
    print("\n--- Checking Model Fit ---")
    # Posterior Predictive Check (Histogram - Basic Overall Check)
    print("  Generating Overall Posterior Predictive Check histogram (Manual Check Required)...")
    if 'theta' in samples and 'a' in samples and 'cutpoints' in samples:
        try:
            n_samples_total = len(samples['theta'])
            sample_idx = np.random.randint(0, n_samples_total)
            sample_theta = samples['theta'][sample_idx]
            sample_a = samples['a'][sample_idx]
            sample_cut = samples['cutpoints'][sample_idx]

            # Get the final q_cat mapping based on questions actually in the model
            q_cat_map_series = df_long[['question_idx', 'q_cat']].drop_duplicates().set_index('question_idx')['q_cat']

            pred_responses = predict_responses(
                sample_theta, sample_a, sample_cut,
                stan_data['group_idx'], stan_data['q_idx'], q_cat_map_series
            )

            plt.figure(figsize=(12, 6))
            # Use density=True for better comparison if group sizes differ
            plt.subplot(1, 2, 1)
            plt.hist(stan_data['y'], bins=np.arange(0.5, C_cats + 1.5, 1), alpha=0.7, label='Observed', density=True)
            plt.title('Observed Response Distribution (Overall)')
            plt.xlabel('Response Category')
            plt.ylabel('Density')
            plt.xticks(response_opts)

            plt.subplot(1, 2, 2)
            plt.hist(pred_responses, bins=np.arange(0.5, C_cats + 1.5, 1), alpha=0.7, label='Predicted (1 Sample)',
                     density=True)
            plt.title('Predicted Response Distribution (from 1 posterior sample)')
            plt.xlabel('Response Category')
            plt.ylabel('Density')
            plt.xticks(response_opts)

            plt.suptitle("Posterior Predictive Check (Overall Distribution)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
            plt.savefig(os.path.join(OUTPUT_DIR, 'ppc_histogram_overall.png'))
            plt.show()
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
    if 'log_lik' in samples and len(samples['log_lik']) > 0:
        print("  Calculating LOOIC using ArviZ (may take time)...")
        try:
            # Ensure log_lik is correctly shaped for ArviZ (chains, draws, N)
            log_lik_data = az.convert_to_dataset({'log_lik': samples['log_lik']})

            # Use the idata object which should contain log_lik if generated quantities ran
            loo_result = az.loo(idata, pointwise=True, var_name='log_lik')
            print("\nLOOIC Results:")
            print(loo_result)
            az.plot_khat(loo_result, threshold=PARETO_K_THRESHOLD)
            plt.savefig(os.path.join(OUTPUT_DIR, 'looic_pareto_k_diagnostic.png'))
            plt.show()

            # Check Pareto k values
            problematic_k_count = np.sum(loo_result.pareto_k > PARETO_K_THRESHOLD)
            if problematic_k_count > 0:
                msg = (f"WARNING: {problematic_k_count} observations have Pareto k > {PARETO_K_THRESHOLD}. "
                       f"LOOIC estimate may be unreliable. These points are highly influential or poorly fit. "
                       "Inspect these data points and consider model refinement. "
                       f"(See looic_pareto_k_diagnostic.png)")
                warnings_found.append(msg)
                print(msg)
            else:
                print(f"  Pareto k diagnostic: OK (All <= {PARETO_K_THRESHOLD})")

        except Exception as e:
            msg = f"WARNING: Could not calculate LOOIC: {e}. Ensure 'log_lik' is correctly computed and available in idata."
            warnings_found.append(msg)
            print(msg)
    else:
        print("  INFO: 'log_lik' not found or empty in samples. Skipping LOOIC calculation.")

    # --- 4. Parameter Estimate Checks ---
    print("\n--- Checking Parameter Estimates ---")

    # Check for High Correlations (Suggest Merging Categories)
    print(f"  Checking for category correlations > {HIGH_CORR_THRESHOLD}...")
    if 'Omega' in samples:
        Omega_mean = samples['Omega'].mean(axis=0)
        merged_suggestions = []
        categories = list(cat_idx_to_name.values())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                if Omega_mean[i, j] > HIGH_CORR_THRESHOLD:
                    merged_suggestions.append((categories[i], categories[j], Omega_mean[i, j]))

        if merged_suggestions:
            msg = (
                f"INFO: Found {len(merged_suggestions)} pairs of categories with mean correlation > {HIGH_CORR_THRESHOLD}. "
                "Consider if merging these categories makes theoretical sense:")
            print(msg)
            # warnings_found.append(msg) # Info, not necessarily a warning
            for cat1, cat2, corr in merged_suggestions:
                print(f"      - '{cat1}' and '{cat2}' (Corr: {corr:.3f})")
        else:
            print("  High Correlations (> {HIGH_CORR_THRESHOLD}): None found.")

        # Print estimated correlation matrix for reference
        category_names_list = list(category_mapping.keys())
        corr_df = pd.DataFrame(Omega_mean, index=category_names_list, columns=category_names_list)
        print("\n  Estimated Mean Correlation Matrix ('Omega'):")
        print(corr_df.round(2))
        corr_df.to_csv(os.path.join(OUTPUT_DIR, 'category_correlations.csv'))
        print(f"  Correlation matrix saved to {os.path.join(OUTPUT_DIR, 'category_correlations.csv')}")

    else:
        print("  INFO: 'Omega' not found in samples. Skipping correlation check.")

    # Check for Low Discrimination Questions
    print(f"  Checking for questions with mean discrimination ('a') < {LOW_DISCRIMINATION_THRESHOLD}...")
    if 'a' in samples:
        a_mean = samples['a'].mean(axis=0)
        idx_to_question = {v: k for k, v in question_idx_map.items()}
        low_a_questions = []
        # Ensure a_mean length matches K_dims
        if len(a_mean) == K_dims:
            for k_idx_0based in range(K_dims):
                stan_idx = k_idx_0based + 1  # Stan uses 1-based indexing
                if a_mean[k_idx_0based] < LOW_DISCRIMINATION_THRESHOLD:
                    q_id = idx_to_question.get(stan_idx, f"Unknown Q (Index {stan_idx})")
                    cat_name = question_to_cat_name.get(q_id, "Unknown Cat")
                    low_a_questions.append({'Question': q_id, 'Category': cat_name, 'Mean_a': a_mean[k_idx_0based]})

            if low_a_questions:
                msg = (
                    f"INFO: Found {len(low_a_questions)} questions with mean discrimination ('a') < {LOW_DISCRIMINATION_THRESHOLD}. "
                    "These questions may weakly distinguish trait levels. "
                    "Consider for revision/removal in future surveys:")
                print(msg)
                # warnings_found.append(msg) # Info, not necessarily a warning
                low_a_df = pd.DataFrame(low_a_questions).sort_values('Mean_a')
                print(low_a_df.round(3))
            else:
                print(f"  Low Discrimination (< {LOW_DISCRIMINATION_THRESHOLD}): None found.")
        else:
            print(
                f"  WARNING: Length mismatch between mean 'a' ({len(a_mean)}) and K ({K_dims}). Skipping low discrimination check.")
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

    return warnings_found


# -----------------------------------------
# Section 8: Results Extraction & Processing
# -----------------------------------------
print("\n--- Section 8: Extracting and Processing Results ---")

# Extract posterior samples
# Handle potential errors if sampling failed
try:
    samples = fit.extract(permuted=True)  # permuted=True mixes samples across chains
    print("Posterior samples extracted.")
except Exception as e:
    print(f"ERROR: Could not extract samples from fit object: {e}")
    print("Cannot proceed with results processing.")
    exit()

# --- Theta Estimates ---
if 'theta' in samples:
    theta_samples = samples['theta']  # Shape (n_samples, J, L)
    # Compute posterior means for theta per group and per latent dimension
    theta_means = theta_samples.mean(axis=0)  # shape (J, L)

    # Create a DataFrame with one row per group and named columns for each category
    theta_columns = [f"theta_{cat_idx_to_name[l + 1]}" for l in range(N_LATENT)]
    theta_df = pd.DataFrame(theta_means, columns=theta_columns)
    # Add group info back
    theta_df['group_idx'] = list(group_idx_map.values())  # Ensure order matches J dimension
    theta_df['group'] = theta_df['group_idx'].map({v: k for k, v in group_idx_map.items()})
    theta_df[[ID_VAR, YEAR_COL]] = theta_df['group'].str.split('_', expand=True)
    theta_df[YEAR_COL] = theta_df[YEAR_COL].astype(int)
    # Reorder columns for clarity
    theta_df = theta_df[[ID_VAR, YEAR_COL, 'group', 'group_idx'] + theta_columns]

    print("\nEstimated latent traits per team-year (Posterior Means):")
    print(theta_df.head())
    theta_df.to_csv(os.path.join(OUTPUT_DIR, 'latent_trait_estimates.csv'), index=False)
    print(f"Latent trait estimates saved to {os.path.join(OUTPUT_DIR, 'latent_trait_estimates.csv')}")

else:
    print("WARNING: 'theta' not found in samples. Cannot process latent trait estimates.")
    theta_df = None

# --- Correlation Matrix Estimate ---
if 'Omega' in samples:
    Omega_samples = samples['Omega']  # Shape (n_samples, L, L)
    Omega_mean = Omega_samples.mean(axis=0)
    category_names_list = list(category_mapping.keys())
    corr_df = pd.DataFrame(Omega_mean, index=category_names_list, columns=category_names_list)
    # Already printed and saved during diagnostics, but can save again if needed
    # corr_df.to_csv(os.path.join(OUTPUT_DIR, 'category_correlations.csv'))
else:
    print("WARNING: 'Omega' not found in samples. Cannot process correlation matrix.")
    corr_df = None


# --- Ranking and Improvement Summary ---
def create_ranking_summary(theta_df, id_var, year_col, category_mapping, years):
    """Creates a summary DataFrame of rankings and improvements."""
    if theta_df is None or theta_df.empty:
        print("INFO: theta_df is empty, skipping ranking summary.")
        return pd.DataFrame()

    summary_data = []
    unique_ids = theta_df[id_var].unique()

    # Ensure years are sorted
    years = sorted(years)
    year1, year2 = years[0], years[1]

    for cat_name in category_mapping.keys():
        col_name = f"theta_{cat_name}"
        if col_name not in theta_df.columns:
            print(f"Warning: Column {col_name} not found in theta_df. Skipping category {cat_name}.")
            continue

        # Get data for both years
        df_y1 = theta_df[theta_df[year_col] == year1].set_index(id_var)
        df_y2 = theta_df[theta_df[year_col] == year2].set_index(id_var)

        # Calculate ranks (handle missing ids if any)
        rank_y1 = df_y1[col_name].rank(ascending=False, method='min').astype(int)
        rank_y2 = df_y2[col_name].rank(ascending=False, method='min').astype(int)

        for team_id in unique_ids:
            # Get values, handling cases where a team might be missing in one year
            val_y1 = df_y1.loc[team_id, col_name] if team_id in df_y1.index else np.nan
            val_y2 = df_y2.loc[team_id, col_name] if team_id in df_y2.index else np.nan

            # Calculate improvement only if both years present
            improvement = val_y2 - val_y1 if pd.notna(val_y1) and pd.notna(val_y2) else np.nan

            r_y1 = rank_y1.get(team_id, np.nan)
            r_y2 = rank_y2.get(team_id, np.nan)

            # Rank change (positive means improved ranking)
            rank_change = r_y1 - r_y2 if pd.notna(r_y1) and pd.notna(r_y2) else np.nan

            summary_data.append({
                'Team':           team_id,
                'Category':       cat_name,
                f'Value_{year1}': val_y1,
                f'Value_{year2}': val_y2,
                'Improvement':    improvement,
                f'Rank_{year1}':  r_y1,
                f'Rank_{year2}':  r_y2,
                'Rank_Change':    rank_change
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


# Create and save the summary (assuming theta_df was created)
if theta_df is not None:
    # Determine years present in the data
    actual_years = sorted(theta_df[YEAR_COL].unique())
    if len(actual_years) >= 2:
        ranking_summary = create_ranking_summary(theta_df, ID_VAR, YEAR_COL, category_mapping, actual_years)
        print("\nTeam Rankings and Improvements by Category (Excerpt):")
        print(ranking_summary.sort_values(['Category', f'Rank_{actual_years[-1]}']).head(10))
        ranking_summary.to_csv(os.path.join(OUTPUT_DIR, 'team_rankings_by_category.csv'), index=False)
        print(f"Ranking summary saved to {os.path.join(OUTPUT_DIR, 'team_rankings_by_category.csv')}")
    else:
        print("INFO: Fewer than two years of data found in theta_df. Cannot calculate improvement or rank change.")
        ranking_summary = pd.DataFrame()
else:
    ranking_summary = pd.DataFrame()

# -----------------------------------------
# Section 9: Visualization Functions
# -----------------------------------------
print("\n--- Section 9: Defining Visualization Functions ---")


def plot_category_slope_with_mapping(theta_df,
                                     category_name,
                                     id_var, 
                                     year_col,
                                     mapping=None):
    """
    Draws a slope chart for `category_name`, but for each original id
    uses the mapped target to look up its second‐year value.

    mapping should be a dict: { old_id: new_id, … }
    """
    col = f"theta_{category_name}"
    years = sorted(theta_df[year_col].unique())
    if len(years) != 2:
        raise ValueError("Expect exactly two years for a slope chart")
    y1, y2 = years

    # Build a small DataFrame of (team, val1, val2)
    rows = []
    for team in theta_df[id_var].unique():
        # get year1 value
        v1 = theta_df.loc[
            (theta_df[id_var] == team) & (theta_df[year_col] == y1), col
        ]
        if v1.empty:
            continue

        # look up its mapped name in year2 (fall back to itself)
        new_name = mapping.get(team, team) if mapping else team
        v2 = theta_df.loc[
            (theta_df[id_var] == new_name) & (theta_df[year_col] == y2), col
        ]
        if v2.empty:
            continue

        rows.append({
            'team': team,
            'val1': v1.iloc[0],
            'val2': v2.iloc[0]
        })

    df_plot = pd.DataFrame(rows).sort_values('val2').reset_index(drop=True)
    if df_plot.empty:
        print(f"No overlapping data for {category_name} after mapping!")
        return

    # now do the slope lines
    plt.figure(figsize=(10, max(6, len(df_plot) * 0.4)))
    for _, r in df_plot.iterrows():
        plt.plot([y1, y2], [r.val1, r.val2],
                 marker='o', linewidth=1.5, markersize=5)
        # left label = original team
        plt.text(y1 - 0.1, r.val1, r.team,
                 horizontalalignment='right', verticalalignment='center')
        # right label = same team (so names stay as-is)
        plt.text(y2 + 0.1, r.val2, r.team,
                 horizontalalignment='left', verticalalignment='center')

    plt.title(f"{category_name}: {y1} vs {y2}")
    plt.xlabel("Year")
    plt.ylabel(f"Estimated θ ({category_name})")
    plt.xticks([y1, y2])
    plt.xlim(y1 - 0.5, y2 + 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def plot_omega_clustermap(corr_df, output_dir):
    """
    Visualizes the correlation matrix Omega as a clustered heatmap.

    Args:
        corr_df (pd.DataFrame): DataFrame containing the mean correlation matrix
                                with category names as index and columns.
        output_dir (str): Path to the directory to save the plot.
    """
    if corr_df is None or corr_df.empty:
        print("INFO: Correlation matrix DataFrame is empty. Skipping Omega clustermap.")
        return

    n_cats = corr_df.shape[0]
    if n_cats < 2:
        print("INFO: Need at least 2 categories to create a clustermap. Skipping Omega plot.")
        return

    print("\n--- Generating Clustered Heatmap for Omega ---")

    # Choose appropriate figure size
    figsize = (max(8, n_cats * 0.8), max(7, n_cats * 0.7))  # Adjust size based on number of categories

    # Determine if annotations are feasible
    annotate = n_cats <= 15  # Only annotate if not too many categories

    try:
        # Create the clustermap
        # 'average' linkage (UPGMA) and 'euclidean' distance on correlations are common defaults
        # 'ward' linkage often produces tighter clusters
        # 'correlation' metric uses 1-|corr| as distance
        cluster_grid = sns.clustermap(
            corr_df,
            method='average',  # Linkage method: 'average', 'ward', 'complete', etc.
            metric='euclidean',  # Distance metric: 'euclidean', 'correlation', etc.
            cmap='vlag',  # Diverging colormap (good for correlations around 0)
            vmin=-1, vmax=1,  # Set scale for correlation
            center=0,  # Center colormap at 0
            annot=annotate,  # Show values in cells if not too large
            fmt=".2f",  # Format for annotations
            linewidths=.5,
            linecolor='lightgray',
            figsize=figsize
        )

        # Adjustments for better readability
        plt.setp(cluster_grid.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)

        # Add a title 
        cluster_grid.fig.suptitle('Clustered Heatmap of Category Correlations (Mean Omega)',
                                  y=1.02)  # Adjust y slightly above plot

        # Save the figure
        filepath = os.path.join(output_dir, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight')  # Use bbox_inches='tight'
        plt.show()
        print(f"Omega clustermap saved to {filepath}")

        print("\nInterpretation Notes for Omega Clustermap:")
        print("- Colors indicate correlation strength (e.g., red=positive, blue=negative).")
        print("- Dendrograms (tree diagrams) show the hierarchical clustering.")
        print(
            "- Categories grouped closely together in the dendrogram and heatmap tend to have similar correlation patterns with other categories.")
        print(
            "- Look for distinct blocks along the diagonal, which represent clusters of highly correlated categories.")

    except Exception as e:
        print(f"WARNING: Could not generate Omega clustermap: {e}")


# -----------------------------------------
# Section 10: Main Execution Flow & Reporting
# -----------------------------------------
print("\n--- Section 10: Running Main Analysis Flow ---")

# --- Perform Diagnostics ---
# Diagnostics are run *after* sampling is complete
diagnostic_warnings = run_diagnostic_checks(
    fit=fit,
    samples=samples,
    stan_data=stan_data,
    df_long=df_long,
    category_mapping=category_mapping,
    question_to_cat_name=question_to_cat_name,
    cat_idx_to_name=cat_idx_to_name,
    question_idx_map=question_idx_map,
    group_idx_to_info=group_idx_to_info,
    n_chains=CHAINS,
    iter_sampling=ITER_SAMPLING
)

# --- Generate Visualizations ---
print("\n--- Generating Visualizations ---")
if theta_df is not None:
    actual_years = sorted(theta_df[YEAR_COL].unique())
    if len(actual_years) >= 2:
        # Plot slope charts for each category comparing first and last year
        years_to_plot = [actual_years[0], actual_years[-1]]
        for cat_name in category_mapping.keys():
            plot_category_slope_chart(theta_df, cat_name, ID_VAR, YEAR_COL, years_to_plot)
    else:
        print("INFO: Skipping slope charts as less than two years of data are available.")
else:
    print("INFO: Skipping visualizations as theta_df could not be generated.")

# Generate Omega clustermap
if 'corr_df' in locals() and corr_df is not None:
    plot_omega_clustermap(corr_df, OUTPUT_DIR)
else:
    print("INFO: Skipping Omega clustermap as corr_df was not generated.")

# --- Final Comments & Next Steps ---
print("\n--- Analysis Complete ---")
print(f"Results, diagnostics, and plots saved in: {OUTPUT_DIR}")

if any("CRITICAL" in w for w in diagnostic_warnings):
    print(
        "\n*** ACTION REQUIRED: Critical warnings detected during diagnostics. Review output carefully. Results may be unreliable. ***")
elif diagnostic_warnings:
    print("\nNOTE: Some potential issues were flagged during diagnostics. Review warnings and manual checks.")

print("\nNext Steps & Considerations:")
print("1. Review Diagnostics: Carefully check R-hat, n_eff, divergences, trace plots, PPC, LOOIC Pareto k values.")
print(
    "2. Interpret Results: Analyze the theta_df for team performance, ranking_summary for improvements, and corr_df for category relationships.")
print(
    "3. Category Structure: Evaluate if categories should be merged (high Omega correlation) or split (based on item parameters/theory).")
print(
    "4. Prior Sensitivity: Consider running the model with slightly different priors for key parameters (e.g., LKJ eta, sigma/a priors) to check stability.")
print(
    "5. Global Ranking: If needed, use weighted averages of standardized theta scores or tiering based on multiple categories. Avoid averaging ranks or using a simplistic unidimensional model unless justified.")
print(
    "6. Compositional Data: Analyze percentage breakdown data separately using appropriate methods (descriptives, log-ratio transforms) and potentially correlate post-hoc with theta estimates.")
print("7. Qualitative Follow-up: Use quantitative findings to inform targeted qualitative investigation.")
print(
    "8. NA Handling Strategy: If running separate analyses for different team types/question sets, adapt this script accordingly.")
