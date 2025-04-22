import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import scipy.special as sps

# -------------------------------
# 1. Define Column Variables
# -------------------------------
year_col = 'year'
id_var = 'product'
questions = ['q1', 'q2', 'q3']  # you can modify this list as needed
response_options = [1, 2, 3, 4, 5, 6]  # ordinal responses

# -------------------------------
# 2. Simulate Sample Data
# -------------------------------
# Settings: two years, a few products, and n_resp responses per product-year.
years = [2023, 2025]
products = ['ProductA', 'ProductB', 'ProductC', 'ProductD']
n_resp = 6  # responses per product-year

# For simulation, assign each product–year a "true" latent ability (theta)
np.random.seed(42)
latent = {}
for year in years:
    for prod in products:
        latent[(prod, year)] = np.random.normal(0, 1)

# For each question, we assume a discrimination (a) and set common thresholds (cutpoints)
true_a = {q: 1.0 for q in questions}
# Define ordered thresholds for a 6-category response (5 cutpoints)
true_cutpoints = {q: np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) for q in questions}


def simulate_response(theta, a, cutpoints):
    """
    Simulate one ordinal response using the graded response (ordered logistic) model.
    We compute:
      P(Y>=k) = logistic(a * (theta - cutpoints[k-1])) for k = 2...6,
    with P(Y>=1)=1 and P(Y>=7)=0.
    Then, the probability for category k is the difference P(Y>=k) - P(Y>=k+1).
    """
    # Compute logistic probabilities for each threshold.
    # Note: For a 6-category scale, there are 5 thresholds.
    logits = a * (theta - cutpoints)
    # Compute cumulative probabilities: for category 1, p_ge[0] is 1,
    # then logistic for each threshold, and p_ge[6] is 0.
    p_ge = np.concatenate(([1.0], sps.expit(logits), [0.0]))
    probs = p_ge[:-1] - p_ge[1:]
    # Sample a category from 1 to 6 based on the computed probabilities.
    return np.random.choice(response_options, p=probs)


# Create a wide-format dataframe (one row per response)
data = []
for year in years:
    for prod in products:
        for i in range(n_resp):
            row = {year_col: year, id_var: prod}
            for q in questions:
                theta_true = latent[(prod, year)]
                a_val = true_a[q]
                cp_val = true_cutpoints[q]
                row[q] = simulate_response(theta_true, a_val, cp_val)
            data.append(row)

df = pd.DataFrame(data)
print("Wide-format sample data:")
print(df.head())

# -------------------------------
# 3. Reshape Data for the IRT Model
# -------------------------------
# Melt the dataframe so that each observation is one product-year, question, response triplet.
df_long = df.melt(id_vars=[year_col, id_var],
                  value_vars=questions,
                  var_name='question',
                  value_name='response')
df_long['response'] = df_long['response'].astype(int)
# The OrderedLogistic likelihood in PyMC assumes responses start at 0, so subtract 1.
df_long['response'] = df_long['response'] - 1

# Create an index for each unique product-year combination.
df_long['group'] = df_long[id_var].astype(str) + "_" + df_long[year_col].astype(str)
groups = df_long['group'].unique()
group_idx = {g: i for i, g in enumerate(groups)}
df_long['group_idx'] = df_long['group'].map(group_idx)

# Create an index for each question.
questions_unique = df_long['question'].unique()
question_idx = {q: i for i, q in enumerate(questions_unique)}
df_long['question_idx'] = df_long['question'].map(question_idx)

# For reference:
n_groups = len(groups)
n_questions = len(questions_unique)
n_obs = len(df_long)
n_categories = len(response_options)  # should be 6

# -------------------------------
# 4. Build & Run the IRT Model in PyMC
# -------------------------------
with pm.Model() as irt_model:
    # Latent trait for each product-year group:
    theta = pm.Normal('theta', mu=0, sigma=1, shape=n_groups)

    # Discrimination parameters for each question: constrain to be positive.
    a = pm.HalfNormal('a', sigma=1, shape=n_questions)

    # Item thresholds (cutpoints) for each question.
    # There will be n_categories - 1 = 5 cutpoints per question.
    cutpoints = pm.Normal('cutpoints', mu=0, sigma=3, shape=(n_questions, n_categories - 1),
                          transform=pm.distributions.transforms.ordered)

    # For each observation i, the latent predictor is a[question] * theta[group]
    eta = a[df_long['question_idx'].values] * theta[df_long['group_idx'].values]

    # Use the OrderedLogistic likelihood.
    # Note: We extract the appropriate row of cutpoints using advanced indexing.
    obs = pm.OrderedLogistic('obs',
                             eta=eta,
                             cutpoints=cutpoints[df_long['question_idx'].values],
                             observed=df_long['response'].values)

    # Sample from the posterior (for a real analysis, you might increase draws and tune parameters further)
    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)

# -------------------------------
# 5. Extract Theta Estimates per Product-Year Group
# -------------------------------
# Compute the posterior mean for theta.
theta_means = trace.posterior['theta'].mean(dim=['chain', 'draw']).values

# Create a DataFrame with one row per group.
theta_df = pd.DataFrame({
    'group': groups,
    'theta': theta_means
})
# Split the combined group string into separate product and year columns.
theta_df[[id_var, year_col]] = theta_df['group'].str.split('_', expand=True)
theta_df[year_col] = theta_df[year_col].astype(int)
print("\nEstimated latent traits (theta) per product-year:")
print(theta_df)

# -------------------------------
# 6. Plot a Slope Chart to Compare Years
# -------------------------------
# Pivot the data so that each product appears with estimated theta for each year.
pivot = theta_df.pivot(index=id_var, columns=year_col, values='theta').reset_index()

plt.figure(figsize=(8, 6))
for idx, row in pivot.iterrows():
    # Plot a line connecting 2023 and 2025 estimates.
    plt.plot([2023, 2025], [row[2023], row[2025]], marker='o')
    # Annotate near the points for clarity.
    plt.text(2023 - 0.5, row[2023], row[id_var], horizontalalignment='right')
    plt.text(2025 + 0.5, row[2025], row[id_var], horizontalalignment='left')

plt.xlabel('Year')
plt.ylabel('Estimated Latent Trait (theta)')
plt.title('Product Ranking Slope Chart: 2023 vs 2025')
plt.xticks([2023, 2025])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
