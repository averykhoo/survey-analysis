"""
run after scratch_15.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Assuming you've already run your Stan model and extracted samples:
#   samples = fit.extract()
# And you have your question_idx_map from Section 2:
#   question_idx_map = {question_name: stan_index (1-based), ...}

# 1. Compute posterior means for 'a' and 'cutpoints'
a_mean = samples['a'].mean(axis=0)              # shape: (K,)
cutpoints_mean = samples['cutpoints'].mean(axis=0)  # shape: (K, C-1)

K = a_mean.shape[0]
C_minus1 = cutpoints_mean.shape[1]
C = C_minus1 + 1

# 2. Invert the question_idx_map to get index → question name
inv_q_map = {idx: name for name, idx in question_idx_map.items()}

# 3. Define a grid of theta values
theta_grid = np.linspace(-4, 4, 200)

# 4. Loop over each question and plot the category probability curves
for q_idx in range(1, K + 1):
    a_q = a_mean[q_idx - 1]
    cuts_q = cutpoints_mean[q_idx - 1]  # length = C-1

    # Compute cumulative probabilities P(Y <= k | theta)
    # cumprob has shape (n_theta, C-1)
    eta = a_q * theta_grid[:, None]  # shape (n_theta, 1)
    cumprob = expit(cuts_q - eta)

    # Compute category probabilities
    p = np.zeros((len(theta_grid), C))
    p[:, 0] = cumprob[:, 0]
    for m in range(1, C - 1):
        p[:, m] = cumprob[:, m] - cumprob[:, m - 1]
    p[:, -1] = 1 - cumprob[:, -1]

    # Plot
    plt.figure(figsize=(8, 4))
    for m in range(C):
        plt.plot(theta_grid, p[:, m], label=f"Category {m + 1}")
    plt.title(f"Item Characteristic Curves: {inv_q_map[q_idx]}")
    plt.xlabel("Latent Ability (θ)")
    plt.ylabel("Probability")
    plt.legend(title="Response Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
