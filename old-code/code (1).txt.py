model {
  // Priors
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
  // Centering cutpoints around 0 for each question can help
  for (k in 1:K) {
      cutpoints[k] ~ normal(0, 3); // Fairly weak prior on cutpoint locations
                                   // Sensitivity Check: cutpoints[k] ~ normal(0, 1.5); // Tighter prior if scale is expected near 0
  }

  // Likelihood: Graded Response Model using ordered_logistic
  // Loop through observations
  for (n in 1:N) {
    int current_q = q_idx[n];       // Which question?
    int current_g = group_idx[n];   // Which group (team-year)?
    int current_dim = question_to_dimension[current_q]; // Which dimension does this question load on?

    // Calculate linear predictor (eta) for ordered logistic
    real eta = a[current_q] * theta[current_g, current_dim];

    // Calculate likelihood of the observed response y[n]
    y[n] ~ ordered_logistic(eta, cutpoints[current_q]);
  }
}