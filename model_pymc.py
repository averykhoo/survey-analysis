# -*- coding: utf-8 -*-
"""
model_pymc.py

Defines the Multidimensional Graded Response Model (MGRM) in PyMC,
incorporating standard default priors alongside clear comments for alternative
sensitivity checks. Uses a JAX (NumPyro) backend when available for parallelized sampling.
"""

from typing import Dict
from typing import Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import config
import data_utils

# Check for parallelization engine
try:
    import numpyro

    numpyro.set_host_device_count(4)
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False


def build_and_run_model(
        df_long: pd.DataFrame,
        question_idx_map: Dict[str, int],
        question_categories: Dict[str, int]
) -> Tuple[az.InferenceData, pm.Model]:
    """
    Constructs the model context and draws posterior samples using MCMC.
    """
    K = len(question_idx_map)
    L = len(np.unique(list(question_categories.values())))
    J = len(df_long["group_idx"].unique())
    C = config.N_CATEGORIES_RESPONSE

    # Derive mapping of questions to dimensions for model graph indexing
    q_idx_to_dim_idx = {q_idx: question_categories[q] for q, q_idx in question_idx_map.items()}
    question_to_dimension = np.array([q_idx_to_dim_idx[k] for k in range(K)])

    # Extract observed indices
    obs_q_idx = df_long["question_idx"].values
    obs_g_idx = df_long["group_idx"].values
    obs_dim_idx = question_to_dimension[obs_q_idx]

    # Generate smart starting points
    smart_inits = data_utils.generate_smart_init(df_long, K, C, L, J, config.RESPONSE_OPTIONS)

    with pm.Model() as dora_mgrm_model:
        # ----------------------------------------------------
        # PRIOR SENSITIVITY TESTING OPTIONS
        # ----------------------------------------------------

        # --- Parameter 1: Dimension Mean Prior (mu) ---
        # Default Prior (Standard normal):
        mu = pm.Normal("mu", mu=0, sigma=1, shape=(L, 1), initval=smart_inits["mu"])
        # Alternative Sensitivity Prior 1 (Tighter Regularization):
        # mu = pm.Normal("mu", mu=0, sigma=0.5, shape=(L, 1), initval=smart_inits["mu"])
        # Alternative Sensitivity Prior 2 (Wider / Weakly Informative):
        # mu = pm.Normal("mu", mu=0, sigma=2.0, shape=(L, 1), initval=smart_inits["mu"])
        # Alternative Sensitivity Prior 3 (Robust Student's t, accommodating outlier teams/dimensions):
        # mu = pm.StudentT("mu", nu=7, mu=0, sigma=1, shape=(L, 1), initval=smart_inits["mu"])

        # --- Parameter 2: Dimension Standard Deviation (sigma) ---
        # Default Prior (Weakly informative half-normal):
        sigma_val = pm.HalfNormal("sigma", sigma=1.0, shape=L, initval=smart_inits["sigma"])
        # Alternative Sensitivity Prior 1 (Heavier tailed Half-Cauchy):
        # sigma_val = pm.HalfCauchy("sigma", beta=2.5, shape=L, initval=smart_inits["sigma"])
        # Alternative Sensitivity Prior 2 (Exponential prior):
        # sigma_val = pm.Exponential("sigma", lam=1.0, shape=L, initval=smart_inits["sigma"])

        # --- Parameter 3: Cholesky Correlation Factor (L_Omega) ---
        # Default Prior (LKJ prior, weakly favoring smaller off-diagonal correlations):
        chol_cov, corr, _ = pm.LKJCholeskyCov(
            "chol_cov", n=L, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0)
        )
        # Alternative Sensitivity Prior 1 (Uniform prior over correlation matrices):
        # chol_cov, corr, _ = pm.LKJCholeskyCov("chol_cov", n=L, eta=1.0, sd_dist=pm.HalfNormal.dist(1.0))
        # Alternative Sensitivity Prior 2 (Strong push towards zero off-diagonal correlation):
        # chol_cov, corr, _ = pm.LKJCholeskyCov("chol_cov", n=L, eta=4.0, sd_dist=pm.HalfNormal.dist(1.0))

        # --- Parameter 4: Question Discrimination (a) ---
        # Default Prior (LogNormal, forcing discrimination to be strictly positive):
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=K, initval=smart_inits["a"])
        # Alternative Sensitivity Prior 1 (More variable LogNormal prior):
        # a = pm.LogNormal("a", mu=0.0, sigma=1.0, shape=K, initval=smart_inits["a"])
        # Alternative Sensitivity Prior 2 (Half-Normal prior):
        # a = pm.HalfNormal("a", sigma=1.0, shape=K, initval=smart_inits["a"])

        # --- Parameter 5: Cutpoints / Ordinal Thresholds ---
        # Default Prior (first cut Normal, offsets modeled as strictly positive Exponential steps):
        first_cut = pm.Normal("first_cut", mu=0.0, sigma=3.0, shape=(K, 1), initval=smart_inits["first_cut"])
        cut_diffs = pm.Exponential("cut_diffs", lam=1.0, shape=(K, C - 2), initval=smart_inits["cut_diffs"])
        # Alternative Sensitivity Prior 1 (Wider distances between thresholds, i.e., larger spacing):
        # cut_diffs = pm.Exponential("cut_diffs", lam=0.5, shape=(K, C - 2), initval=smart_inits["cut_diffs"])
        # Alternative Sensitivity Prior 2 (Tighter grouping/closer boundaries):
        # cut_diffs = pm.Exponential("cut_diffs", lam=2.0, shape=(K, C - 2), initval=smart_inits["cut_diffs"])

        # ----------------------------------------------------
        # LATENT TRAIT REPRESENTATION & LIKELIHOOD
        # ----------------------------------------------------
        z = pm.Normal("z", mu=0, sigma=1, shape=(L, J), initval=smart_inits["z"])
        theta = pm.Deterministic("theta", (mu + pt.dot(chol_cov, z)).T)
        pm.Deterministic("theta_z", (theta - mu.T) / sigma_val)

        # Reconstruct full correlation matrix & covariance
        pm.Deterministic("cov", pt.dot(chol_cov, chol_cov.T))
        pm.Deterministic("Omega", corr)

        # Concatenate threshold steps safely to represent ordered cutpoints
        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate([first_cut, first_cut + pt.cumsum(cut_diffs, axis=1)], axis=1)
        )

        # Standard Graded Response logit calculation
        eta = a[obs_q_idx] * theta[obs_g_idx, obs_dim_idx]

        # Likelihood
        y_obs = pm.OrderedLogistic(
            "y_obs",
            eta=eta,
            cutpoints=cutpoints[obs_q_idx],
            observed=df_long["response"].values - 1
        )

        # ----------------------------------------------------
        # SAMPLE posterior & posterior predictive
        # ----------------------------------------------------
        print(f"\nSampler configuration initiated with target_accept={config.TARGET_ACCEPT}.")
        sampler_choice = "numpyro" if HAS_NUMPYRO else "nuts"

        idata = pm.sample(
            draws=config.ITER_SAMPLING,
            tune=config.ITER_WARMUP,
            chains=config.CHAINS,
            random_seed=config.RANDOM_SEED,
            target_accept=config.TARGET_ACCEPT,
            init="adapt_diag",
            nuts_sampler=sampler_choice,
            compute_convergence_stat=True
        )

        print("Executing posterior predictive simulation checks...")
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=config.RANDOM_SEED)

    return idata, dora_mgrm_model
