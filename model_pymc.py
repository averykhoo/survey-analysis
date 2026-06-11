# -*- coding: utf-8 -*-
"""
model_pymc.py

Defines the Hierarchical Multidimensional Graded Response Model (H-MGRM).
It uses a 2-level structure (Sections -> Categories) heavily anchored to
a Z-Score space to eliminate parameter identification collapse.
1-question categories are mathematically handled via a masking constraint.
"""

import os
from typing import Any
from typing import Dict
from typing import Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import config
import data_utils

try:
    import numpyro

    numpyro.set_host_device_count(4)
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False


def build_and_run_model(
        df_long: pd.DataFrame,
        struct_maps: Dict[str, Any]
) -> Tuple[az.InferenceData, pm.Model]:
    """
    Constructs and samples the hierarchical MGRM. Compiles GraphViz representation.

    Args:
        df_long: Parsed observation row table.
        struct_maps: Dictionary containing PyMC array vectors (Sections/Categories/Masks).

    Returns:
        ArviZ Inference Data object containing all traces, and the PyMC model context.
    """
    K = len(struct_maps["questions"])
    C_cat = len(struct_maps["categories"])
    S_sec = len(struct_maps["sections"])
    J = len(struct_maps["group_idx_map"])
    C_resp = config.N_CATEGORIES_RESPONSE

    obs_q_idx = df_long["question_idx"].values
    obs_g_idx = df_long["group_idx"].values
    question_to_category = struct_maps["question_to_category"]
    category_to_section = struct_maps["category_to_section"]
    is_multi_item_mask = struct_maps["is_multi_item_mask"]
    obs_cat_idx = question_to_category[obs_q_idx]

    smart_inits = data_utils.generate_smart_init(df_long, K, C_resp)

    with pm.Model() as dora_hmgrm:
        # ----------------------------------------------------
        # PRIORS & SENSITIVITY TESTING OPTIONS
        # ----------------------------------------------------

        # --- Parameter 1: Question Discrimination (a) ---
        # Default Prior: LogNormal ensures strict positivity.
        # Sensitivity Check: pm.LogNormal("a", mu=0.0, sigma=1.0) for wider variance.
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=K)

        # --- Parameter 2: Cutpoints ---
        # Default Prior: Initial boundary is normal, steps are positive exponentials.
        first_cut = pm.Normal("first_cut", mu=-1.5, sigma=2.0, shape=(K, 1), initval=smart_inits["first_cut"])
        cut_diffs = pm.Exponential("cut_diffs", lam=1.0, shape=(K, C_resp - 2), initval=smart_inits["cut_diffs"])
        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate([first_cut, first_cut + pt.cumsum(cut_diffs, axis=1)], axis=1)
        )

        # ----------------------------------------------------
        # LEVEL 1: Correlated Section Scores
        # ----------------------------------------------------
        # The latent trait z_sec represents abstract unscaled capability
        z_sec = pm.Normal("z_sec", mu=0, sigma=1, shape=(S_sec, J))

        # LKJ Cholesky Prior models the structural correlation between high-level Sections.
        # Sensitivity Check: pm.LKJCholeskyCov("chol_cov", n=S_sec, eta=1.0) for uniform distribution matrices.
        chol_cov, corr, stds = pm.LKJCholeskyCov(
            "chol_cov", n=S_sec, eta=2.0, sd_dist=pm.Exponential.dist(1.0)
        )

        # We enforce standardized z-score structure (Mean 0, Var 1) by dropping the varying stds.
        chol_corr = chol_cov / stds[:, None]

        # Final Section Scores: (J, S_sec)
        theta_sec = pm.Deterministic("theta_sec", pt.dot(chol_corr, z_sec).T)

        # ----------------------------------------------------
        # LEVEL 2: Nested Category Scores
        # ----------------------------------------------------
        # sigma_cat_raw specifies how much each Category can deviate from its parent Section.
        sigma_cat_raw = pm.HalfNormal("sigma_cat_raw", sigma=0.5, shape=C_cat)

        # ANCHORING MASK: If a category is single-question, its variance is clamped to 0.0
        sigma_cat = pm.Deterministic("sigma_cat", sigma_cat_raw * is_multi_item_mask)
        z_cat = pm.Normal("z_cat", mu=0, sigma=1, shape=(C_cat, J))

        # Calculate localized deviations for each team
        category_offset = pm.Deterministic("category_offset", (z_cat * sigma_cat[:, None]).T)

        # Final Category Scores: Broadcast Section scores to Category Map + Offsets
        theta_cat = pm.Deterministic(
            "theta_cat",
            theta_sec[:, category_to_section] + category_offset
        )

        # Downstream graphing aliases
        pm.Deterministic("Omega", corr)
        pm.Deterministic("mu", pt.zeros((S_sec, 1)))
        pm.Deterministic("sigma", pt.ones(S_sec))

        # ----------------------------------------------------
        # LIKELIHOOD
        # ----------------------------------------------------
        eta = a[obs_q_idx] * theta_cat[obs_g_idx, obs_cat_idx]

        y_obs = pm.OrderedLogistic(
            "y_obs",
            eta=eta,
            cutpoints=cutpoints[obs_q_idx],
            observed=df_long["response"].values - 1
        )

        # --- GraphViz DAG Export ---
        try:
            g = pm.model_to_graphviz(dora_hmgrm)
            g.render(filename=os.path.join(config.MODEL_DIR, 'model_dag'), format='png', cleanup=True)
            print("  Successfully saved Model DAG representation to model_dag.png")
        except Exception as e:
            print(f"  [SKIPPED] GraphViz DAG representation (Install graphviz system binaries): {e}")

        # --- Execute Sampler ---
        sampler_choice = "numpyro" if HAS_NUMPYRO else "nuts"
        print(f"  Executing H-MGRM compilation on {sampler_choice}...")

        idata = pm.sample(
            draws=config.ITER_SAMPLING,
            tune=config.ITER_WARMUP,
            chains=config.CHAINS,
            random_seed=config.RANDOM_SEED,
            target_accept=config.TARGET_ACCEPT,
            nuts_sampler=sampler_choice,
            compute_convergence_stat=True
        )

        print("Executing posterior predictive simulation checks...")
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=config.RANDOM_SEED)

    return idata, dora_hmgrm
