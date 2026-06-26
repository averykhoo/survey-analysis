# -*- coding: utf-8 -*-
"""
model_pymc.py

Defines the Hierarchical Multidimensional Graded Response Model (H-MGRM).
Implements a Nested Block-Cholesky structure across Sections and Categories,
using Student's t robust baselines, Variance-Anchor Normalization, and Gamma cutpoint gaps.
"""

import os
import warnings
from typing import Any
from typing import Dict
from typing import Tuple

import xarray
import pandas as pd
import pymc as pm
import pymc.sampling.jax as pm_jax
import pytensor.tensor as pt

import config
import data_utils

try:
    import numpyro

    HAS_NUMPYRO = True
    CHAIN_METHOD = 'parallel'

    try:
        import jax

        print('FOUND NUMPYRO + JAX')
        print(f'{jax.default_backend()=}')
        print(f'{jax.devices()=}')

        if jax.default_backend() == 'gpu':
            CHAIN_METHOD = 'vectorized'
            warnings.warn('unless you have 40+gb of vram i highly recommend using the cpu instead, '
                          'set `JAX_PLATFORM_NAME=cpu` before importing jax')

    except ImportError:
        print('FOUND NUMPYRO (CPU ONLY)')
        numpyro.set_host_device_count(48)  # many cores in prod server, decrease when testing

except ImportError:
    HAS_NUMPYRO = False


def build_and_run_model(
        df_long: pd.DataFrame,
        struct_maps: Dict[str, Any],
        run_sampling: bool = True,
) -> Tuple[xarray.DataTree, pm.Model]:
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

    # Establish reverse-mapping lists to construct block-Cholesky groups
    section_to_cats = {s: [] for s in range(S_sec)}
    for c, s in enumerate(category_to_section):
        section_to_cats[s].append(c)

    with pm.Model() as dora_hmgrm:
        # ----------------------------------------------------
        # PRIORS & SENSITIVITY TESTING OPTIONS
        # ----------------------------------------------------

        # --- Parameter 1: Question Discrimination (a) ---
        # Default Prior: LogNormal ensures strict positivity.
        # Sensitivity Check: pm.LogNormal("a", mu=0.0, sigma=1.0) for wider variance.
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=K)

        # --- Parameter 2: Robust Cutpoints ---
        # Gamma prior enforces minimum spacing, preventing threshold crossings during warmup
        first_cut = pm.Normal("first_cut", mu=-1.5, sigma=2.0, shape=(K, 1), initval=smart_inits["first_cut"])
        cut_diffs = pm.Gamma("cut_diffs", alpha=3.0, beta=4.0, shape=(K, C_resp - 2), initval=smart_inits["cut_diffs"])

        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate([first_cut, first_cut + pt.cumsum(cut_diffs, axis=1)], axis=1)
        )

        # ----------------------------------------------------
        # LEVEL 1: Section Scores (Correlated Robust baselines)
        # ----------------------------------------------------
        # Student's t baseline protects against over-shrinking highly extreme teams
        z_sec = pm.StudentT("z_sec", nu=config.STUDENT_T_NU, mu=0, sigma=1, shape=(S_sec, J))

        # Sample correlation Cholesky directly with high eta regularization.
        # LKJ Cholesky Prior models the structural correlation between high-level Sections.
        # Sensitivity Check: pm.LKJCholeskyCov("chol_cov", n=S_sec, eta=1.0) for uniform distribution matrices.
        # Sensitivity Check: pm.LKJCholeskyCov("chol_cov", n=S_sec, eta=2.0, sd_dist=pm.Exponential.dist(1.0))
        chol_cov_sec, corr_sec, std_sec = pm.LKJCholeskyCov(
            "chol_cov_sec", n=S_sec, eta=config.LKJ_ETA, sd_dist=pm.HalfNormal.dist(0.5)
        )

        # We enforce standardized z-score structure (Mean 0, Var 1) by dropping the varying stds.
        chol_corr_sec = chol_cov_sec / std_sec[:, None]

        # Final Section Scores: (J, S_sec)
        theta_sec = pm.Deterministic("theta_sec", pt.dot(chol_corr_sec, z_sec).T)

        # ----------------------------------------------------
        # LEVEL 2: Block-Cholesky Nested Category Scores
        # ----------------------------------------------------
        category_offsets = [None] * C_cat
        sigma_cat_raw = pm.HalfNormal("sigma_cat_raw", sigma=config.SIGMA_CAT_PRIOR_SIGMA, shape=C_cat)

        # Clamp 1-question category variances strictly to 0
        sigma_cat = pm.Deterministic("sigma_cat", sigma_cat_raw * is_multi_item_mask)
        z_cat_indep = pm.StudentT("z_cat_indep", nu=config.STUDENT_T_NU, mu=0, sigma=1, shape=(C_cat, J))

        for s_idx in range(S_sec):
            cats_in_s = section_to_cats[s_idx]
            multi_item_cats_in_s = [c for c in cats_in_s if is_multi_item_mask[c] == 1.0]

            if len(multi_item_cats_in_s) > 1:
                n_multi = len(multi_item_cats_in_s)
                # Sample block Cholesky for offsets within this section
                z_block = pm.StudentT(f"z_block_{s_idx}", nu=config.STUDENT_T_NU, mu=0, sigma=1, shape=(n_multi, J))

                # Check for 2x2 blocks to bypass the PyTensor rewrite Scan bug
                if n_multi == 2:
                    # rho is the correlation coefficient between the two categories
                    # Restricting to [-0.99, 0.99] ensures numerical stability in sqrt(1 - rho^2)
                    rho = pm.Uniform(f"rho_block_{s_idx}", lower=-0.99, upper=0.99)

                    # Manually construct 2x2 Cholesky correlation matrix
                    row1 = pt.stack([1.0, 0.0])
                    row2 = pt.stack([rho, pt.sqrt(1.0 - pt.square(rho))])
                    chol_corr_block = pt.stack([row1, row2])
                else:
                    # Fallback general block-Cholesky if you change the hierarchy in the future
                    chol_cov_block, corr_block, std_block = pm.LKJCholeskyCov(
                        f"chol_cov_block_{s_idx}", n=n_multi, eta=config.LKJ_ETA, sd_dist=pm.HalfNormal.dist(0.5)
                    )
                    chol_corr_block = chol_cov_block / std_block[:, None]

                rotated_block = pt.dot(chol_corr_block, z_block).T  # (J, n_multi)

                for idx, c in enumerate(multi_item_cats_in_s):
                    category_offsets[c] = rotated_block[:, idx] * sigma_cat[c]

                # Populate single-item categories in this section that were skipped
                for c in cats_in_s:
                    if is_multi_item_mask[c] == 0.0:
                        category_offsets[c] = z_cat_indep[c, :] * sigma_cat[c]
            else:
                for c in cats_in_s:
                    category_offsets[c] = z_cat_indep[c, :] * sigma_cat[c]

        # Convert offset tensor list into a unified symbolic matrix of shape (J, C_cat)
        category_offset_matrix = pt.stack(category_offsets, axis=1)

        # ----------------------------------------------------
        # VARIANCE-ANCHOR NORMALIZATION
        # ----------------------------------------------------
        # Enforces a strict, standard-normal scale (Var = 1.0) to break the scale-indeterminacy loop
        theta_cat_unnormalized = theta_sec[:, category_to_section] + category_offset_matrix
        normalization_factor = pt.sqrt(1.0 + pt.square(sigma_cat))
        theta_cat = pm.Deterministic(
            "theta_cat",
            theta_cat_unnormalized / normalization_factor[None, :]
        )

        # Verification exports
        pm.Deterministic("Omega", corr_sec)
        pm.Deterministic("mu", pt.zeros((S_sec, 1)))
        pm.Deterministic("sigma", pt.ones(S_sec))

        # ----------------------------------------------------
        # LIKELIHOOD
        # ----------------------------------------------------
        eta = a[obs_q_idx] * theta_cat[obs_g_idx, obs_cat_idx]

        pm.OrderedLogistic(
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

        # --- Execute Sampler Block ---
        if run_sampling:
            sampler_choice = "numpyro" if HAS_NUMPYRO else "nuts"
            print(f"  Executing H-MGRM compilation on {sampler_choice} (Warmup: {config.ITER_WARMUP})...")

            idata = pm_jax.sample_numpyro_nuts(  # pm.sample(
                draws=config.ITER_SAMPLING,
                tune=config.ITER_WARMUP,
                chains=config.CHAINS,
                chain_method=CHAIN_METHOD,
                random_seed=config.RANDOM_SEED,
                target_accept=config.TARGET_ACCEPT,
                nuts_sampler=sampler_choice,
                compute_convergence_checks=True,
                postprocessing_backend='cpu',  # move to cpu immediately to avoid allocating more vram
                postprocessing_chunks=4,  # reduce peak gpu memory
                idata_kwargs={'log_likelihood': True},  # needed for looic diagnostics (optional, defaults to true)
            )

            print("Executing posterior predictive simulation checks...")
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=config.RANDOM_SEED)

            return idata, dora_hmgrm

    return None, dora_hmgrm
