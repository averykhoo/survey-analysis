# -*- coding: utf-8 -*-
"""
data_utils.py

Handles raw data loading, clean preprocessing, simulation of multi-year
DORA survey datasets containing reorganizations, and smart initialization.
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.special as sps

import config


def simulate_dora_data_reorg(
        sim_years_reorg: List[int],
        reorg_map_child_to_parents: Dict[str, List[str]],
        sim_n_resp_per_team_year: int,
        category_map_sim: Dict[str, List[str]],
        response_options_sim: List[int]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates a 3-year DORA survey dataset complete with team reorganizations,
    improving capabilities, and missing entries over years.
    """
    year1, year2, year3 = sim_years_reorg[0], sim_years_reorg[1], sim_years_reorg[2]

    year1_teams_sim = sorted(p for parents in reorg_map_child_to_parents.values() for p in parents)
    year2_teams_sim = sorted(p for parents in reorg_map_child_to_parents.values() for p in parents)
    year3_teams_sim = sorted(reorg_map_child_to_parents.keys())

    team_to_dept = {
        "Team_A": "Engineering", "Team_B": "Engineering", "Team_X": "Engineering",
        "Team_Y": "Engineering", "Team_C": "Product", "Team_D": "Product",
        "Team_Z": "Product", "Team_E": "Operations", "Team_F": "Operations", "Team_G": "Marketing"
    }

    rng = np.random.default_rng(config.RANDOM_SEED)
    n_cats_response_sim = len(response_options_sim)
    n_latent_sim = len(category_map_sim)
    questions_sim = [q for qs in category_map_sim.values() for q in qs]
    question_categories_sim = {q: idx for idx, (cat, qs) in enumerate(category_map_sim.items(), 1) for q in qs}

    base_corr = 0.4
    corr_matrix = np.ones((n_latent_sim, n_latent_sim)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = rng.uniform(-0.2, 0.2, (n_latent_sim, n_latent_sim))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    true_Omega = np.clip(corr_matrix + noise, 0.05, 0.95)
    np.fill_diagonal(true_Omega, 1.0)

    true_mu = rng.normal(0, 0.2, n_latent_sim)
    true_sigma = rng.lognormal(-0.5, 0.2, n_latent_sim)
    cov_matrix = np.diag(true_sigma) @ true_Omega @ np.diag(true_sigma)

    def generate_correlated_traits(mean_vector, cov_matrix):
        return rng.multivariate_normal(mean_vector, cov_matrix)

    true_latent_y1 = {}
    latent_for_resp = {}
    for team in year1_teams_sim:
        team_base_mean_offset = rng.normal(0, 0.4)
        mean_vector = true_mu + team_base_mean_offset
        traits_y1 = generate_correlated_traits(mean_vector, cov_matrix)
        true_latent_y1[team] = traits_y1
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year1, cat_idx)] = traits_y1[cat_idx - 1]

    true_latent_y2 = {}
    for team in year2_teams_sim:
        y1_traits = true_latent_y1.get(team, true_mu)
        improvements = rng.normal(0.2, 0.1, n_latent_sim)
        traits_y2 = y1_traits + improvements
        true_latent_y2[team] = traits_y2
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year2, cat_idx)] = traits_y2[cat_idx - 1]

    true_latent_y3 = {}
    for child_team in year3_teams_sim:
        parent_teams = reorg_map_child_to_parents.get(child_team, [])
        if not parent_teams:
            team_base_mean_offset = rng.normal(0, 0.4)
            mean_vector = true_mu + team_base_mean_offset
            traits_y3 = generate_correlated_traits(mean_vector, cov_matrix)
        else:
            parent_thetas = [true_latent_y2.get(p_team) for p_team in parent_teams if p_team in true_latent_y2]
            if not parent_thetas:
                team_base_mean_offset = rng.normal(0, 0.4)
                mean_vector = true_mu + team_base_mean_offset
                traits_y3 = generate_correlated_traits(mean_vector, cov_matrix)
            else:
                avg_parent_theta = np.mean(np.array(parent_thetas), axis=0)
                category_improvement_mean = 0.3
                improvements = rng.normal(category_improvement_mean, 0.2, n_latent_sim)
                traits_y3 = avg_parent_theta + improvements
        true_latent_y3[child_team] = traits_y3
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(child_team, year3, cat_idx)] = traits_y3[cat_idx - 1]

    true_latent_combined = {(team, year1): traits for team, traits in true_latent_y1.items()}
    true_latent_combined.update({(team, year2): traits for team, traits in true_latent_y2.items()})
    true_latent_combined.update({(team, year3): traits for team, traits in true_latent_y3.items()})

    true_a = {q: rng.lognormal(-0.1, 0.4) for q in questions_sim}
    true_cutpoints = {}
    for q in questions_sim:
        item_difficulty = rng.normal(0, 0.7)
        num_cutpoints = n_cats_response_sim - 1
        spacings = np.maximum(rng.lognormal(0, 0.3, num_cutpoints), 0.3)
        raw_cutpoints = np.cumsum(spacings)
        centered_cutpoints = raw_cutpoints - np.mean(raw_cutpoints) + item_difficulty
        true_cutpoints[q] = np.sort(centered_cutpoints)

    def simulate_response_gr(theta_val, a, cutpoints, response_options_sim):
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
        return rng.choice(response_options_sim, p=probs)

    data = []
    for year in sim_years_reorg:
        teams_in_year = year1_teams_sim if year == year1 else (year2_teams_sim if year == year2 else year3_teams_sim)
        for team in teams_in_year:
            if (team, year) not in true_latent_combined:
                continue
            for _ in range(sim_n_resp_per_team_year):
                row = {config.YEAR_COL: year, config.ID_VAR: team, "dept": team_to_dept.get(team, "Other")}
                for q in questions_sim:
                    cat = question_categories_sim[q]
                    theta_key = (team, year, cat)
                    if theta_key in latent_for_resp:
                        theta_true = latent_for_resp[theta_key]
                        a_val = true_a[q]
                        cp_val = true_cutpoints[q]
                        if year in [year1, year2] and q in ["q14", "q15", "q16"]:
                            row[q] = np.nan
                        elif rng.random() < 0.05:
                            row[q] = np.nan
                        else:
                            row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
                    else:
                        row[q] = np.nan
                data.append(row)

    df = pd.DataFrame(data)
    true_params = {
        "true_a":         true_a,
        "true_cutpoints": true_cutpoints,
        "true_Omega":     true_Omega,
        "true_latent":    true_latent_combined,
        "true_mu":        true_mu,
        "true_sigma":     true_sigma,
    }
    return df, true_params


def load_and_preprocess_data(df_raw: pd.DataFrame) -> Tuple[
    pd.DataFrame, Dict[str, List[str]], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    Performs preprocessing, filters missing items, builds 1-based indexing
    for PyMC compatibility, and evaluates completely dropped teams.
    """
    actual_years = sorted(df_raw[config.YEAR_COL].unique())
    actual_teams = df_raw[config.ID_VAR].unique()
    question_to_cat_initial = {q: cat for cat, qs in config.CATEGORY_MAPPING_INITIAL.items() for q in qs}
    actual_questions_in_data = df_raw.columns.intersection(question_to_cat_initial.keys())

    # Drop questions with 100% missing data in any year
    questions_to_use = []
    for q in actual_questions_in_data:
        if df_raw[q].notna().any():
            questions_to_use.append(q)
    questions_to_use = sorted(list(set(questions_to_use)))

    # Reconstruct category mappings based on active items
    category_mapping_final = {}
    question_categories_final = {}
    cat_idx_final = 0
    final_cat_names = []

    for cat_name_orig, qs_orig in config.CATEGORY_MAPPING_INITIAL.items():
        qs_in_cat_to_use = [q for q in qs_orig if q in questions_to_use]
        if qs_in_cat_to_use:
            category_mapping_final[cat_name_orig] = qs_in_cat_to_use
            final_cat_names.append(cat_name_orig)
            for q in qs_in_cat_to_use:
                question_categories_final[q] = cat_idx_final
            cat_idx_final += 1

    cat_idx_to_name_final = {idx: name for idx, name in enumerate(final_cat_names)}

    # Pivot to long format
    df_long = df_raw.melt(
        id_vars=[config.YEAR_COL, config.ID_VAR, "dept"],
        value_vars=questions_to_use,
        var_name="question",
        value_name="response"
    ).dropna(subset=["response"])

    df_long["response"] = pd.to_numeric(df_long["response"], errors="coerce").dropna().astype(int)
    df_long = df_long[df_long["response"].isin(config.RESPONSE_OPTIONS)]

    # Dropped Teams Verification Check
    teams_in_processed_data = df_long[config.ID_VAR].unique()
    teams_dropped_completely = sorted(list(set(actual_teams) - set(teams_in_processed_data)))
    if teams_dropped_completely:
        print(f"\n[WARNING] {len(teams_dropped_completely)} teams from raw data have NO valid responses "
              f"after filtering: {teams_dropped_completely}")

    # Index Mappings
    df_long["q_cat_final"] = df_long["question"].map(question_categories_final)
    df_long["group"] = df_long[config.ID_VAR].astype(str) + "|" + df_long[config.YEAR_COL].astype(str)

    groups_final = sorted(df_long["group"].unique())
    group_idx_map = {g: i for i, g in enumerate(groups_final)}
    df_long["group_idx"] = df_long["group"].map(group_idx_map)

    question_idx_map = {q: i for i, q in enumerate(questions_to_use)}
    df_long["question_idx"] = df_long["question"].map(question_idx_map)

    return df_long, category_mapping_final, cat_idx_to_name_final, question_idx_map, question_categories_final


def generate_smart_init(
        df_long: pd.DataFrame,
        K: int,
        C: int,
        L: int,
        J: int,
        response_options: List[int]
) -> Dict[str, np.ndarray]:
    """
    Computes data-informed empirical initial parameters for the model parameters
    to help prevent sampling bottlenecks or initialization failures.
    """
    initial_cutpoints = np.zeros((K, C - 1))
    probs = np.linspace(0, 1, C + 1)[1:-1]
    grouped_q = df_long.groupby("question_idx")["response"]

    for k in range(K):
        q_median = np.median(df_long["response"])
        if k in grouped_q.groups:
            responses_k = grouped_q.get_group(k)
            if len(responses_k) > 1:
                q_median = np.median(responses_k)

        q_quantiles = np.quantile(df_long["response"], probs)
        latent_approx = np.interp(q_quantiles, [min(response_options), max(response_options)], [-2.0, 2.0])
        q_cuts = latent_approx - np.interp(q_median, [min(response_options), max(response_options)], [-2.0, 2.0])
        q_cuts = np.sort(q_cuts)

        # Enforce minimal spacing constraint
        min_diff = 0.15
        for i in range(C - 2):
            q_cuts[i + 1] = max(q_cuts[i + 1], q_cuts[i] + min_diff)
        initial_cutpoints[k, :] = q_cuts

    first_cut = initial_cutpoints[:, 0:1]
    cut_diffs = np.diff(initial_cutpoints, axis=1)
    cut_diffs = np.maximum(cut_diffs, 0.05)  # Constrain strictly positive

    return {
        "z":         np.zeros((L, J)),
        "mu":        np.zeros((L, 1)),
        "sigma":     np.ones(L) * 0.8,
        "a":         np.ones(K),
        "first_cut": first_cut,
        "cut_diffs": cut_diffs
    }
