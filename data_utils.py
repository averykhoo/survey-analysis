# -*- coding: utf-8 -*-
"""
data_utils.py

Handles hierarchical data simulation and preprocessing, building index mappings
and structural masks for the hierarchical IRT model. Parses multiple years of lineages.
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
        lineage_map: Dict[int, Dict[str, List[str]]],
        sim_n_resp_per_team_year: int,
        hierarchy_map: Dict[str, Dict[str, List[str]]],
        response_options_sim: List[int]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates a DORA dataset using a true two-level Section -> Category hierarchy,
    respecting multi-year reorganization lineages.

    Args:
        sim_years_reorg: Ordered list of years to simulate (e.g. [2023, 2025, 2026]).
        lineage_map: Year-by-year reorg transition map.
        sim_n_resp_per_team_year: Responders to simulate per team.
        hierarchy_map: Section/Category/Question mapping dictionary.
        response_options_sim: Likert response integer options.

    Returns:
        Dataframe of simulated responses and Dictionary of true generative parameters.
    """
    print("--- SIMULATING SYNTHETIC DORA DATA WITH REORGANIZATIONS ---")
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Determine teams active per year based on the lineage chaining
    teams_per_year = {}
    for i, yr in enumerate(sim_years_reorg):
        if yr in lineage_map:
            teams_per_year[yr] = list(lineage_map[yr].keys())
        else:
            # First year derives from the parent values of the subsequent year
            if i + 1 < len(sim_years_reorg) and sim_years_reorg[i + 1] in lineage_map:
                next_yr = sim_years_reorg[i + 1]
                parents = {p for plist in lineage_map[next_yr].values() for p in plist}
                teams_per_year[yr] = sorted(list(parents))
            else:
                teams_per_year[yr] = ["Team_A", "Team_B"]  # Fallback

    team_to_dept = {
        "Team_A": "Engineering", "Team_B": "Engineering", "Team_X": "Engineering",
        "Team_Y": "Engineering", "Team_C": "Product", "Team_D": "Product",
        "Team_Z": "Product", "Team_E": "Operations", "Team_F": "Operations", "Team_G": "Marketing"
    }

    # Flatten hierarchy
    sections = list(hierarchy_map.keys())
    categories = []
    cat_to_sec = {}
    questions = []
    cat_questions = {}

    for sec, cats in hierarchy_map.items():
        for cat, qs in cats.items():
            categories.append(cat)
            cat_to_sec[cat] = sec
            questions.extend(qs)
            cat_questions[cat] = qs

    S = len(sections)
    C = len(categories)

    # 1. Simulate true section correlation matrix
    base_corr = 0.4
    corr_matrix = np.ones((S, S)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = rng.uniform(-0.1, 0.1, (S, S))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    true_Omega_sec = np.clip(corr_matrix + noise, 0.1, 0.9)
    np.fill_diagonal(true_Omega_sec, 1.0)

    # 2. Simulate latent section values mapped cleanly across lineages
    true_latent_sec = {}

    def generate_correlated_sec_traits():
        return rng.multivariate_normal(np.zeros(S), true_Omega_sec)

    for i, yr in enumerate(sim_years_reorg):
        for team in teams_per_year[yr]:
            if i == 0 or yr not in lineage_map or not lineage_map[yr].get(team):
                # New origin team
                true_latent_sec[(team, yr)] = generate_correlated_sec_traits()
            else:
                # Evolving team: Inherit mean of parents + improvement
                parents = lineage_map[yr][team]
                prev_yr = sim_years_reorg[i - 1]
                parent_thetas = [true_latent_sec[(p, prev_yr)] for p in parents if (p, prev_yr) in true_latent_sec]

                if parent_thetas:
                    avg_parent = np.mean(np.array(parent_thetas), axis=0)
                    true_latent_sec[(team, yr)] = avg_parent + rng.normal(0.2, 0.1, S)
                else:
                    true_latent_sec[(team, yr)] = generate_correlated_sec_traits()

    # 3. Simulate Category Offsets (independent relative to parent Section)
    true_latent_cat = {}
    for (team, yr), sec_traits in true_latent_sec.items():
        cat_traits = np.zeros(C)
        for c_idx, cat in enumerate(categories):
            s_idx = sections.index(cat_to_sec[cat])
            sec_base = sec_traits[s_idx]

            # If multi-question, add category deviation. If 1-question, force deviation to 0
            deviation = rng.normal(0, 0.3) if len(cat_questions[cat]) > 1 else 0.0
            cat_traits[c_idx] = sec_base + deviation
        true_latent_cat[(team, yr)] = cat_traits

    # 4. Simulate Question IRT Parameters
    true_a = {q: rng.lognormal(-0.1, 0.4) for q in questions}
    true_cutpoints = {}
    for q in questions:
        item_difficulty = rng.normal(0, 0.7)
        spacings = np.maximum(rng.lognormal(0, 0.3, len(response_options_sim) - 1), 0.3)
        raw_cutpoints = np.cumsum(spacings)
        centered_cutpoints = raw_cutpoints - np.mean(raw_cutpoints) + item_difficulty
        true_cutpoints[q] = np.sort(centered_cutpoints)

    def simulate_response_gr(theta_val, a, cutpoints):
        eta = a * theta_val
        cum_prob_le = sps.expit(cutpoints - eta)
        probs = np.zeros(len(response_options_sim))
        probs[0] = cum_prob_le[0]
        for k in range(1, len(response_options_sim) - 1):
            probs[k] = cum_prob_le[k] - cum_prob_le[k - 1]
        probs[-1] = 1.0 - cum_prob_le[-2]
        probs = np.maximum(probs, 1e-9)
        probs /= probs.sum()
        return rng.choice(response_options_sim, p=probs)

    data = []
    for yr in sim_years_reorg:
        for team in teams_per_year[yr]:
            for _ in range(sim_n_resp_per_team_year):
                row = {config.YEAR_COL: yr, config.ID_VAR: team, "dept": team_to_dept.get(team, "Other")}
                for q in questions:
                    parent_cat = [c for c, qs in cat_questions.items() if q in qs][0]
                    c_idx = categories.index(parent_cat)

                    theta_true = true_latent_cat[(team, yr)][c_idx]

                    if rng.random() < 0.05:
                        row[q] = np.nan
                    else:
                        row[q] = simulate_response_gr(theta_true, true_a[q], true_cutpoints[q])
                data.append(row)

    df = pd.DataFrame(data)
    true_params = {"true_a": true_a, "true_cutpoints": true_cutpoints, "true_Omega_sec": true_Omega_sec}
    print(f"  Successfully simulated {len(df)} rows across {sim_years_reorg} with reorg structures.")
    return df, true_params


def load_and_preprocess_data(df_raw: pd.DataFrame, hierarchy_map: Dict[str, Dict[str, List[str]]]) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    """
    Parses structural mappings, generates indices and mapping tensors for PyMC constraints.

    Args:
        df_raw: Flat raw responses dataframe.
        hierarchy_map: Mapping dictionary.

    Returns:
        Processed long-format DataFrame and structural tensor dictionaries for the model.
    """
    print("--- PARSING STRUCTURAL LAYERS & INDEX MAPPINGS ---")

    # Strict Type Coercion to prevent dictionary lineage lookup failures
    df_raw[config.YEAR_COL] = df_raw[config.YEAR_COL].astype(int)
    df_raw[config.ID_VAR] = df_raw[config.ID_VAR].astype(str).str.strip()

    sections_list = list(hierarchy_map.keys())
    categories_list = []
    questions_list = []

    cat_to_sec_idx = []
    is_multi_item_list = []
    question_to_cat_idx = {}

    for s_idx, sec in enumerate(sections_list):
        for cat, qs in hierarchy_map[sec].items():
            categories_list.append(cat)
            cat_to_sec_idx.append(s_idx)
            is_multi_item_list.append(1.0 if len(qs) > 1 else 0.0)

            for q in qs:
                questions_list.append(q)
                question_to_cat_idx[q] = len(categories_list) - 1

    df_long = df_raw.melt(
        id_vars=[config.YEAR_COL, config.ID_VAR, "dept"],
        value_vars=questions_list,
        var_name="question",
        value_name="response"
    ).dropna(subset=["response"])

    df_long["response"] = pd.to_numeric(df_long["response"]).astype(int)

    df_long["group"] = df_long[config.ID_VAR].astype(str) + "|" + df_long[config.YEAR_COL].astype(str)
    groups_list = sorted(df_long["group"].unique())
    group_idx_map = {g: i for i, g in enumerate(groups_list)}
    df_long["group_idx"] = df_long["group"].map(group_idx_map)

    question_idx_map = {q: i for i, q in enumerate(questions_list)}
    df_long["question_idx"] = df_long["question"].map(question_idx_map)

    question_to_category = np.array([question_to_cat_idx[q] for q in questions_list])
    category_to_section = np.array(cat_to_sec_idx)
    is_multi_item_mask = np.array(is_multi_item_list)

    struct_maps = {
        "sections":             sections_list,
        "categories":           categories_list,
        "questions":            questions_list,
        "question_to_category": question_to_category,
        "category_to_section":  category_to_section,
        "is_multi_item_mask":   is_multi_item_mask,
        "group_idx_map":        group_idx_map,
        "question_idx_map":     question_idx_map
    }

    print(f"  Preprocessed counts: {len(questions_list)} questions, {len(categories_list)} categories, "
          f"{len(sections_list)} sections across {len(groups_list)} team-year groups.")
    return df_long, struct_maps


def generate_smart_init(df_long: pd.DataFrame, K: int, C_resp: int) -> Dict[str, np.ndarray]:
    """
    Computes data-informed starting points for ordinal cutpoints to prevent warm-up crashes.
    """
    initial_cutpoints = np.zeros((K, C_resp - 1))
    probs = np.linspace(0, 1, C_resp + 1)[1:-1]

    for k in range(K):
        q_quantiles = np.quantile(df_long["response"], probs)
        latent_approx = np.interp(q_quantiles, [1, 6], [-1.5, 1.5])
        initial_cutpoints[k, :] = np.sort(latent_approx)

    first_cut = initial_cutpoints[:, 0:1]
    cut_diffs = np.diff(initial_cutpoints, axis=1)
    cut_diffs = np.maximum(cut_diffs, 0.05)

    return {
        "first_cut": first_cut,
        "cut_diffs": cut_diffs
    }
