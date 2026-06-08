# -*- coding: utf-8 -*-
"""
Comprehensive MGRM Analysis Script for DORA Survey Data (PyMC Version)

Purpose: Analyze Likert-scale DORA survey responses using a Bayesian
         Multidimensional Graded Response Model (MGRM) with correlated latent traits.
         Estimates latent capabilities (theta) for engineering teams over time,
         handles team reorganizations in visualization (using standardized scores),
         assesses improvement, estimates correlations between capabilities,
         plots item-level diagnostics (ICC/KDE, CRF, IIF), Test Information (TIF)
         and performs rigorous diagnostic checks.
"""

import os
import re
import warnings

import arviz as az
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.special as sps
import seaborn as sns
from adjustText import adjust_text

# -----------------------------------------
# Global Constants & MCMC Settings
# -----------------------------------------
ITER_WARMUP = 1000
ITER_SAMPLING = 1500
CHAINS = 4
RANDOM_SEED = 42

# --- Diagnostic Thresholds ---
RHAT_THRESHOLD = 1.05
NEFF_RATIO_THRESHOLD = 0.1
PARETO_K_THRESHOLD = 0.7
HIGH_CORR_THRESHOLD = 0.85
LOW_DISCRIMINATION_THRESHOLD = 0.3
MIN_RESPONSES_PER_GROUP = 5

# --- File Paths & Schema ---
OUTPUT_DIR = 'dora_analysis_output_pymc'
YEAR_COL = 'year'
ID_VAR = 'team_id'

# --- Response & Category Mapping ---
RESPONSE_OPTIONS = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE = len(RESPONSE_OPTIONS)

CATEGORY_MAPPING_INITIAL = {
    'Deployment & Release':       ['q1', 'q2', 'q3'],
    'Monitoring & Observability': ['q4', 'q5'],
    'Technical Practices':        ['q6', 'q7', 'q8'],
    'Team Collaboration':         ['q9', 'q10'],
    'Process Efficiency':         ['q11'],
    'Learning & Development':     ['q12', 'q13'],
    'System Reliability':         ['q14', 'q15'],
    'Change Management':          ['q16']
}

REORG_MAPPING_Y_LAST_TO_Y1 = {
    'Team_X': ['Team_A', 'Team_B'],
    'Team_Y': ['Team_B', 'Team_C'],
    'Team_Z': ['Team_A', 'Team_C', 'Team_D'],
    'Team_E': ['Team_E'],
    'Team_F': ['Team_F'],
    'Team_G': []
}

# --- Derive Global Mappings ---
QUESTION_TO_CAT_NAME_INITIAL = {q: cat_name for cat_name, qs in CATEGORY_MAPPING_INITIAL.items() for q in qs}

PARENT_TO_CHILD_MAPPING = {}
all_parent_teams_in_map = set(p for parents in REORG_MAPPING_Y_LAST_TO_Y1.values() for p in parents)
all_child_teams_in_map = set(REORG_MAPPING_Y_LAST_TO_Y1.keys())

for parent in all_parent_teams_in_map:
    PARENT_TO_CHILD_MAPPING[parent] = [child for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items() if
                                       parent in parents]

for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items():
    if len(parents) == 1 and parents[0] == child and child not in PARENT_TO_CHILD_MAPPING:
        PARENT_TO_CHILD_MAPPING[child] = [child]


# -----------------------------------------
# Helper Functions
# -----------------------------------------

def simulate_dora_data_reorg(sim_years_reorg, reorg_map_child_to_parents, sim_n_resp_per_team_year, category_map_sim,
                             response_options_sim):
    """
    Generates simulated DORA survey data reflecting transitions over 3 years.
    """
    print("--- Section 1: Simulating Sample DORA Data with Reorg ---")
    year1, year2, year3 = sim_years_reorg[0], sim_years_reorg[1], sim_years_reorg[2]

    year1_teams_sim = sorted(list(set(p for parents in reorg_map_child_to_parents.values() for p in parents)))
    year2_teams_sim = sorted(list(set(p for parents in reorg_map_child_to_parents.values() for p in parents)))
    year3_teams_sim = sorted(list(reorg_map_child_to_parents.keys()))

    np.random.seed(RANDOM_SEED)
    n_cats_response_sim = len(response_options_sim)
    n_latent_sim = len(category_map_sim)
    questions_sim = [q for qs in category_map_sim.values() for q in qs]
    question_categories_sim = {q: idx for idx, (cat, qs) in enumerate(category_map_sim.items(), 1) for q in qs}

    base_corr = 0.4
    corr_matrix = np.ones((n_latent_sim, n_latent_sim)) * base_corr
    np.fill_diagonal(corr_matrix, 1.0)
    noise = np.random.uniform(-0.2, 0.2, (n_latent_sim, n_latent_sim))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    true_Omega = np.clip(corr_matrix + noise, 0.05, 0.95)
    np.fill_diagonal(true_Omega, 1.0)

    true_mu = np.random.normal(0, 0.2, n_latent_sim)
    true_sigma = np.random.lognormal(-0.5, 0.2, n_latent_sim)
    cov_matrix = np.diag(true_sigma) @ true_Omega @ np.diag(true_sigma)

    def generate_correlated_traits(mean_vector, cov_matrix):
        return np.random.multivariate_normal(mean_vector, cov_matrix)

    true_latent_y1 = {}
    latent_for_resp = {}
    for team in year1_teams_sim:
        team_base_mean_offset = np.random.normal(0, 0.4)
        mean_vector = true_mu + team_base_mean_offset
        traits_y1 = generate_correlated_traits(mean_vector, cov_matrix)
        true_latent_y1[team] = traits_y1
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year1, cat_idx)] = traits_y1[cat_idx - 1]

    true_latent_y2 = {}
    for team in year2_teams_sim:
        y1_traits = true_latent_y1.get(team, true_mu)
        improvements = np.random.normal(0.2, 0.1, n_latent_sim)
        traits_y2 = y1_traits + improvements
        true_latent_y2[team] = traits_y2
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(team, year2, cat_idx)] = traits_y2[cat_idx - 1]

    true_latent_y3 = {}
    for child_team in year3_teams_sim:
        parent_teams = reorg_map_child_to_parents.get(child_team, [])
        if not parent_teams:
            team_base_mean_offset = np.random.normal(0, 0.4)
            mean_vector = true_mu + team_base_mean_offset
            traits_y3 = generate_correlated_traits(mean_vector, cov_matrix)
        else:
            parent_thetas = [true_latent_y2.get(p_team) for p_team in parent_teams if p_team in true_latent_y2]
            if not parent_thetas:
                team_base_mean_offset = np.random.normal(0, 0.4)
                mean_vector = true_mu + team_base_mean_offset
                traits_y3 = generate_correlated_traits(mean_vector, cov_matrix)
            else:
                avg_parent_theta = np.mean(np.array(parent_thetas), axis=0)
                category_improvement_mean = 0.3
                improvements = np.random.normal(category_improvement_mean, 0.2, n_latent_sim)
                traits_y3 = avg_parent_theta + improvements
        true_latent_y3[child_team] = traits_y3
        for cat_idx in range(1, n_latent_sim + 1):
            latent_for_resp[(child_team, year3, cat_idx)] = traits_y3[cat_idx - 1]

    true_latent_combined = {(team, year1): traits for team, traits in true_latent_y1.items()}
    true_latent_combined.update({(team, year2): traits for team, traits in true_latent_y2.items()})
    true_latent_combined.update({(team, year3): traits for team, traits in true_latent_y3.items()})

    true_a = {q: np.random.lognormal(-0.1, 0.4) for q in questions_sim}
    true_cutpoints = {}
    for q in questions_sim:
        item_difficulty = np.random.normal(0, 0.7)
        num_cutpoints = n_cats_response_sim - 1
        spacings = np.maximum(np.random.lognormal(0, 0.3, num_cutpoints), 0.3)
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
        return np.random.choice(response_options_sim, p=probs)

    data = []
    for year in sim_years_reorg:
        if year == year1:
            teams_in_year = year1_teams_sim
        elif year == year2:
            teams_in_year = year2_teams_sim
        else:
            teams_in_year = year3_teams_sim

        for team in teams_in_year:
            if (team, year) not in true_latent_combined:
                continue

            for i in range(sim_n_resp_per_team_year):
                row = {YEAR_COL: year, ID_VAR: team}
                for q in questions_sim:
                    cat = question_categories_sim[q]
                    theta_key = (team, year, cat)
                    if theta_key in latent_for_resp:
                        theta_true = latent_for_resp[theta_key]
                        a_val = true_a[q]
                        cp_val = true_cutpoints[q]

                        if year in [year1, year2] and q in ['q14', 'q15', 'q16']:
                            row[q] = np.nan
                        elif np.random.rand() < 0.05:
                            row[q] = np.nan
                        else:
                            row[q] = simulate_response_gr(theta_true, a_val, cp_val, response_options_sim)
                    else:
                        row[q] = np.nan
                data.append(row)

    df = pd.DataFrame(data)
    true_params = {'true_a':      true_a, 'true_cutpoints': true_cutpoints, 'true_Omega': true_Omega,
                   'true_latent': true_latent_combined, 'true_mu': true_mu, 'true_sigma': true_sigma}
    return df, true_params


def filename_escape(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def plot_reorg_slope_chart_standardized_internal(theta_df_viz, category_name_viz, id_var_viz, year_col_viz, years_viz,
                                                 mu_map_viz, sigma_map_viz, parent_to_child_map, output_dir_viz):
    pop_mean = mu_map_viz.get(category_name_viz)
    pop_sd = sigma_map_viz.get(category_name_viz)

    if pop_mean is None or pop_sd is None or pop_sd <= 1e-6:
        return

    raw_col_name = f"theta_{category_name_viz}"
    if raw_col_name not in theta_df_viz.columns:
        return

    year1, year2 = min(years_viz), max(years_viz)
    df_y1 = theta_df_viz[theta_df_viz[year_col_viz] == year1].set_index(id_var_viz)
    df_y2 = theta_df_viz[theta_df_viz[year_col_viz] == year2].set_index(id_var_viz)

    fig, ax = plt.subplots(figsize=(12, max(8, (len(df_y1) + len(df_y2)) * 0.4)))
    texts = []
    plotted_points_y1 = {}
    plotted_points_y2 = {}

    for team_id, row in df_y1.iterrows():
        raw_val = row[raw_col_name]
        if pd.notna(raw_val):
            y1_val_z = (raw_val - pop_mean) / pop_sd
            ax.plot(year1, y1_val_z, 'o', color='black', markersize=5)
            texts.append(ax.text(year1 - 0.05, y1_val_z, team_id, ha='right', va='center', fontsize=8))
            plotted_points_y1[team_id] = y1_val_z

    all_year2_teams = sorted(df_y2.index.unique().tolist())
    cmap_func = cm.get_cmap('tab20', max(1, len(all_year2_teams)))
    color_map_y2 = {team: cmap_func(i % 20) for i, team in enumerate(all_year2_teams)}

    for team_id, row in df_y2.iterrows():
        raw_val = row[raw_col_name]
        if pd.notna(raw_val):
            y2_val_z = (raw_val - pop_mean) / pop_sd
            point_color = color_map_y2.get(team_id, 'gray')
            ax.plot(year2, y2_val_z, 'o', color=point_color, markersize=6)
            texts.append(ax.text(year2 + 0.05, y2_val_z, team_id, ha='left', va='center', fontsize=8))
            plotted_points_y2[team_id] = y2_val_z

    for parent, children in parent_to_child_map.items():
        if parent in plotted_points_y1:
            y1_val_z = plotted_points_y1[parent]
            for child in children:
                if child in plotted_points_y2:
                    y2_val_z = plotted_points_y2[child]
                    line_color = color_map_y2.get(child, 'gray')
                    ax.plot([year1, year2], [y1_val_z, y2_val_z], linestyle='-', lw=2.5, color=line_color, alpha=0.6)

    adjust_text(texts, ax=ax, force_points=(0.2, 0.3), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.set_title(f'Standardized Capability Evolution: {category_name_viz} ({year1} vs {year2})', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'Capability Z-Score (Std. Devs from Mean)', fontsize=12)
    ax.set_xticks([year1, year2])
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.0, alpha=0.8)
    ax.set_xlim(year1 - 0.4, year2 + 0.4)

    filepath = os.path.join(output_dir_viz, f'std_reorg_slope_{filename_escape(category_name_viz)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


def plot_omega_clustermap(corr_df_viz, output_dir_viz):
    if corr_df_viz is None or corr_df_viz.empty:
        return
    n_cats = corr_df_viz.shape[0]
    figsize = (max(8, n_cats * 0.9), max(7, n_cats * 0.8))
    try:
        cluster_grid = sns.clustermap(
            corr_df_viz, method='average', metric='euclidean', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, annot=(n_cats <= 12), fmt=".2f",
            linewidths=.5, linecolor='lightgray', figsize=figsize
        )
        cluster_grid.fig.suptitle('Capability Correlations (Mean Omega)', y=1.02)
        filepath = os.path.join(output_dir_viz, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"WARNING: Clustermap generation failed: {e}")


def plot_likert_response_distributions(df_raw, theta_df, category_mapping, response_options, id_var, year_col,
                                       target_year, output_dir):
    df_overall_year = theta_df[theta_df[year_col] == target_year]
    df_raw_year = df_raw[df_raw[year_col] == target_year]

    if df_overall_year.empty or df_raw_year.empty:
        return

    cmap = plt.cm.get_cmap('vlag', len(response_options))
    colors = [cmap(i) for i in range(len(response_options))]

    for category, questions in category_mapping.items():
        df_long_cat = df_raw_year.melt(id_vars=[id_var], value_vars=questions, var_name='question',
                                       value_name='response').dropna()
        if df_long_cat.empty:
            continue

        df_long_cat['response'] = pd.to_numeric(df_long_cat['response'], errors='coerce')
        df_long_cat = df_long_cat[df_long_cat['response'].isin(response_options)]
        df_long_cat['response'] = df_long_cat['response'].astype(int)

        counts = df_long_cat.groupby([id_var, 'response']).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        theta_col = f"theta_{category}"
        if theta_col in df_overall_year.columns:
            ordered_teams = df_overall_year.sort_values(by=theta_col, ascending=True)[id_var].tolist()
        else:
            ordered_teams = percentages.index.tolist()

        final_order = [t for t in ordered_teams if t in percentages.index]
        plot_df = percentages.reindex(index=final_order, columns=response_options[::-1], fill_value=0)

        fig_height = max(5, len(plot_df) * 0.6)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        plot_df.plot(kind='barh', stacked=True, color=colors[::-1], edgecolor='white', linewidth=0.5, ax=ax)

        ax.set_title(f"{category} ({target_year})\nTeam Response Distribution (Sorted by Rank)\n", fontsize=13,
                     weight='bold')
        ax.set_xlabel('Percentage of Responses', fontsize=11)
        ax.set_ylabel('Team (Capability Rank - Bottom to Top)', fontsize=11)
        ax.set_xlim([0, 100])
        ax.legend(response_options[::-1], title='Response', loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=len(response_options), frameon=False)

        for idx, (team, row) in enumerate(plot_df.iterrows()):
            start_pos = 0
            for j, val in enumerate(row):
                if val > 3.5:
                    text_color = 'white' if j in (0, len(response_options) - 1) else 'black'
                    ax.text(start_pos + val / 2, idx, f'{val:.0f}%', va='center', ha='center', color=text_color,
                            fontsize=9)
                start_pos += val

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'likert_{filename_escape(category)}.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)


def run_diagnostic_checks(idata, df_long, cat_idx_to_name_map, question_idx_map):
    print("\n--- RUNNING DIAGNOSTIC CHECKS ---")
    warnings_found = []

    # Divergences
    divergences = idata.sample_stats.diverging.sum().item()
    if divergences > 0:
        msg = f"CRITICAL WARNING: {divergences} divergent transitions found! Review parametrization."
        warnings_found.append(msg)
        print(msg)
    else:
        print("  Divergences: OK (0 found)")

    summary = az.summary(idata, var_names=["mu", "sigma", "a"], round_to=3)

    max_rhat = summary['r_hat'].max()
    print(f"  Max R-hat observed: {max_rhat:.3f}")
    if max_rhat > RHAT_THRESHOLD:
        msg = f"WARNING: Parameters have R-hat > {RHAT_THRESHOLD}."
        warnings_found.append(msg)
        print(msg)

    min_ess = summary['ess_bulk'].min()
    print(f"  Min ESS (bulk) observed: {min_ess:.1f}")
    if min_ess < (NEFF_RATIO_THRESHOLD * ITER_SAMPLING * CHAINS):
        msg = "WARNING: Low effective sample size detected."
        warnings_found.append(msg)
        print(msg)

    return warnings_found


def create_ranking_summary(theta_df_in, id_var_in, year_col_in, cat_map_in, years_in):
    years_sorted = sorted(years_in)
    if len(years_sorted) < 2:
        return pd.DataFrame()
    year1, year_last = years_sorted[0], years_sorted[-1]
    summary_data = []
    unique_ids = theta_df_in[id_var_in].unique()
    df_y1 = theta_df_in[theta_df_in[year_col_in] == year1].set_index(id_var_in)
    df_y_last = theta_df_in[theta_df_in[year_col_in] == year_last].set_index(id_var_in)

    for cat_name in cat_map_in.keys():
        col_name = f"theta_{cat_name}"
        if col_name not in theta_df_in.columns:
            continue
        rank_y1 = df_y1[col_name].rank(ascending=False, method='min')
        rank_y_last = df_y_last[col_name].rank(ascending=False, method='min')

        for team_id in unique_ids:
            val_y1 = df_y1.loc[team_id, col_name] if team_id in df_y1.index else np.nan
            val_y_last = df_y_last.loc[team_id, col_name] if team_id in df_y_last.index else np.nan
            improvement = val_y_last - val_y1 if pd.notna(val_y1) and pd.notna(val_y_last) else np.nan
            r_y1 = rank_y1.get(team_id, np.nan)
            r_y_last = rank_y_last.get(team_id, np.nan)
            rank_change = r_y1 - r_y_last if pd.notna(r_y1) and pd.notna(r_y_last) else np.nan

            summary_data.append({
                'Team':                team_id, 'Category': cat_name,
                f'Capability_{year1}': val_y1, f'Capability_{year_last}': val_y_last,
                'Improvement':         improvement, f'Rank_{year1}': r_y1, f'Rank_{year_last}': r_y_last,
                'Rank_Change':         rank_change
            })
    return pd.DataFrame(summary_data)


# -----------------------------------------
# Main Execution Entry Point
# -----------------------------------------
def main():
    # --- Section 1: Data Simulation ---
    RUN_SIMULATION = True
    df_raw = None
    true_params = None

    if RUN_SIMULATION:
        sim_years_list = [2023, 2024, 2025]
        sim_n_resp = 20
        df_raw, true_params = simulate_dora_data_reorg(
            sim_years_reorg=sim_years_list,
            reorg_map_child_to_parents=REORG_MAPPING_Y_LAST_TO_Y1,
            sim_n_resp_per_team_year=sim_n_resp,
            category_map_sim=CATEGORY_MAPPING_INITIAL,
            response_options_sim=RESPONSE_OPTIONS
        )
    else:
        print("--- Section 1: Loading Real Data ---")
        try:
            # df_raw = pd.read_csv('your_dora_survey_data.csv')
            print("ERROR: Real data loading path not set. Please load data into df_raw.")
            return
        except FileNotFoundError:
            print("ERROR: Real data file not found.")
            return

    # --- Section 2: Data Preprocessing ---
    print("\n--- Section 2: Loading & Preprocessing Data ---")
    try:
        actual_years = sorted(df_raw[YEAR_COL].unique())
        actual_teams = df_raw[ID_VAR].unique()
        actual_questions_in_data = df_raw.columns.intersection(QUESTION_TO_CAT_NAME_INITIAL.keys())
    except KeyError as e:
        print(f"ERROR: Missing expected column: {e}")
        return

    print(f"Data contains years: {actual_years}")
    print(f"Data contains {len(actual_teams)} unique teams.")

    questions_to_use = []
    for q in actual_questions_in_data:
        if df_raw[q].notna().any():
            questions_to_use.append(q)

    questions_to_use = sorted(list(set(questions_to_use)))
    if not questions_to_use:
        print("ERROR: No valid questions found in data.")
        return

    category_mapping_final = {}
    question_categories_final = {}
    question_to_cat_name_final = {}
    cat_idx_final = 0
    final_cat_names = []

    for cat_name_orig, qs_orig in CATEGORY_MAPPING_INITIAL.items():
        qs_in_cat_to_use = [q for q in qs_orig if q in questions_to_use]
        if qs_in_cat_to_use:
            category_mapping_final[cat_name_orig] = qs_in_cat_to_use
            final_cat_names.append(cat_name_orig)
            for q in qs_in_cat_to_use:
                question_categories_final[q] = cat_idx_final
                question_to_cat_name_final[q] = cat_name_orig
            cat_idx_final += 1

    cat_idx_to_name_final = {idx: name for idx, name in enumerate(final_cat_names)}
    N_LATENT_final = len(category_mapping_final)

    df_long = df_raw.melt(id_vars=[YEAR_COL, ID_VAR], value_vars=questions_to_use, var_name='question',
                          value_name='response')
    df_long.dropna(subset=['response'], inplace=True)

    df_long['response_numeric'] = pd.to_numeric(df_long['response'], errors='coerce')
    df_long.dropna(subset=['response_numeric'], inplace=True)
    df_long['response'] = df_long['response_numeric'].astype(int)

    is_valid_response = df_long['response'].isin(RESPONSE_OPTIONS)
    df_long = df_long[is_valid_response]
    df_long.drop(columns=['response_numeric'], inplace=True)

    df_long['q_cat_final'] = df_long['question'].map(question_categories_final)

    df_long['group'] = df_long[ID_VAR].astype(str) + "|" + df_long[YEAR_COL].astype(str)
    groups_final = sorted(df_long['group'].unique())
    group_idx_map = {g: i for i, g in enumerate(groups_final)}
    df_long['group_idx'] = df_long['group'].map(group_idx_map)
    group_idx_to_info = {idx: {"team": g.split('|')[0], "year": int(g.split('|')[1])} for g, idx in
                         group_idx_map.items()}

    question_idx_map = {q: i for i, q in enumerate(questions_to_use)}
    df_long['question_idx'] = df_long['question'].map(question_idx_map)

    q_idx_to_dim_idx = {q_idx: question_categories_final[q] for q, q_idx in question_idx_map.items()}
    question_to_dimension_pymc = np.array([q_idx_to_dim_idx[k] for k in range(len(question_idx_map))])

    N_final = df_long.shape[0]
    J_final = len(groups_final)
    K_final = len(questions_to_use)
    C_final = N_CATEGORIES_RESPONSE
    L_final = N_LATENT_final
    print(f"\nFinal Data Counts: N={N_final}, J={J_final}, K={K_final}, C={C_final}, L={L_final}")

    # --- Section 3: Defining PyMC Model ---
    print("\n--- Section 3: Defining PyMC Model ---")
    with pm.Model() as dora_mgrm_model:
        z = pm.Normal("z", mu=0, sigma=1, shape=(L_final, J_final))

        # Unpacked Cholesky correlation mapping directly
        chol_cov, corr, sigma_val = pm.LKJCholeskyCov(
            "chol_cov", n=L_final, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0)
        )

        cov = pm.Deterministic("cov", pt.dot(chol_cov, chol_cov.T))
        sigma = pm.Deterministic("sigma", sigma_val)
        Omega = pm.Deterministic("Omega", corr)

        mu = pm.Normal("mu", mu=0, sigma=1, shape=(L_final, 1))
        theta = pm.Deterministic("theta", (mu + pt.dot(chol_cov, z)).T)

        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=K_final)

        first_cut = pm.Normal("first_cut", mu=0.0, sigma=3.0, shape=(K_final, 1))
        cut_diffs = pm.Exponential("cut_diffs", lam=1.0, shape=(K_final, C_final - 2))
        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate([first_cut, first_cut + pt.cumsum(cut_diffs, axis=1)], axis=1)
        )

        obs_q_idx = df_long['question_idx'].values
        obs_g_idx = df_long['group_idx'].values
        obs_dim_idx = question_to_dimension_pymc[obs_q_idx]

        eta = a[obs_q_idx] * theta[obs_g_idx, obs_dim_idx]

        y_obs = pm.OrderedLogistic(
            "y_obs",
            eta=eta,
            cutpoints=cutpoints[obs_q_idx],
            observed=df_long['response'].values - 1
        )

    # --- Section 4: Running PyMC Sampler ---
    print("\n--- Section 4: Running PyMC Sampler ---")
    with dora_mgrm_model:
        idata = pm.sample(
            draws=ITER_SAMPLING,
            tune=ITER_WARMUP,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            target_accept=0.85,
            init="jitter+adapt_diag"
        )

    # --- Section 5: Run Diagnostics ---
    diagnostic_warnings = run_diagnostic_checks(idata, df_long, cat_idx_to_name_final, question_idx_map)

    # --- Section 6: Extract and Save Results ---
    print("\n--- Section 6: Extracting and Processing Results ---")
    posterior_means = idata.posterior.mean(dim=["chain", "draw"])

    theta_means = posterior_means["theta"].values
    theta_columns = [f"theta_{cat_idx_to_name_final[l]}" for l in range(L_final)]
    theta_df_out = pd.DataFrame(theta_means, columns=theta_columns)

    theta_df_out['group_idx'] = list(group_idx_map.values())
    theta_df_out['group'] = theta_df_out['group_idx'].map({v: k for k, v in group_idx_map.items()})
    theta_df_out[[ID_VAR, YEAR_COL]] = theta_df_out['group'].str.split('|', expand=True)
    theta_df_out[YEAR_COL] = theta_df_out[YEAR_COL].astype(int)
    theta_df_out = theta_df_out[[ID_VAR, YEAR_COL, 'group', 'group_idx'] + theta_columns]

    for group in theta_df_out['group']:
        _id_var, _year = group.split('|')
        _year = int(_year)
        for theta_column in theta_columns:
            cat_name = theta_column[6:]
            questions_in_category = category_mapping_final.get(cat_name, [])
            raw_subset = df_raw[(df_raw[ID_VAR] == _id_var) & (df_raw[YEAR_COL] == _year)][questions_in_category]
            if raw_subset.dropna(how='all').empty:
                theta_df_out.loc[(theta_df_out['group'] == group), theta_column] = float('nan')

    theta_filepath = os.path.join(OUTPUT_DIR, 'team_capability_estimates.csv')
    theta_df_out.to_csv(theta_filepath, index=False)
    print(f"Team capability estimates saved to {theta_filepath}")

    Omega_mean_val = posterior_means["Omega"].values
    category_names_final = list(cat_idx_to_name_final.values())
    corr_df_out = pd.DataFrame(Omega_mean_val, index=category_names_final, columns=category_names_final)
    corr_filepath = os.path.join(OUTPUT_DIR, 'capability_correlations.csv')
    corr_df_out.to_csv(corr_filepath)
    print(f"Capability correlation matrix saved to {corr_filepath}")

    mu_means_val = posterior_means["mu"].values.flatten()
    sigma_means_val = posterior_means["sigma"].values.flatten()
    mu_map_for_plot = dict(zip(category_names_final, mu_means_val))
    sigma_map_for_plot = dict(zip(category_names_final, sigma_means_val))

    ranking_summary_df = create_ranking_summary(theta_df_out, ID_VAR, YEAR_COL, category_mapping_final, actual_years)
    if not ranking_summary_df.empty:
        ranking_filepath = os.path.join(OUTPUT_DIR, 'team_rankings_by_category.csv')
        ranking_summary_df.to_csv(ranking_filepath, index=False, float_format='%.3f')

    # --- Section 8: Visualizations ---
    print("\n--- Section 8: Generating Visualizations ---")
    if theta_df_out is not None:
        actual_years_final = sorted(theta_df_out[YEAR_COL].unique())
        if len(actual_years_final) >= 2:
            years_to_plot = [actual_years_final[0], actual_years_final[-1]]
            for cat_name in category_mapping_final.keys():
                plot_reorg_slope_chart_standardized_internal(
                    theta_df_viz=theta_df_out, category_name_viz=cat_name,
                    id_var_viz=ID_VAR, year_col_viz=YEAR_COL, years_viz=years_to_plot,
                    mu_map_viz=mu_map_for_plot, sigma_map_viz=sigma_map_for_plot,
                    parent_to_child_map=PARENT_TO_CHILD_MAPPING, output_dir_viz=OUTPUT_DIR
                )

            plot_likert_response_distributions(
                df_raw=df_raw, theta_df=theta_df_out,
                category_mapping=category_mapping_final, response_options=RESPONSE_OPTIONS,
                id_var=ID_VAR, year_col=YEAR_COL, target_year=actual_years_final[-1],
                output_dir=OUTPUT_DIR
            )

        if corr_df_out is not None:
            plot_omega_clustermap(corr_df_out, OUTPUT_DIR)

    print("\n--- Analysis Complete ---")
    print(f"Results, diagnostics, and plots saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()