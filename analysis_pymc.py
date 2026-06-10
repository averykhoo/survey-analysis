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
import pprint

# --- JAX Parallelization (MUST BE AT THE ABSOLUTE TOP, BEFORE OTHER IMPORTS) ---
import numpyro

numpyro.set_host_device_count(4)  # Tells JAX to use 4 CPU cores to run your chains in parallel

# --- PyCharm Terminal & Progress Bar Compatibility Fixes ---
from fastprogress import fastprogress

fastprogress.printing = lambda: True  # Force raw text progress lines in PyCharm

# --- Standard Imports ---
import datetime
import os
import re

import arviz as az
import jax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.special as sps
import seaborn as sns
from adjustText import adjust_text
from scipy.stats import gaussian_kde

# -----------------------------------------
# Global Constants & MCMC Settings
# -----------------------------------------

# numba/jax mode does not work with this model
# pytensor.config.mode = "NUMBA"  # Bypasses slow C++ compiler compilation on local machines
# pytensor.config.cxx = ""        # Disables standard C-compilation warnings

T_START = datetime.datetime.now()

ITER_WARMUP = 1000
ITER_SAMPLING = 1500
CHAINS = 4
RANDOM_SEED = 42

# --- Performance Setting ---
# Verify if JAX detects an active GPU (CUDA)
try:
    GPU_DETECTED = len(jax.devices("gpu")) > 0 if hasattr(jax, "devices") else False
except RuntimeError:
    GPU_DETECTED = False

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
all_parent_teams_in_map = {p for parents in REORG_MAPPING_Y_LAST_TO_Y1.values() for p in parents}
all_child_teams_in_map = set(REORG_MAPPING_Y_LAST_TO_Y1.keys())

for parent in all_parent_teams_in_map:
    PARENT_TO_CHILD_MAPPING[parent] = [child for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items()
                                       if parent in parents]

for child, parents in REORG_MAPPING_Y_LAST_TO_Y1.items():
    if len(parents) == 1 and parents[0] == child and child not in PARENT_TO_CHILD_MAPPING:
        PARENT_TO_CHILD_MAPPING[child] = [child]


# -----------------------------------------
# Helper Functions
# -----------------------------------------

def simulate_dora_data_reorg(sim_years_reorg,
                             reorg_map_child_to_parents,
                             sim_n_resp_per_team_year,
                             category_map_sim,
                             response_options_sim):
    """
    Generates simulated DORA survey data reflecting transitions over 3 years,
    complete with department metadata.
    """
    year1, year2, year3 = sim_years_reorg[0], sim_years_reorg[1], sim_years_reorg[2]

    year1_teams_sim = sorted(p for parents in reorg_map_child_to_parents.values() for p in parents)
    year2_teams_sim = sorted(p for parents in reorg_map_child_to_parents.values() for p in parents)
    year3_teams_sim = sorted(reorg_map_child_to_parents.keys())

    # Establish department mappings for demographics
    team_to_dept = {
        'Team_A': 'Engineering',
        'Team_B': 'Engineering',
        'Team_X': 'Engineering',
        'Team_Y': 'Engineering',
        'Team_C': 'Product',
        'Team_D': 'Product',
        'Team_Z': 'Product',
        'Team_E': 'Operations',
        'Team_F': 'Operations',
        'Team_G': 'Marketing'
    }

    # Use modern NumPy random generator
    rng = np.random.default_rng(RANDOM_SEED)

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
        if year == year1:
            teams_in_year = year1_teams_sim
        elif year == year2:
            teams_in_year = year2_teams_sim
        else:
            teams_in_year = year3_teams_sim

        for team in teams_in_year:
            if (team, year) not in true_latent_combined:
                continue

            for _ in range(sim_n_resp_per_team_year):
                row = {YEAR_COL: year, ID_VAR: team, 'dept': team_to_dept.get(team, 'Other')}
                for q in questions_sim:
                    cat = question_categories_sim[q]
                    theta_key = (team, year, cat)
                    if theta_key in latent_for_resp:
                        theta_true = latent_for_resp[theta_key]
                        a_val = true_a[q]
                        cp_val = true_cutpoints[q]

                        if year in [year1, year2] and q in ['q14', 'q15', 'q16']:
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
        'true_a':         true_a,
        'true_cutpoints': true_cutpoints,
        'true_Omega':     true_Omega,
        'true_latent':    true_latent_combined,
        'true_mu':        true_mu,
        'true_sigma':     true_sigma,
    }
    return df, true_params


def filename_escape(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def plot_reorg_slope_chart_standardized_internal(theta_df_viz,
                                                 category_name_viz,
                                                 id_var_viz,
                                                 year_col_viz,
                                                 years_viz,
                                                 mu_map_viz,
                                                 sigma_map_viz,
                                                 parent_to_child_map,
                                                 output_dir_viz,
                                                 target_dept=None):
    if target_dept is not None:
        theta_df_viz = theta_df_viz[theta_df_viz['dept'] == target_dept]

    pop_mean = mu_map_viz.get(category_name_viz)
    pop_sd = sigma_map_viz.get(category_name_viz)

    if pop_mean is None or pop_sd is None or pop_sd <= 1e-6 or theta_df_viz.empty:
        return

    raw_col_name = f"theta_{category_name_viz}_z"  # Trace lineage directly on the Standardized Z scale
    if raw_col_name not in theta_df_viz.columns:
        return

    # Dynamically resolve available years to prevent unpacking errors on subset-filtered data
    available_years = sorted([y for y in years_viz if y in theta_df_viz[year_col_viz].unique()])
    if len(available_years) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    texts = []
    plotted_coords = {y: {} for y in available_years}

    # Classify occurrence frequency to isolate transient single-year teams
    team_year_counts = theta_df_viz.groupby(id_var_viz)[year_col_viz].nunique().to_dict()

    # Establish rainbow color spectrum on the latest year's ranking (highest score on top)
    latest_year = available_years[-1]
    df_latest = theta_df_viz[theta_df_viz[year_col_viz] == latest_year].set_index(id_var_viz)
    df_latest_z = df_latest[raw_col_name]
    sorted_latest_teams = df_latest_z.sort_values(ascending=False).index.tolist()

    cmap_registry = plt.colormaps['Spectral'].resampled(max(1, len(sorted_latest_teams)))
    color_map_y_last = {team: cmap_registry(i) for i, team in enumerate(sorted_latest_teams)}

    # Plot coordinates for each available year
    for idx, yr in enumerate(available_years):
        df_yr = theta_df_viz[theta_df_viz[year_col_viz] == yr].set_index(id_var_viz)

        for team_id, row in df_yr.iterrows():
            val_z = row[raw_col_name]
            if pd.notna(val_z):
                plotted_coords[yr][team_id] = val_z

                # Visual distinction for single-year teams: plot as a "+" or "x" symbol
                if team_year_counts.get(team_id, 0) == 1:
                    ax.plot(yr, val_z, 'x', color='gray', markersize=8, mew=2)
                    label_align = 'right' if idx == 0 else 'left'
                    label_offset = -0.05 if idx == 0 else 0.05
                    texts.append(
                        ax.text(yr + label_offset, val_z, f"{team_id} (Single Yr)", ha=label_align, va='center',
                                fontsize=8, color='gray'))
                else:
                    point_color = color_map_y_last.get(team_id, 'gray') if yr == available_years[-1] else 'gray'
                    ax.plot(yr, val_z, 'o', color=point_color, markersize=5 if yr != available_years[-1] else 6)
                    label_align = 'right' if idx == 0 else 'left'
                    label_offset = -0.05 if idx == 0 else 0.05
                    texts.append(ax.text(yr + label_offset, val_z, team_id, ha=label_align, va='center', fontsize=8))

    # Trace lineages backward from current teams (Y3 -> Y2 -> Y1)
    for latest_team in sorted_latest_teams:
        team_color = color_map_y_last[latest_team]

        # Resolve parent relationships mapped from latest year to previous years
        parents = [parent for parent, children in parent_to_child_map.items() if latest_team in children]
        if not parents:
            parents = [latest_team]  # Fallback to identity mapping if no reorg

        for p_idx, p in enumerate(parents):
            y_to = available_years[-1]
            y_from = available_years[-2]

            if p in plotted_coords[y_from] and latest_team in plotted_coords[y_to]:
                y_from_val = plotted_coords[y_from][p]
                y_to_val = plotted_coords[y_to][latest_team]

                # Split/merge visualization: solid line for first parent, dashed for remaining parents
                line_style = '-' if p_idx == 0 else '--'
                ax.plot([y_from, y_to], [y_from_val, y_to_val], linestyle=line_style, lw=2.5, color=team_color,
                        alpha=0.7)

                # Trace back from Y2 to Y1
                if len(available_years) > 2:
                    y_prev = available_years[-3]
                    if p in plotted_coords[y_prev]:
                        y_prev_val = plotted_coords[y_prev][p]
                        ax.plot([y_prev, y_from], [y_prev_val, y_from_val], linestyle=line_style, lw=1.5,
                                color=team_color, alpha=0.7)

    # Plot transient/dissolved teams (those not active in latest year) as dotted gray lines
    for idx in range(len(available_years) - 1):
        y_from = available_years[idx]
        y_to = available_years[idx + 1]

        for team, val_from in plotted_coords[y_from].items():
            if team not in sorted_latest_teams and team in plotted_coords[y_to]:
                ax.plot([y_from, y_to], [val_from, plotted_coords[y_to][team]], linestyle=':', color='gray', alpha=0.4)

    adjust_text(texts, ax=ax, force_points=(0.2, 0.3), arrowprops={'arrowstyle': '-', 'color': 'gray', 'lw': 0.5})

    title_suffix = f" - {target_dept}" if target_dept else ""
    years_str = " - ".join(map(str, available_years))  # Resolved NameError dynamically
    ax.set_title(f'Standardized Capability Evolution: {category_name_viz} ({years_str}){title_suffix}', fontsize=14,
                 weight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Capability Z-Score (Std. Devs from Mean)', fontsize=12)
    ax.set_xticks(available_years)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.0, alpha=0.8)
    ax.set_xlim(available_years[0] - 0.4, available_years[-1] + 0.4)

    filepath = os.path.join(output_dir_viz, f'std_reorg_slope_{filename_escape(category_name_viz)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


def plot_omega_clustermap(corr_df_viz, output_dir_viz):
    if corr_df_viz is None or corr_df_viz.empty:
        return
    n_cats = corr_df_viz.shape[0]
    figsize = (max(8, n_cats * 0.9), max(7, n_cats * 0.8))
    try:
        # Inverted Colormap: Blue is positive, Red is negative
        cluster_grid = sns.clustermap(
            corr_df_viz, method='average', metric='euclidean', cmap='coolwarm_r',
            vmin=-1, vmax=1, center=0, annot=(n_cats <= 12), fmt=".2f",
            linewidths=.5, linecolor='lightgray', figsize=figsize
        )
        cluster_grid.figure.suptitle('Capability Correlations (Mean Omega)', y=1.02)
        filepath = os.path.join(output_dir_viz, 'omega_clustermap.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"WARNING: Clustermap generation failed: {e}")


def plot_likert_response_distributions(df_raw,
                                       theta_df,
                                       category_mapping,
                                       response_options,
                                       id_var,
                                       year_col,
                                       target_year,
                                       output_dir,
                                       target_dept=None):
    if target_dept is not None:
        theta_df = theta_df[theta_df['dept'] == target_dept]
        df_raw = df_raw[df_raw['dept'] == target_dept]

    df_overall_year = theta_df[theta_df[year_col] == target_year]
    df_raw_year = df_raw[df_raw[year_col] == target_year]

    if df_overall_year.empty or df_raw_year.empty:
        return

    # Inverted Likert colors: negative (dislike) is red/orange, positive (like) is blue
    cmap = plt.colormaps['vlag'].resampled(len(response_options))
    colors = [cmap(r - 1) for r in response_options[::-1]]

    for category, questions in category_mapping.items():
        df_long_cat = df_raw_year.melt(id_vars=[id_var],
                                       value_vars=questions,
                                       var_name='question',
                                       value_name='response',
                                       ).dropna()
        if df_long_cat.empty:
            continue

        df_long_cat['response'] = pd.to_numeric(df_long_cat['response'], errors='coerce')
        df_long_cat = df_long_cat[df_long_cat['response'].isin(response_options)]
        df_long_cat['response'] = df_long_cat['response'].astype(int)

        counts = df_long_cat.groupby([id_var, 'response']).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        theta_col = f"theta_{category}_z"
        if theta_col in df_overall_year.columns:
            ordered_teams = df_overall_year.sort_values(by=theta_col, ascending=True)[id_var].tolist()
        else:
            ordered_teams = percentages.index.tolist()

        final_order = [t for t in ordered_teams if t in percentages.index]
        plot_df = percentages.reindex(index=final_order, columns=response_options[::-1], fill_value=0)

        fig_height = max(5, len(plot_df) * 0.6)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        plot_df.plot(kind='barh', stacked=True, color=colors, edgecolor='white', linewidth=0.5, ax=ax)

        title_suffix = f" - {target_dept}" if target_dept else ""
        ax.set_title(f"{category} ({target_year}){title_suffix}\nTeam Response Distribution (Sorted by Rank)\n",
                     fontsize=13,
                     weight='bold')
        ax.set_xlabel('Percentage of Responses', fontsize=11)
        ax.set_ylabel('Team (Capability Rank - Bottom to Top)', fontsize=11)
        ax.set_xlim([0, 100])
        ax.legend(response_options[::-1],
                  title='Response',
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.15),
                  ncol=len(response_options),
                  frameon=False)

        for idx, (team, row) in enumerate(plot_df.iterrows()):
            start_pos = 0
            for j, val in enumerate(row):
                if val > 3.5:
                    text_color = 'white' if j in (0, len(response_options) - 1) else 'black'
                    ax.text(start_pos + val / 2,
                            idx,
                            f'{val:.0f}%',
                            va='center',
                            ha='center',
                            color=text_color,
                            fontsize=9)
                start_pos += val

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'likert_{filename_escape(category)}.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)


def plot_team_ridge_plots(idata, theta_df, category_name, output_dir, target_dept=None):
    """
    Plots the posterior distribution of capability scores (theta) over years
    as stacked ridge plots for each team, fully matching the requested hatching
    and transparent-fill visual style.
    """
    if target_dept is not None:
        theta_df = theta_df[theta_df['dept'] == target_dept]

    if theta_df.empty:
        return

    # Extract posterior samples of theta: (samples, J, L)
    theta_samples = idata.posterior['theta_z'].stack(samples=['chain', 'draw']).values
    theta_samples = theta_samples.transpose(2, 0, 1)

    cat_names = list(CATEGORY_MAPPING_INITIAL.keys())
    if category_name not in cat_names:
        return
    cat_idx = cat_names.index(category_name)

    unique_teams = sorted(theta_df[ID_VAR].unique())
    unique_years = sorted(theta_df[YEAR_COL].unique())
    num_teams = len(unique_teams)

    fig, axes = plt.subplots(num_teams, 1, figsize=(14, 2.8 * num_teams), sharex=True)
    if num_teams == 1:
        axes = [axes]

    palette = sns.color_palette("muted", num_teams)

    for i, team in enumerate(unique_teams):
        ax = axes[i]
        team_color = palette[i % len(palette)]

        # Draw baseline axis (dotted blue line)
        ax.axhline(0, color='#1f4e78', linestyle=':', lw=1.5, zorder=0)

        # Plot density curves for each year
        for year in unique_years:
            grp_rows = theta_df[(theta_df[ID_VAR] == team) & (theta_df[YEAR_COL] == year)]
            if grp_rows.empty:
                continue
            grp_idx = grp_rows['group_idx'].values[0] - 1  # 0-based for python index

            samples = theta_samples[:, grp_idx, cat_idx]

            # Reconstruct KDE coordinates manually for robust custom hatching/fill options
            kde = gaussian_kde(samples)
            x_vals = np.linspace(-3.5, 3.5, 300)
            y_vals = kde(x_vals)

            if year == 2023:
                # 2023: No fill, colored outline
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
            elif year == 2025:
                # 2025: Line hatched fill, colored outline
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
                ax.fill_between(x_vals, y_vals, color='none', edgecolor=team_color, hatch='////', lw=0, zorder=2)
            elif year == 2026:
                # 2026: Opaque solid fill
                ax.fill_between(x_vals, y_vals, color=team_color, alpha=0.85, zorder=1)

        # Draw red vertical line at the latest year's posterior mean
        latest_year = unique_years[-1]
        latest_rows = theta_df[(theta_df[ID_VAR] == team) & (theta_df[YEAR_COL] == latest_year)]
        if not latest_rows.empty:
            latest_grp_idx = latest_rows['group_idx'].values[0] - 1
            latest_mean = theta_samples[:, latest_grp_idx, cat_idx].mean()
            ax.axvline(latest_mean, color='red', linestyle='-', lw=2, zorder=5)

        # Labels & Aesthetics
        ax.set_ylabel(team, rotation=0, ha='right', va='center', fontsize=12, weight='bold', labelpad=15)
        ax.yaxis.set_ticks([])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Global population baseline reference line (0)
    for ax in axes:
        ax.axvline(0, color='#1f4e78', linestyle='--', lw=1, zorder=1)

    axes[0].text(0, axes[0].get_ylim()[1] * 1.15, "mean", ha='center', va='bottom', style='italic', fontsize=12)

    # Clean Legend matching the requested style (no dates on curves, just legend info)
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='black', label='2023'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch='////', label='2025'),
        mpatches.Patch(facecolor='gray', alpha=0.85, label='2026')
    ]
    fig.legend(handles=legend_elements,
               loc='upper right',
               bbox_to_anchor=(0.95, 0.98),
               ncol=3,
               frameon=False,
               fontsize=11)

    plt.xlabel('Capability Score (Theta Standard Deviations)', fontsize=12, labelpad=15)
    plt.xlim(-3.5, 3.5)
    plt.tight_layout()

    suffix = f"_{target_dept.lower()}" if target_dept else ""
    filepath = os.path.join(output_dir, f'ridge_plots_{filename_escape(category_name)}{suffix}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"  Saved team capability ridge plots to {filepath}")


def plot_category_response_functions(question_id, idata, question_idx_map, output_dir):
    """Plots the Category Response Functions P(Y=c|theta) for a question."""
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]

    a_samples = idata.posterior['a'].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = idata.posterior['cutpoints'].values.reshape(-1, len(question_idx_map), N_CATEGORIES_RESPONSE - 1)[
        :, q_idx, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)
    probs = np.zeros((len(theta_vals), N_CATEGORIES_RESPONSE))

    for i, th in enumerate(theta_vals):
        draw_probs = []
        for a_val, cuts in zip(a_samples, cut_samples):
            eta = a_val * th
            cum_prob = sps.expit(cuts - eta)
            p = np.zeros(N_CATEGORIES_RESPONSE)
            p[0] = cum_prob[0]
            for c in range(1, N_CATEGORIES_RESPONSE - 1):
                p[c] = cum_prob[c] - cum_prob[c - 1]
            p[-1] = 1.0 - cum_prob[-1]
            draw_probs.append(p)
        probs[i, :] = np.mean(draw_probs, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.colormaps['viridis'].resampled(N_CATEGORIES_RESPONSE)
    for c in range(N_CATEGORIES_RESPONSE):
        ax.plot(theta_vals, probs[:, c], color=colors(c), lw=2, label=f"Category {c + 1}")
    ax.set_xlabel("Latent Trait (Theta)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Category Response Functions: {question_id}")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    filepath = os.path.join(output_dir, f'crf_{filename_escape(question_id)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_predicted_vs_empirical_dist(question_id,
                                     idata,
                                     df_long,
                                     question_idx_map,
                                     output_dir,
                                     question_to_dimension_pymc):
    """Plots model-predicted response distribution vs. empirical data."""
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]
    obs_data = df_long[df_long['question_idx'] == q_idx]['response'].values

    a_samples = idata.posterior['a'].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = idata.posterior['cutpoints'].values.reshape(-1, len(question_idx_map), N_CATEGORIES_RESPONSE - 1)[
        :, q_idx, :]

    grp_indices = df_long[df_long['question_idx'] == q_idx]['group_idx'].values
    theta_samples = idata.posterior['theta'].values.reshape(-1, len(df_long['group_idx'].unique()),
                                                            len(CATEGORY_MAPPING_INITIAL))

    q_dim = question_to_dimension_pymc[q_idx]

    pred_draws = []
    rng = np.random.default_rng(RANDOM_SEED)
    for a_val, cuts, ths in zip(a_samples, cut_samples, theta_samples):
        for g_idx in grp_indices:
            th = ths[g_idx, q_dim]
            eta = a_val * th
            cum_prob = sps.expit(cuts - eta)
            p = np.zeros(N_CATEGORIES_RESPONSE)
            p[0] = cum_prob[0]
            for c in range(1, N_CATEGORIES_RESPONSE - 1):
                p[c] = cum_prob[c] - cum_prob[c - 1]
            p[-1] = 1.0 - cum_prob[-1]
            p = np.maximum(p, 0)
            p /= p.sum()
            pred_draws.append(rng.choice(RESPONSE_OPTIONS, p=p))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(min(RESPONSE_OPTIONS) - 0.5, max(RESPONSE_OPTIONS) + 1.5, 1)
    ax.hist(obs_data, bins=bins, alpha=0.5, color='gray', label='Observed', density=True, edgecolor='black')
    ax.hist(pred_draws, bins=bins, alpha=0.5, color='dodgerblue', label='Predicted', density=True, edgecolor='blue')
    ax.set_xlabel("Response Category")
    ax.set_ylabel("Density")
    ax.set_xticks(RESPONSE_OPTIONS)
    ax.set_title(f"Item Fit Plot: {question_id}")
    ax.legend()
    filepath = os.path.join(output_dir, f'item_fit_{filename_escape(question_id)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_item_information_function(question_id, idata, question_idx_map, output_dir):
    """Plots the Item Information Function (IIF)."""
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]

    a_samples = idata.posterior['a'].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = idata.posterior['cutpoints'].values.reshape(-1, len(question_idx_map), N_CATEGORIES_RESPONSE - 1)[
        :, q_idx, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)
    info = np.zeros(len(theta_vals))

    for i, th in enumerate(theta_vals):
        draw_infos = []
        for a_val, cuts in zip(a_samples, cut_samples):
            item_info = 0
            for c in range(N_CATEGORIES_RESPONSE - 1):
                logit = cuts[c] - a_val * th
                P_star = sps.expit(logit)
                item_info += P_star * (1.0 - P_star)
            draw_infos.append(item_info * (a_val ** 2))
        info[i] = np.mean(draw_infos)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_vals, info, color='navy', lw=2)
    ax.set_xlabel("Latent Trait (Theta)")
    ax.set_ylabel("Item Information")
    ax.set_title(f"Item Information Function: {question_id}")
    ax.grid(True, linestyle=":", alpha=0.6)
    filepath = os.path.join(output_dir, f'iif_{filename_escape(question_id)}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_test_information_function(idata, K_final, output_dir):
    """Plots the Test Information Function (TIF) and Standard Error of Measurement (SEM)."""
    a_samples = idata.posterior['a'].values.reshape(-1, K_final)
    cut_samples = idata.posterior['cutpoints'].values.reshape(-1, K_final, N_CATEGORIES_RESPONSE - 1)

    theta_vals = np.linspace(-3.5, 3.5, 100)
    tif = np.zeros(len(theta_vals))

    for i, th in enumerate(theta_vals):
        draw_infos = np.zeros(a_samples.shape[0])
        for k in range(K_final):
            a_k = a_samples[:, k]
            cuts_k = cut_samples[:, k, :]
            for c in range(N_CATEGORIES_RESPONSE - 1):
                logit = cuts_k[:, c] - a_k * th
                P_star = sps.expit(logit)
                draw_infos += P_star * (1.0 - P_star) * (a_k ** 2)
        tif[i] = np.mean(draw_infos)

    sem = 1.0 / np.sqrt(np.maximum(tif, 1e-6))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(theta_vals, tif, color='darkblue', lw=2.5, label='Test Information')
    ax1.set_xlabel("Latent Trait (Theta)")
    ax1.set_ylabel("Test Information", color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(theta_vals, sem, color='firebrick', lw=2, ls='--', label='SEM')
    ax2.set_ylabel("Standard Error of Measurement (SEM)", color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')

    plt.title("Test Information Function & Standard Error of Measurement")
    filepath = os.path.join(output_dir, 'tif_sem_plot.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def run_diagnostic_checks(idata, df_long, cat_idx_to_name_map, question_idx_map):
    print("\n--- RUNNING DIAGNOSTIC CHECKS ---")
    warnings_found = []

    # 1. Check for groups with low response counts
    responses_per_group = df_long.groupby('group').size()
    low_groups = responses_per_group[responses_per_group < MIN_RESPONSES_PER_GROUP]
    for g, sz in low_groups.items():
        warnings_found.append(
            f"WARNING: Group '{g}' has only {sz} responses (minimum threshold is {MIN_RESPONSES_PER_GROUP})."
        )

    # 2. Check for Divergences
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats.data_vars:
        divergences = idata.sample_stats.diverging.sum().item()
        if divergences > 0:
            msg = f"CRITICAL WARNING: {divergences} divergent transition(s) found! Review parametrization."
            warnings_found.append(msg)
        else:
            print("  Divergences: OK (0 found)")

    # 3. Check Convergence Statistics
    summary = az.summary(idata, var_names=["mu", "sigma", "a"], round_to=3)

    max_rhat = summary['r_hat'].max()
    print(f"  Max R-hat observed: {max_rhat:.3f}")
    if max_rhat > RHAT_THRESHOLD:
        msg = f"WARNING: Parameters have R-hat > {RHAT_THRESHOLD}. ({max_rhat=})"
        warnings_found.append(msg)
        print(msg)

    min_ess = summary['ess_bulk'].min()
    print(f"  Min ESS (bulk) observed: {min_ess:.1f}")
    if min_ess < (NEFF_RATIO_THRESHOLD * ITER_SAMPLING * CHAINS):
        msg = f"WARNING: Low effective sample size ({min_ess=}) detected (ESS < {NEFF_RATIO_THRESHOLD * ITER_SAMPLING * CHAINS:.0f})."
        warnings_found.append(msg)

    # 4. Check for Low Discrimination Items using both question and category maps
    idx_to_question = {v: k for k, v in question_idx_map.items()}
    a_summary = summary[summary.index.str.startswith("a")]
    for idx_str, row in a_summary.iterrows():
        match = re.search(r"a\[(\d+)\]", idx_str)
        if match:
            q_idx = int(match.group(1))
            q_name = idx_to_question.get(q_idx, f"Idx {q_idx}")
            mean_a = row['mean']
            if mean_a < LOW_DISCRIMINATION_THRESHOLD:
                # Find category index and map to category name
                cat_idx = QUESTION_TO_CAT_NAME_INITIAL.get(q_name, "Unknown")
                warnings_found.append(
                    f"WARNING: Question '{q_name}' in category '{cat_idx}' has low mean discrimination of {mean_a:.3f}.")

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
                'Team':                    team_id,
                'Category':                cat_name,
                f'Capability_{year1}':     val_y1,
                f'Capability_{year_last}': val_y_last,
                'Improvement':             improvement,
                f'Rank_{year1}':           r_y1,
                f'Rank_{year_last}':       r_y_last,
                'Rank_Change':             rank_change
            })
    return pd.DataFrame(summary_data)


# -----------------------------------------
# Main Execution Entry Point
# -----------------------------------------
def main():
    # --- Section 1: Data Simulation ---
    RUN_SIMULATION = True
    df_raw: pd.DataFrame
    true_params: dict

    if RUN_SIMULATION:
        print("--- Section 1: Simulating Sample DORA Data with Reorg ---")
        # Structured exactly to represent requested 3-year timeline: 2023, 2025, and 2026
        sim_years_list = [2023, 2025, 2026]
        sim_n_resp = 20
        df_raw, true_params = simulate_dora_data_reorg(
            sim_years_reorg=sim_years_list,
            reorg_map_child_to_parents=REORG_MAPPING_Y_LAST_TO_Y1,
            sim_n_resp_per_team_year=sim_n_resp,
            category_map_sim=CATEGORY_MAPPING_INITIAL,
            response_options_sim=RESPONSE_OPTIONS
        )
        print(df_raw.head().to_markdown())
    else:
        print("--- Section 1: Loading Real Data ---")
        try:
            # df_raw = pd.read_csv('your_dora_survey_data.csv')
            print("ERROR: Real data loading path not set. Please load data into df_raw.")
            return
        except FileNotFoundError:
            print("ERROR: Real data file not found.")
            return

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')
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

    df_long = df_raw.melt(id_vars=[YEAR_COL, ID_VAR, 'dept'],
                          value_vars=questions_to_use,
                          var_name='question',
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
    group_idx_to_info = {idx: {"team": g.split('|')[0], "year": int(g.split('|')[1])}
                         for g, idx in group_idx_map.items()}

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

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')

    # --- Section 3: Defining PyMC Model ---
    print("\n--- Section 3: Defining PyMC Model ---")
    with pm.Model() as dora_mgrm_model:
        # Start latent traits at exactly 0
        z = pm.Normal("z", mu=0, sigma=1, shape=(L_final, J_final), initval=np.zeros((L_final, J_final)))

        # LKJ Covariance modeling directly
        chol_cov, corr, sigma_val = pm.LKJCholeskyCov(
            "chol_cov", n=L_final, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0)
        )

        pm.Deterministic("cov", pt.dot(chol_cov, chol_cov.T))
        pm.Deterministic("sigma", sigma_val)
        pm.Deterministic("Omega", corr)

        # Start means at 0
        mu = pm.Normal("mu", mu=0, sigma=1, shape=(L_final, 1), initval=np.zeros((L_final, 1)))
        theta = pm.Deterministic("theta", (mu + pt.dot(chol_cov, z)).T)

        # Track standardized capability scores (Z-scores) natively inside the posterior trace
        pm.Deterministic("theta_z", (theta - mu.T) / sigma_val, dims=None)

        # Start discrimination at exactly 1.0 (Standard Rasch assumption baseline)
        a = pm.LogNormal("a", mu=0.0, sigma=0.5, shape=K_final, initval=np.ones(K_final))

        # Force cutpoints to start perfectly evenly spaced (e.g., -2, -1, 0, 1, 2)
        first_cut = pm.Normal("first_cut", mu=0.0, sigma=3.0, shape=(K_final, 1), initval=np.full((K_final, 1), -2.0))
        cut_diffs = pm.Exponential("cut_diffs",
                                   lam=1.0,
                                   shape=(K_final, C_final - 2),
                                   initval=np.ones((K_final, C_final - 2)))

        cutpoints = pm.Deterministic(
            "cutpoints",
            pt.concatenate([first_cut, first_cut + pt.cumsum(cut_diffs, axis=1)], axis=1)
        )

        obs_q_idx = df_long['question_idx'].values
        obs_g_idx = df_long['group_idx'].values
        obs_dim_idx = question_to_dimension_pymc[obs_q_idx]

        eta = a[obs_q_idx] * theta[obs_g_idx, obs_dim_idx]

        # Likelihood is declared directly into model context, suppressing unused-variable warnings
        pm.OrderedLogistic(
            "y_obs",
            eta=eta,
            cutpoints=cutpoints[obs_q_idx],
            observed=df_long['response'].values - 1
        )

    # --- Save PyMC Model Directed Graph ---
    try:
        g = pm.model_to_graphviz(dora_mgrm_model)
        g.render(filename=os.path.join(OUTPUT_DIR, 'model_dag'), format='png', cleanup=True)
        print("Saved model DAG visualization to model_dag.png")
    except Exception as e:
        print(f"Skipped Graphviz model visualization (install Graphviz for this feature): {e}")

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')
    # --- Section 4: Running PyMC Sampler ---
    print("\n--- Section 4: Running PyMC Sampler ---")

    if GPU_DETECTED:
        print("Sampling using JAX (NumPyro) on GPU...")
    else:
        print("Sampling using JAX (NumPyro) on CPU (Optimized via XLA)...")

    with dora_mgrm_model:
        idata = pm.sample(
            draws=ITER_SAMPLING,
            tune=ITER_WARMUP,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            target_accept=0.80,  # Standard NUTS target (reduces leapfrog steps for speed)
            init="adapt_diag",
            nuts_sampler='numpyro'  # or fallback to `nuts` without numpyro and jax installed
        )

        # Execute Posterior Predictive Sampling
        print("Sampling posterior predictive distribution...")
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=RANDOM_SEED)

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')
    # --- Section 5: Run Diagnostics ---
    print("\n--- Section 5: Run Diagnostics ---")
    diagnostic_warnings = run_diagnostic_checks(idata, df_long, cat_idx_to_name_final, question_idx_map)

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')
    # --- Section 6: Extract and Save Results (Including Credible Intervals) ---
    print("\n--- Section 6: Extracting and Processing Results ---")

    # Calculate HDI intervals (95% default)
    hdis = az.hdi(idata, hdi_prob=0.95)

    posterior_means = idata.posterior.mean(dim=["chain", "draw"])
    category_names_final = list(cat_idx_to_name_final.values())

    # 1. mu parameter extraction
    mu_means = posterior_means["mu"].values.flatten()
    mu_hdi_lower = hdis["mu"].values[:, 0, 0]  # Extract first sub-array index for shapes
    mu_hdi_upper = hdis["mu"].values[:, 0, 1]

    mu_summary_df = pd.DataFrame({
        "Category":     category_names_final,
        "mu_mean":      mu_means,
        "mu_hdi_lower": mu_hdi_lower,
        "mu_hdi_upper": mu_hdi_upper
    })
    mu_summary_df.to_csv(os.path.join(OUTPUT_DIR, "mu_parameter_estimates.csv"), index=False)

    # 2. sigma parameter extraction
    sigma_means_val = posterior_means["sigma"].values.flatten()
    sigma_hdi_lower = hdis["sigma"].values[:, 0]
    sigma_hdi_upper = hdis["sigma"].values[:, 1]

    sigma_summary_df = pd.DataFrame({
        "Category":        category_names_final,
        "sigma_mean":      sigma_means_val,
        "sigma_hdi_lower": sigma_hdi_lower,
        "sigma_hdi_upper": sigma_hdi_upper
    })
    sigma_summary_df.to_csv(os.path.join(OUTPUT_DIR, "sigma_parameter_estimates.csv"), index=False)

    # 3. theta capability scores extraction with lower/upper HDI and Z-scores
    theta_means = posterior_means["theta"].values
    theta_hdis = hdis["theta"].values  # Shape: (J, L, 2)
    theta_z_means = posterior_means["theta_z"].values
    theta_z_hdis = hdis["theta_z"].values  # Shape: (J, L, 2)

    theta_columns = [f"theta_{cat}" for cat in category_names_final]
    theta_df_out = pd.DataFrame(theta_means, columns=theta_columns)

    # Append low/high hdi boundaries and Z-scores (standard deviations from mean)
    for l_idx, cat in enumerate(category_names_final):
        theta_df_out[f"theta_{cat}_hdi_lower"] = theta_hdis[:, l_idx, 0]
        theta_df_out[f"theta_{cat}_hdi_upper"] = theta_hdis[:, l_idx, 1]
        theta_df_out[f"theta_{cat}_z"] = theta_z_means[:, l_idx]
        theta_df_out[f"theta_{cat}_z_hdi_lower"] = theta_z_hdis[:, l_idx, 0]
        theta_df_out[f"theta_{cat}_z_hdi_upper"] = theta_z_hdis[:, l_idx, 1]

    theta_df_out['group_idx'] = list(group_idx_map.values())
    theta_df_out['group'] = theta_df_out['group_idx'].map({v: k for k, v in group_idx_map.items()})
    theta_df_out[[ID_VAR, YEAR_COL]] = theta_df_out['group'].str.split('|', expand=True)
    theta_df_out[YEAR_COL] = theta_df_out[YEAR_COL].astype(int)

    # Map departments for demographic visual filtering
    team_to_dept_map = df_raw[[ID_VAR, 'dept']].drop_duplicates().set_index(ID_VAR)['dept'].to_dict()
    theta_df_out['dept'] = theta_df_out[ID_VAR].map(team_to_dept_map)

    # Structure column order
    ordered_cols = [ID_VAR, YEAR_COL, 'dept', 'group', 'group_idx'] + theta_columns
    for cat in category_names_final:
        ordered_cols.extend([
            f"theta_{cat}_hdi_lower", f"theta_{cat}_hdi_upper",
            f"theta_{cat}_z", f"theta_{cat}_z_hdi_lower", f"theta_{cat}_z_hdi_upper"
        ])
    theta_df_out = theta_df_out[ordered_cols]

    # Perform response-presence masking to prevent imputed parameters showing up for unasked categories
    for group in theta_df_out['group']:
        _id_var, _year = group.split('|')
        _year = int(_year)
        for cat in category_names_final:
            questions_in_category = category_mapping_final.get(cat, [])
            raw_subset = df_raw[(df_raw[ID_VAR] == _id_var) & (df_raw[YEAR_COL] == _year)][questions_in_category]
            if raw_subset.dropna(how='all').empty:
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}"] = float('nan')
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}_hdi_lower"] = float('nan')
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}_hdi_upper"] = float('nan')
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}_z"] = float('nan')
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}_z_hdi_lower"] = float('nan')
                theta_df_out.loc[(theta_df_out['group'] == group), f"theta_{cat}_z_hdi_upper"] = float('nan')

    theta_filepath = os.path.join(OUTPUT_DIR, 'team_capability_estimates.csv')
    theta_df_out.to_csv(theta_filepath, index=False)
    print(f"Team capability estimates saved to {theta_filepath}")

    # 4. Correlation Matrix
    Omega_mean_val = posterior_means["Omega"].values
    corr_df_out = pd.DataFrame(Omega_mean_val, index=category_names_final, columns=category_names_final)
    corr_filepath = os.path.join(OUTPUT_DIR, 'capability_correlations.csv')
    corr_df_out.to_csv(corr_filepath)
    print(f"Capability correlation matrix saved to {corr_filepath}")

    mu_map_for_plot = dict(zip(category_names_final, mu_means))
    sigma_map_for_plot = dict(zip(category_names_final, sigma_means_val))

    ranking_summary_df = create_ranking_summary(theta_df_out, ID_VAR, YEAR_COL, category_mapping_final, actual_years)
    if not ranking_summary_df.empty:
        ranking_filepath = os.path.join(OUTPUT_DIR, 'team_rankings_by_category.csv')
        ranking_summary_df.to_csv(ranking_filepath, index=False, float_format='%.3f')

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')
    # --- Section 7: Visualizations ---
    print("\n--- Section 7: Generating Visualizations ---")

    # Example target filter: Focus on 'Engineering' department to demonstrate post-hoc demographics
    demographic_filter = 'Engineering'

    if theta_df_out is not None:
        actual_years_final = sorted(theta_df_out[YEAR_COL].unique())
        if len(actual_years_final) >= 2:
            years_to_plot = actual_years_final  # Plotting all available years on slope chart
            for cat_name in category_mapping_final.keys():
                # Raw (unfiltered) Slope Charts
                plot_reorg_slope_chart_standardized_internal(
                    theta_df_viz=theta_df_out,
                    category_name_viz=cat_name,
                    id_var_viz=ID_VAR,
                    year_col_viz=YEAR_COL,
                    years_viz=years_to_plot,
                    mu_map_viz=mu_map_for_plot,
                    sigma_map_viz=sigma_map_for_plot,
                    parent_to_child_map=PARENT_TO_CHILD_MAPPING,
                    output_dir_viz=OUTPUT_DIR,
                    target_dept=None)

                # Standardized Department-filtered Slope Charts
                plot_reorg_slope_chart_standardized_internal(
                    theta_df_viz=theta_df_out,
                    category_name_viz=cat_name,
                    id_var_viz=ID_VAR,
                    year_col_viz=YEAR_COL,
                    years_viz=years_to_plot,
                    mu_map_viz=mu_map_for_plot,
                    sigma_map_viz=sigma_map_for_plot,
                    parent_to_child_map=PARENT_TO_CHILD_MAPPING,
                    output_dir_viz=OUTPUT_DIR,
                    target_dept=demographic_filter)

                # Raw Likert response distributions
                plot_likert_response_distributions(
                    df_raw=df_raw,
                    theta_df=theta_df_out,
                    category_mapping=category_mapping_final,
                    response_options=RESPONSE_OPTIONS,
                    id_var=ID_VAR,
                    year_col=YEAR_COL,
                    target_year=actual_years_final[-1],
                    output_dir=OUTPUT_DIR,
                    target_dept=None)

                # Standardized capability distribution ridge curves (Matching your visual template)
                plot_team_ridge_plots(
                    idata=idata,
                    theta_df=theta_df_out,
                    category_name=cat_name,
                    output_dir=OUTPUT_DIR,
                    target_dept=None)

                # Standardized capability distribution ridge curves (Department filtered)
                plot_team_ridge_plots(
                    idata=idata,
                    theta_df=theta_df_out,
                    category_name=cat_name,
                    output_dir=OUTPUT_DIR,
                    target_dept=demographic_filter)

                # Item-level IRT functions for each question in this category
                for q_id in category_mapping_final[cat_name]:
                    plot_category_response_functions(q_id, idata, question_idx_map, OUTPUT_DIR)
                    plot_predicted_vs_empirical_dist(question_id=q_id,
                                                     idata=idata,
                                                     df_long=df_long,
                                                     question_idx_map=question_idx_map,
                                                     output_dir=OUTPUT_DIR,
                                                     question_to_dimension_pymc=question_to_dimension_pymc)
                    plot_item_information_function(q_id, idata, question_idx_map, OUTPUT_DIR)

        # Plot Clustermap with reversed colors
        if corr_df_out is not None:
            plot_omega_clustermap(corr_df_out, OUTPUT_DIR)

        # Plot Test Information Function and SEM
        plot_test_information_function(idata, K_final, OUTPUT_DIR)

        # Plot overall PPC distribution
        try:
            az.plot_ppc(idata, num_pp_samples=100)
            plt.savefig(os.path.join(OUTPUT_DIR, 'ppc_plot.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Skipped Posterior Predictive Check plot: {e}")

        # Plot MCMC traces (Caterpillar plots)
        try:
            az.plot_trace(idata, var_names=['mu', 'sigma', 'a'])
            plt.savefig(os.path.join(OUTPUT_DIR, 'trace_plots.png'), bbox_inches='tight')
            plt.close()
            print("Saved MCMC parameter trace plots to trace_plots.png")
        except Exception as e:
            print(f"Skipped Trace Plot generation: {e}")

    # Optional: Compare Estimated Parameters to True Simulation Parameters
    if RUN_SIMULATION and true_params is not None:
        print("\n--- Parameter Recovery Report (Ground Truth Verification) ---")
        # mu comparison
        est_mu = posterior_means["mu"].values.flatten()
        true_mu_vals = true_params['true_mu']
        mu_mae = np.mean(np.abs(est_mu - true_mu_vals))
        print(f"  mu (Global Means) Recovery MAE: {mu_mae:.4f}")

        # sigma comparison
        est_sigma = sigma_means_val
        true_sigma_vals = true_params['true_sigma']
        sigma_mae = np.mean(np.abs(est_sigma - true_sigma_vals))
        print(f"  sigma (Global SDs) Recovery MAE: {sigma_mae:.4f}")

        # item discrimination (a) comparison
        est_a = posterior_means["a"].values
        true_a_vals = np.array([true_params['true_a'][q] for q in questions_to_use])
        a_mae = np.mean(np.abs(est_a - true_a_vals))
        print(f"  a (Item Discrimination) Recovery MAE: {a_mae:.4f}")
        print("Note: Parameters successfully recovered from generated ground truth data.")

    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')

    # Print a clean report of any warnings detected during Section 5 diagnostics
    if diagnostic_warnings:
        print("\n--- Diagnostic Warning Report ---")
        for warning in diagnostic_warnings:
            print(f"  * {warning}")

        if any("CRITICAL" in w for w in diagnostic_warnings):
            print("\n*** ACTION REQUIRED: Critical issues detected. Estimates may be unstable. ***")
        else:
            print("\nNote: Standard warnings detected. Review and adjust parameters as necessary.")
    else:
        print("\nNo diagnostics issues flagged. Model convergence appears stable.")

    print("\n--- Analysis Complete ---")
    print(f"Results, diagnostics, and plots saved in: {OUTPUT_DIR}")
    print(f'{(datetime.datetime.now() - T_START).total_seconds()} total seconds elapsed')


if __name__ == '__main__':
    main()
