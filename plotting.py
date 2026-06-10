# -*- coding: utf-8 -*-
"""
plotting.py

Handles rendering of all diagnostic, parameter, team-capability,
and item-level (CRF, IIF, TIF) visualizations with robust HDI bounds.
"""

import os
import re
from typing import Dict
from typing import List
from typing import Optional

import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
import seaborn as sns
from scipy.stats import gaussian_kde

import config

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def filename_escape(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def plot_reorg_slope_chart_standardized_internal(
        theta_df: pd.DataFrame,
        category_name: str,
        years: List[int],
        mu_map: Dict[str, float],
        sigma_map: Dict[str, float],
        parent_to_child_map: Dict[str, List[str]],
        target_dept: Optional[str] = None
) -> None:
    """
    Renders standardized lineage evolution slope charts mapping reorg splits and merges.
    """
    df_plot = theta_df.copy()
    if target_dept is not None:
        df_plot = df_plot[df_plot["dept"] == target_dept]

    pop_mean = mu_map.get(category_name)
    pop_sd = sigma_map.get(category_name)

    if pop_mean is None or pop_sd is None or pop_sd <= 1e-6 or df_plot.empty:
        return

    raw_col = f"theta_{category_name}_z"
    if raw_col not in df_plot.columns:
        return

    available_years = sorted([y for y in years if y in df_plot[config.YEAR_COL].unique()])
    if len(available_years) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    texts = []
    plotted_coords = {y: {} for y in available_years}
    team_year_counts = df_plot.groupby(config.ID_VAR)[config.YEAR_COL].nunique().to_dict()

    latest_year = available_years[-1]
    df_latest = df_plot[df_plot[config.YEAR_COL] == latest_year].set_index(config.ID_VAR)
    sorted_latest_teams = df_latest[raw_col].sort_values(ascending=False).index.tolist()

    cmap_registry = plt.colormaps["Spectral"].resampled(max(1, len(sorted_latest_teams)))
    color_map_y_last = {team: cmap_registry(i) for i, team in enumerate(sorted_latest_teams)}

    for idx, yr in enumerate(available_years):
        df_yr = df_plot[df_plot[config.YEAR_COL] == yr].set_index(config.ID_VAR)
        for team_id, row in df_yr.iterrows():
            val_z = row[raw_col]
            if pd.notna(val_z):
                plotted_coords[yr][team_id] = val_z
                align = "right" if idx == 0 else "left"
                offset = -0.05 if idx == 0 else 0.05

                if team_year_counts.get(team_id, 0) == 1:
                    ax.plot(yr, val_z, "x", color="gray", markersize=8, mew=2)
                    texts.append(ax.text(yr + offset, val_z, f"{team_id} (New)", ha=align, va="center", fontsize=8,
                                         color="gray"))
                else:
                    pt_color = color_map_y_last.get(team_id, "gray") if yr == latest_year else "gray"
                    ax.plot(yr, val_z, "o", color=pt_color, markersize=6)
                    texts.append(ax.text(yr + offset, val_z, team_id, ha=align, va="center", fontsize=8))

    # Connect nodes backward tracing lineage
    for latest_team in sorted_latest_teams:
        team_color = color_map_y_last[latest_team]
        parents = [parent for parent, children in parent_to_child_map.items() if latest_team in children]
        if not parents:
            parents = [latest_team]

        for p_idx, p in enumerate(parents):
            y_to = available_years[-1]
            y_from = available_years[-2]
            if p in plotted_coords[y_from] and latest_team in plotted_coords[y_to]:
                line_style = "-" if p_idx == 0 else "--"
                ax.plot([y_from, y_to], [plotted_coords[y_from][p], plotted_coords[y_to][latest_team]],
                        linestyle=line_style, lw=2.5, color=team_color, alpha=0.7)

                if len(available_years) > 2:
                    y_prev = available_years[-3]
                    if p in plotted_coords[y_prev]:
                        ax.plot([y_prev, y_from], [plotted_coords[y_prev][p], plotted_coords[y_from][p]],
                                linestyle=line_style, lw=1.5, color=team_color, alpha=0.7)

    if HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts, ax=ax, force_points=(0.2, 0.3),
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0", color="gray", lw=0.5, shrinkA=3, shrinkB=3)
        )

    title_suffix = f" - {target_dept}" if target_dept else ""
    years_str = " - ".join(map(str, available_years))
    ax.set_title(f"Standardized Capability Evolution: {category_name} ({years_str}){title_suffix}", fontsize=14,
                 weight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Capability Z-Score (Std. Devs from Mean)", fontsize=12)
    ax.set_xticks(available_years)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.set_xlim(available_years[0] - 0.4, available_years[-1] + 0.4)

    suffix = f"_{filename_escape(target_dept)}" if target_dept else ""
    filepath = os.path.join(config.OUTPUT_DIR, f"std_reorg_slope_{filename_escape(category_name)}{suffix}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_omega_clustermap(corr_df: pd.DataFrame) -> None:
    """
    Renders correlation matrices as hierarchical clustermaps.
    """
    if corr_df is None or corr_df.empty:
        return
    n_cats = corr_df.shape[0]
    figsize = (max(8, n_cats * 0.9), max(7, n_cats * 0.8))
    try:
        cluster_grid = sns.clustermap(
            corr_df, method="average", metric="euclidean", cmap="coolwarm_r",
            vmin=-1, vmax=1, center=0, annot=(n_cats <= 12), fmt=".2f",
            linewidths=.5, linecolor="lightgray", figsize=figsize
        )
        cluster_grid.figure.suptitle("Capability Correlations (Mean Omega)", y=1.02)
        plt.savefig(os.path.join(config.OUTPUT_DIR, "omega_clustermap.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [ERROR] Heatmap clustering failed: {e}")


def plot_likert_response_distributions(
        df_raw: pd.DataFrame,
        theta_df: pd.DataFrame,
        category_mapping: Dict[str, List[str]],
        target_year: int,
        target_dept: Optional[str] = None
) -> None:
    """
    Renders horizontal stacked response distributions sorted by team latent capabilities.
    Applies demographic filters to drop non-engineering departments from designated domains.
    """
    df_overall = theta_df[theta_df[config.YEAR_COL] == target_year]
    df_responses = df_raw[df_raw[config.YEAR_COL] == target_year]

    if target_dept:
        df_overall = df_overall[df_overall["dept"] == target_dept]
        df_responses = df_responses[df_responses["dept"] == target_dept]

    if df_overall.empty or df_responses.empty:
        return

    cmap = plt.colormaps["vlag"].resampled(len(config.RESPONSE_OPTIONS))
    colors = [cmap(r - 1) for r in config.RESPONSE_OPTIONS[::-1]]

    for category, questions in category_mapping.items():
        # Clean & melt data
        df_long_cat = df_responses.melt(
            id_vars=[config.ID_VAR, "dept"],
            value_vars=questions,
            var_name="question",
            value_name="response"
        ).dropna()

        if df_long_cat.empty:
            continue

        df_long_cat["response"] = pd.to_numeric(df_long_cat["response"], errors="coerce")
        df_long_cat = df_long_cat[df_long_cat["response"].isin(config.RESPONSE_OPTIONS)]
        df_long_cat["response"] = df_long_cat["response"].astype(int)

        # Drop non-engineering groups from designated categories
        if category in config.ENGG_ONLY_CATEGORIES:
            df_long_cat = df_long_cat[~df_long_cat["dept"].isin(config.NON_ENGG_TEAMS)]

        if df_long_cat.empty:
            continue

        counts = df_long_cat.groupby([config.ID_VAR, "response"]).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        theta_col = f"theta_{category}_z"
        if theta_col in df_overall.columns:
            ordered_teams = df_overall.sort_values(by=theta_col, ascending=True)[config.ID_VAR].tolist()
        else:
            ordered_teams = percentages.index.tolist()

        final_order = [t for t in ordered_teams if t in percentages.index]
        if not final_order:
            continue

        plot_df = percentages.reindex(index=final_order, columns=config.RESPONSE_OPTIONS[::-1], fill_value=0)

        fig_height = max(5, len(plot_df) * 0.6)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        plot_df.plot(kind="barh", stacked=True, color=colors, edgecolor="white", linewidth=0.5, ax=ax)

        title_suffix = f" - {target_dept}" if target_dept else ""
        ax.set_title(f"{category} ({target_year}){title_suffix}\nTeam Response Distribution (Sorted by Rank)\n",
                     fontsize=13, weight="bold")
        ax.set_xlabel("Percentage of Responses", fontsize=11)
        ax.set_ylabel("Team (Ranked bottom to top)", fontsize=11)
        ax.set_xlim([0, 100])
        ax.legend(config.RESPONSE_OPTIONS[::-1], title="Response", loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=len(config.RESPONSE_OPTIONS), frameon=False)

        for idx, (team, row) in enumerate(plot_df.iterrows()):
            start_pos = 0.0
            for j, val in enumerate(row):
                if val > 3.5:
                    text_color = "white" if j in (0, len(config.RESPONSE_OPTIONS) - 1) else "black"
                    ax.text(start_pos + val / 2, idx, f"{val:.0f}%", va="center", ha="center", color=text_color,
                            fontsize=9)
                start_pos += val

        suffix = f"_{filename_escape(target_dept)}" if target_dept else ""
        filepath = os.path.join(config.OUTPUT_DIR, f"likert_{filename_escape(category)}{suffix}.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)


def plot_team_ridge_plots(
        idata: az.InferenceData,
        theta_df: pd.DataFrame,
        category_name: str,
        target_dept: Optional[str] = None
) -> None:
    """
    Renders posterior densities for team capabilities styled as stacked ridge curves.
    Standardized post-hoc using stable global posterior means to prevent flat curves
    caused by near-zero sigma division in individual MCMC draws.
    """
    df_plot = theta_df.copy()
    if target_dept is not None:
        df_plot = df_plot[df_plot["dept"] == target_dept]

    if df_plot.empty:
        return

    cat_names = list(config.CATEGORY_MAPPING_INITIAL.keys())
    if category_name not in cat_names:
        return
    cat_idx = cat_names.index(category_name)

    # 1. Extract raw posterior samples of theta: shape (samples, group, category)
    theta_samples = idata.posterior["theta"].stack(samples=["chain", "draw"]).values
    theta_samples = theta_samples.transpose(2, 0, 1)

    # 2. Extract stable global posterior means for mu and sigma to safely scale post-hoc
    mu_mean = idata.posterior["mu"].mean(dim=["chain", "draw"]).values.flatten()[cat_idx]
    sigma_mean = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values.flatten()[cat_idx]

    unique_teams = sorted(df_plot[config.ID_VAR].unique())
    unique_years = sorted(df_plot[config.YEAR_COL].unique())
    num_teams = len(unique_teams)

    fig, axes = plt.subplots(num_teams, 1, figsize=(14, 2.8 * num_teams), sharex=True)
    if num_teams == 1:
        axes = [axes]

    palette = sns.color_palette("muted", num_teams)
    x_vals = np.linspace(-3.5, 3.5, 300)

    for i, team in enumerate(unique_teams):
        ax = axes[i]
        team_color = palette[i % len(palette)]
        ax.axhline(0, color="#1f4e78", linestyle=":", lw=1.5, zorder=0)

        for year in unique_years:
            grp_rows = df_plot[(df_plot[config.ID_VAR] == team) & (df_plot[config.YEAR_COL] == year)]
            if grp_rows.empty:
                continue
            grp_idx = grp_rows["group_idx"].values[0]

            # Get raw theta samples for this group and category
            raw_samples = theta_samples[:, grp_idx, cat_idx]

            # Post-hoc scale to standardized Z-scores safely
            samples_z = (raw_samples - mu_mean) / sigma_mean

            # Generate standard-deviation scaled density coordinates
            kde = gaussian_kde(samples_z)
            y_vals = kde(x_vals)

            if year == 2023:
                # 2023: No fill, colored outline
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
            elif year == 2025:
                # 2025: Line hatched fill, colored outline
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
                ax.fill_between(x_vals, y_vals, color="none", edgecolor=team_color, hatch="////", lw=0, zorder=2)
            elif year == 2026:
                # 2026: Opaque solid fill
                ax.fill_between(x_vals, y_vals, color=team_color, alpha=0.85, zorder=1)

        # Draw red vertical reference indicator at the latest year's post-hoc standardized posterior mean
        latest_year = unique_years[-1]
        latest_rows = df_plot[(df_plot[config.ID_VAR] == team) & (df_plot[config.YEAR_COL] == latest_year)]
        if not latest_rows.empty:
            latest_grp_idx = latest_rows["group_idx"].values[0]
            latest_raw_mean = theta_samples[:, latest_grp_idx, cat_idx].mean()
            latest_z_mean = (latest_raw_mean - mu_mean) / sigma_mean
            ax.axvline(latest_z_mean, color="red", linestyle="-", lw=2, zorder=5)

        ax.set_ylabel(team, rotation=0, ha="right", va="center", fontsize=12, weight="bold", labelpad=15)
        ax.yaxis.set_ticks([])
        ax.set_ylim(bottom=0)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    for ax in axes:
        ax.axvline(0, color="#1f4e78", linestyle="--", lw=1, zorder=1)

    axes[0].text(0, axes[0].get_ylim()[1] * 1.15, "mean", ha="center", va="bottom", style="italic", fontsize=12)

    # Clean Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="black", label="2023"),
        mpatches.Patch(facecolor="none", edgecolor="black", hatch="////", label="2025"),
        mpatches.Patch(facecolor="gray", alpha=0.85, label="2026")
    ]
    fig.legend(handles=legend_elements,
               loc="upper right",
               bbox_to_anchor=(0.95, 0.98),
               ncol=3,
               frameon=False,
               fontsize=11)

    plt.xlabel("Capability Score (Theta Standard Deviations)", fontsize=12, labelpad=15)
    plt.xlim(-3.5, 3.5)
    plt.tight_layout()

    suffix = f"_{filename_escape(target_dept)}" if target_dept else ""
    filepath = os.path.join(config.OUTPUT_DIR, f"ridge_plots_{filename_escape(category_name)}{suffix}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

def plot_category_response_functions(question_id: str, idata: az.InferenceData,
                                     question_idx_map: Dict[str, int]) -> None:
    """
    Renders the Category Response Functions P(Y=c|theta) showing probability transitions.
    """
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]

    a_samples = idata.posterior["a"].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = \
        idata.posterior["cutpoints"].values.reshape(-1, len(question_idx_map), config.N_CATEGORIES_RESPONSE - 1)[
            :, q_idx, :]

    thin_idx = np.linspace(0, len(a_samples) - 1, 100, dtype=int)
    a_samples = a_samples[thin_idx, None, None]
    cut_samples = cut_samples[thin_idx, None, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)
    theta_grid = theta_vals[None, :, None]

    eta = a_samples * theta_grid
    cum_prob = sps.expit(cut_samples - eta)

    probs_all = np.zeros((100, 100, config.N_CATEGORIES_RESPONSE))
    probs_all[:, :, 0] = cum_prob[:, :, 0]
    for c in range(1, config.N_CATEGORIES_RESPONSE - 1):
        probs_all[:, :, c] = cum_prob[:, :, c] - cum_prob[:, :, c - 1]
    probs_all[:, :, -1] = 1.0 - cum_prob[:, :, -1]

    probs = np.mean(probs_all, axis=0)

    _, ax = plt.subplots(figsize=(8, 5))
    colors = plt.colormaps["viridis"].resampled(config.N_CATEGORIES_RESPONSE)
    for c in range(config.N_CATEGORIES_RESPONSE):
        ax.plot(theta_vals, probs[:, c], color=colors(c), lw=2, label=f"Category {c + 1}")
    ax.set_xlabel("Latent Trait (Theta)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Category Response Functions: {question_id}")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"crf_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_predicted_vs_empirical_dist(
        question_id: str,
        idata: az.InferenceData,
        df_long: pd.DataFrame,
        question_idx_map: Dict[str, int],
        question_categories: Dict[str, int],
        ci_level: float = 0.95
) -> None:
    """
    Plots predictive check proportions against empirical frequencies with posterior HDIs.
    """
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]
    obs_indices = np.where(df_long["question_idx"] == q_idx)[0]
    if len(obs_indices) == 0:
        return

    empirical_responses = df_long.iloc[obs_indices]["response"].values
    groups_for_q = df_long.iloc[obs_indices]["group_idx"].values
    q_dim = question_categories[question_id]

    # Reshape posterior parameters
    a_samples = idata.posterior["a"].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = \
        idata.posterior["cutpoints"].values.reshape(-1, len(question_idx_map), config.N_CATEGORIES_RESPONSE - 1)[
            :, q_idx, :]
    theta_samples = idata.posterior["theta"].values.reshape(-1, len(df_long["group_idx"].unique()),
                                                            len(config.CATEGORY_MAPPING_INITIAL))

    # Thin to 200 samples to optimize performance while preserving shape stability
    sample_indices = np.linspace(0, len(a_samples) - 1, 200, dtype=int)
    predicted_proportions = np.zeros((len(sample_indices), config.N_CATEGORIES_RESPONSE))

    for idx, s_idx in enumerate(sample_indices):
        a_s = a_samples[s_idx]
        cuts_s = cut_samples[s_idx]
        theta_s = theta_samples[s_idx, groups_for_q, q_dim]

        # Calculate GRM probabilities vectorized
        eta = a_s * theta_s[:, None]
        cum_prob = sps.expit(cuts_s - eta)
        probs = np.zeros((len(theta_s), config.N_CATEGORIES_RESPONSE))
        probs[:, 0] = cum_prob[:, 0]
        for c in range(1, config.N_CATEGORIES_RESPONSE - 1):
            probs[:, c] = cum_prob[:, c] - cum_prob[:, c - 1]
        probs[:, -1] = 1.0 - cum_prob[:, -1]
        predicted_proportions[idx, :] = np.mean(probs, axis=0)

    pred_mean = np.mean(predicted_proportions, axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    pred_lower = np.percentile(predicted_proportions, lower_perc, axis=0)
    pred_upper = np.percentile(predicted_proportions, upper_perc, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x_ticks = np.array(config.RESPONSE_OPTIONS)

    # Render empirical counts
    counts, _ = np.histogram(empirical_responses, bins=np.arange(0.5, config.N_CATEGORIES_RESPONSE + 1.5, 1),
                             density=True)
    ax.bar(x_ticks, counts, width=0.6, color="lightgray", alpha=0.7, edgecolor="black", label="Empirical Frequencies")

    # Overlay model predictive intervals
    ax.plot(x_ticks, pred_mean, marker="o", linestyle="-", color="dodgerblue", label="Model Predicted Mean", zorder=10)
    ax.fill_between(x_ticks, pred_lower, pred_upper, color="dodgerblue", alpha=0.25,
                    label=f"Predicted {ci_level * 100:.0f}% HDI")

    ax.set_xlabel("Response Category", fontsize=11)
    ax.set_ylabel("Density / Proportion", fontsize=11)
    ax.set_xticks(x_ticks)
    ax.set_title(f"Item Fit Check: Question {question_id}", fontsize=12, weight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.savefig(os.path.join(config.OUTPUT_DIR, f"item_fit_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_item_information_function(
        question_id: str,
        idata: az.InferenceData,
        question_idx_map: Dict[str, int],
        ci_level: float = 0.95
) -> None:
    """
    Renders approximate Item Information Function (IIF) curves incorporating shaded HDI bounds.
    """
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]

    a_samples = idata.posterior["a"].values.reshape(-1, len(question_idx_map))[:, q_idx]
    cut_samples = \
        idata.posterior["cutpoints"].values.reshape(-1, len(question_idx_map), config.N_CATEGORIES_RESPONSE - 1)[
            :, q_idx, :]

    # Thin samples to speed up mathematical integration
    sample_indices = np.linspace(0, len(a_samples) - 1, 200, dtype=int)
    theta_vals = np.linspace(-3.5, 3.5, 100)
    info_samples = np.zeros((len(sample_indices), len(theta_vals)))

    for idx, s_idx in enumerate(sample_indices):
        a_s = a_samples[s_idx]
        cuts_s = cut_samples[s_idx]
        for j, th in enumerate(theta_vals):
            # Calculate information across ordinal categories
            eta = a_s * th
            info_val = 0.0
            for c in range(config.N_CATEGORIES_RESPONSE - 1):
                logit = cuts_s[c] - eta
                P_star = sps.expit(logit)
                info_val += P_star * (1.0 - P_star)
            info_samples[idx, j] = info_val * (a_s ** 2)

    info_mean = np.mean(info_samples, axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    info_lower = np.percentile(info_samples, lower_perc, axis=0)
    info_upper = np.percentile(info_samples, upper_perc, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_vals, info_mean, color="navy", lw=2, label="Mean Item Information")
    ax.fill_between(theta_vals, info_lower, info_upper, color="skyblue", alpha=0.3, label=f"{ci_level * 100:.0f}% HDI")
    ax.set_xlabel("Latent Trait (Theta)")
    ax.set_ylabel("Item Information")
    ax.set_title(f"Item Information Curve: {question_id}", fontsize=12, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.savefig(os.path.join(config.OUTPUT_DIR, f"iif_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_test_information_function(
        idata: az.InferenceData,
        K: int,
        ci_level: float = 0.95
) -> None:
    """
    Renders Test Information Functions (TIF) and Standard Error of Measurement (SEM)
    curves with fully shaded credible uncertainty intervals.
    """
    a_samples = idata.posterior["a"].values.reshape(-1, K)
    cut_samples = idata.posterior["cutpoints"].values.reshape(-1, K, config.N_CATEGORIES_RESPONSE - 1)

    sample_indices = np.linspace(0, len(a_samples) - 1, 150, dtype=int)
    theta_vals = np.linspace(-3.5, 3.5, 100)
    tif_samples = np.zeros((len(sample_indices), len(theta_vals)))

    for idx, s_idx in enumerate(sample_indices):
        a_s = a_samples[s_idx]
        cut_s = cut_samples[s_idx]
        for j, th in enumerate(theta_vals):
            total_info = 0.0
            for k in range(K):
                a_k = a_s[k]
                cuts_k = cut_s[k, :]
                eta = a_k * th
                for c in range(config.N_CATEGORIES_RESPONSE - 1):
                    P_star = sps.expit(cuts_k[c] - eta)
                    total_info += P_star * (1.0 - P_star) * (a_k ** 2)
            tif_samples[idx, j] = total_info

    # Calculate TIF parameters
    tif_mean = np.mean(tif_samples, axis=0)
    lower_perc = (1.0 - ci_level) / 2.0 * 100
    upper_perc = (1.0 + ci_level) / 2.0 * 100
    tif_lower = np.percentile(tif_samples, lower_perc, axis=0)
    tif_upper = np.percentile(tif_samples, upper_perc, axis=0)

    # Calculate SEM parameters (SEM = 1/sqrt(TIF))
    sem_samples = 1.0 / np.sqrt(np.maximum(tif_samples, 1e-6))
    sem_mean = np.mean(sem_samples, axis=0)
    sem_lower = np.percentile(sem_samples, lower_perc, axis=0)
    sem_upper = np.percentile(sem_samples, upper_perc, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_tif = "darkblue"
    ax1.plot(theta_vals, tif_mean, color=color_tif, lw=2.5, label="Test Information (TIF)")
    ax1.fill_between(theta_vals, tif_lower, tif_upper, color=color_tif, alpha=0.15,
                     label=f"TIF {ci_level * 100:.0f}% HDI")
    ax1.set_xlabel("Latent Trait (Theta)", fontsize=11)
    ax1.set_ylabel("Total Test Information", color=color_tif, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_tif)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    color_sem = "firebrick"
    ax2.plot(theta_vals, sem_mean, color=color_sem, lw=2, ls="--", label="Standard Error (SEM)")
    ax2.fill_between(theta_vals, sem_lower, sem_upper, color=color_sem, alpha=0.12,
                     label=f"SEM {ci_level * 100:.0f}% HDI")
    ax2.set_ylabel("Standard Error of Measurement (SEM)", color=color_sem, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_sem)
    ax2.legend(loc="upper right")

    plt.title("Test Information Function & Measurement Error bounds", fontsize=12, weight="bold")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "tif_sem_plot.png"), bbox_inches="tight")
    plt.close()
