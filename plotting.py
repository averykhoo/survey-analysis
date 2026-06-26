# -*- coding: utf-8 -*-
"""
plotting.py

Handles rendering of all diagnostic, parameter, team-capability,
and item-level (CRF, IIF, TIF) visualizations with robust HDI bounds.
"""

import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import arviz as az
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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
    """Escapes strings for filesystem compliance."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def get_team_lineage_label(team_id: str, year: int, lineage_map: Dict[int, Dict[str, List[str]]]) -> str:
    """
    Traces parentage backward from 2026, compiling inline descriptive labels.
    """
    if year == 2026:
        parents = lineage_map.get(2026, {}).get(team_id, [])
        if parents and parents != [team_id]:
            parent_str = ", ".join(parents)
            grandparents = []
            for p in parents:
                gps = lineage_map.get(2025, {}).get(p, [])
                grandparents.extend(gps)
            grandparents = sorted(list(set(grandparents)))
            if grandparents and grandparents != parents and grandparents != [team_id]:
                gp_str = ", ".join(grandparents)
                return f"{team_id}\n(from 2025: {parent_str})\n(from 2023: {gp_str})"
            return f"{team_id}\n(from 2025: {parent_str})"
    return team_id


def plot_slope_chart_hierarchical(
        df_estimates: pd.DataFrame,
        name: str,
        level: str,
        years: List[int],
        lineage_map: Dict[int, Dict[str, List[str]]],
        output_dir: str,
        target_dept: Optional[str] = None
) -> None:
    """
    Renders standardized lineage slope charts.

    Marker Rules:
      - 1 appearance total: 'x', gray
      - Multiple appearances, hits final year: colored 'o', solid lines
      - Multiple appearances, dies before final year: gray 'o', dashed lines

    Color Rule:
      - Rainbow spectrum (Spectral) driven by value in their FINAL year of appearance.
      - Red maps to lowest score, Blue/Purple to highest score.
    """
    df_plot = df_estimates.copy()
    if target_dept is not None:
        df_plot = df_plot[df_plot["dept"] == target_dept]

    val_col = f"{level}_{name}"
    if val_col not in df_plot.columns or df_plot.empty:
        return

    available_years = sorted([y for y in years if y in df_plot[config.YEAR_COL].unique()])
    if len(available_years) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 8))
    texts = []

    coords = {y: {} for y in available_years}
    for yr in available_years:
        df_yr = df_plot[df_plot[config.YEAR_COL] == yr].set_index(config.ID_VAR)
        for team_id, row in df_yr.iterrows():
            if pd.notna(row[val_col]):
                coords[yr][team_id] = row[val_col]

    all_teams = set()
    for y_dict in coords.values():
        all_teams.update(y_dict.keys())

    team_meta = {}
    for team in all_teams:
        active_years = [y for y in available_years if team in coords[y]]
        final_yr = max(active_years)
        team_meta[team] = {
            "active_years": active_years,
            "final_yr":     final_yr,
            "final_score":  coords[final_yr][team]
        }

    final_year_overall = available_years[-1]

    # Map terminal scores to a Spectral Colormap (Lowest -> Red, Highest -> Blue)
    scores = [m["final_score"] for m in team_meta.values()]
    vmin, vmax = min(scores), max(scores)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap("Spectral")

    team_colors = {team: cmap(norm(m["final_score"])) for team, m in team_meta.items()}

    # Plot Nodes
    for yr in available_years:
        for team, val in coords[yr].items():
            meta = team_meta[team]
            offset = -0.05 if yr == available_years[0] else 0.05
            align = "right" if yr == available_years[0] else "left"

            display_label = get_team_lineage_label(team, yr, lineage_map)

            if len(meta["active_years"]) == 1:
                ax.plot(yr, val, "x", color="gray", markersize=7, mew=2)
                texts.append(
                    ax.text(yr + offset, val, f"{display_label} (1-Yr)", ha=align, va="center", fontsize=8, color="gray"))
            else:
                if meta["final_yr"] == final_year_overall:
                    ax.plot(yr, val, "o", color=team_colors[team], markersize=6)
                    texts.append(ax.text(yr + offset, val, display_label, ha=align, va="center", fontsize=8))
                else:
                    ax.plot(yr, val, "o", color="gray", markersize=6)
                    texts.append(ax.text(yr + offset, val, display_label, ha=align, va="center", fontsize=8, color="gray"))

    # Plot Parent-Child Connection Lines
    for idx in range(len(available_years) - 1, 0, -1):
        y_to = available_years[idx]
        y_from = available_years[idx - 1]

        yr_mapping = lineage_map.get(y_to, {})
        for child_team, parents in yr_mapping.items():
            if child_team in coords[y_to]:
                val_to = coords[y_to][child_team]

                if team_meta[child_team]["final_yr"] == final_year_overall:
                    line_color = team_colors[child_team]
                    base_style = "-"
                else:
                    line_color = "gray"
                    base_style = "--"

                for p_idx, parent_team in enumerate(parents):
                    if parent_team in coords[y_from]:
                        val_from = coords[y_from][parent_team]
                        style = base_style if p_idx == 0 else ":"
                        ax.plot([y_from, y_to], [val_from, val_to], linestyle=style, lw=2.5, color=line_color,
                                alpha=0.75)

    if HAS_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax, force_points=(0.2, 0.3), iterations=30, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    title_suffix = f" ({target_dept})" if target_dept else ""
    ax.set_title(f"Standardized {level.capitalize()} Evolution: {name}{title_suffix}", fontsize=13, weight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Standardized Score (St. Devs from Mean)", fontsize=11)
    ax.set_xticks(available_years)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.set_xlim(available_years[0] - 0.4, available_years[-1] + 0.4)

    suffix = f"_{filename_escape(target_dept)}" if target_dept else ""
    filename = f"{level}_{filename_escape(name)}{suffix}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_omega_clustermap(corr_df: pd.DataFrame, output_dir: str) -> None:
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
        cluster_grid.figure.suptitle("Hierarchical Clustermap of Dimensions (Mean Omega)", y=1.02, weight="bold")
        plt.savefig(os.path.join(output_dir, "omega_clustermap.png"), bbox_inches="tight")
        plt.close()
        print("  Successfully plotted Omega hierarchical clustermap.")
    except Exception as e:
        print(f"  [SKIPPED] Heatmap clustering failed: {e}")


def plot_likert_response_distributions(
        df_raw: pd.DataFrame,
        df_estimates: pd.DataFrame,
        category: str,
        questions: List[str],
        target_year: int,
        output_dir: str,
        target_dept: Optional[str] = None
) -> None:
    """
    Renders horizontal stacked response distributions sorted by team latent capabilities.
    """
    df_overall = df_estimates[df_estimates[config.YEAR_COL] == target_year]
    df_responses = df_raw[df_raw[config.YEAR_COL] == target_year]

    if target_dept:
        df_overall = df_overall[df_overall["dept"] == target_dept]
        df_responses = df_responses[df_responses["dept"] == target_dept]

    if df_overall.empty or df_responses.empty:
        return

    cmap = plt.colormaps["vlag"].resampled(len(config.RESPONSE_OPTIONS))
    colors = [cmap(r - 1) for r in config.RESPONSE_OPTIONS[::-1]]

    # Clean & melt data
    df_long_cat = df_responses.melt(
        id_vars=[config.ID_VAR, "dept"],
        value_vars=questions,
        var_name="question",
        value_name="response"
    ).dropna()

    if df_long_cat.empty:
        return

    df_long_cat["response"] = pd.to_numeric(df_long_cat["response"]).astype(int)
    df_long_cat = df_long_cat[df_long_cat["response"].isin(config.RESPONSE_OPTIONS)]

    # Drop non-engineering groups from designated categories
    if category in config.ENGG_ONLY_CATEGORIES:
        df_long_cat = df_long_cat[~df_long_cat["dept"].isin(config.NON_ENGG_TEAMS)]

    if df_long_cat.empty:
        return

    counts = df_long_cat.groupby([config.ID_VAR, "response"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    theta_col = f"category_{category}"
    if theta_col in df_overall.columns:
        ordered_teams = df_overall.sort_values(by=theta_col, ascending=True)[config.ID_VAR].tolist()
    else:
        ordered_teams = percentages.index.tolist()

    final_order = [t for t in ordered_teams if t in percentages.index]
    if not final_order:
        return

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
    filepath = os.path.join(output_dir, f"likert_{filename_escape(category)}{suffix}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_ridge_plots_hierarchical(
        idata: az.InferenceData,
        df_estimates: pd.DataFrame,
        name: str,
        level: str,
        idx: int,
        draws_to_use: int,
        output_dir: str,
        target_dept: Optional[str] = None
) -> None:
    """
    Generates posterior capabilities density ridge plots with thinned draws to accelerate KDE.
    """
    df_plot = df_estimates.copy()
    if target_dept is not None:
        df_plot = df_plot[df_plot["dept"] == target_dept]

    if df_plot.empty:
        return

    param_key = "theta_sec" if level == "section" else "theta_cat"
    raw_samples = idata.posterior[param_key].stack(draws=["chain", "draw"]).values

    total_samples = raw_samples.shape[-1]
    step_size = max(1, total_samples // draws_to_use)
    samples = raw_samples[:, :, ::step_size]

    unique_teams = sorted(df_plot[config.ID_VAR].unique())
    unique_years = sorted(df_plot[config.YEAR_COL].unique())
    num_teams = len(unique_teams)

    fig, axes = plt.subplots(num_teams, 1, figsize=(13, 2.5 * num_teams), sharex=True)
    if num_teams == 1:
        axes = [axes]

    colors = plt.cm.get_cmap("tab10", num_teams)
    x_vals = np.linspace(-3.5, 3.5, 100)

    for i, team in enumerate(unique_teams):
        ax = axes[i]
        team_color = colors(i)
        ax.axhline(0, color="#1f4e78", linestyle=":", lw=1.5, zorder=0)

        for yr in unique_years:
            rows = df_plot[(df_plot[config.ID_VAR] == team) & (df_plot[config.YEAR_COL] == yr)]
            if rows.empty:
                continue
            grp_idx = rows["group_idx"].values[0]

            group_samples = samples[grp_idx, idx, :]
            kde = gaussian_kde(group_samples)
            y_vals = kde(x_vals)

            if yr == 2023:
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
            elif yr == 2025:
                ax.plot(x_vals, y_vals, color=team_color, lw=2, zorder=3)
                ax.fill_between(x_vals, y_vals, color="none", edgecolor=team_color, hatch="////", lw=0, zorder=2)
            elif yr == 2026:
                ax.fill_between(x_vals, y_vals, color=team_color, alpha=0.8, zorder=1)

        ax.set_ylabel(team, rotation=0, ha="right", va="center", fontsize=11, weight="bold", labelpad=15)
        ax.yaxis.set_ticks([])
        ax.set_ylim(bottom=0)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    for ax in axes:
        ax.axvline(0, color="#1f4e78", linestyle="--", lw=1, zorder=1)

    plt.xlabel(f"{level.capitalize()} Score (St. Devs from Mean)", fontsize=11, labelpad=12)
    plt.xlim(-3.5, 3.5)
    plt.tight_layout()

    suffix = f"_{filename_escape(target_dept)}" if target_dept else ""
    filename = f"ridge_{level}_{filename_escape(name)}{suffix}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_category_response_functions(
        question_id: str,
        idata: az.InferenceData,
        struct_maps: Dict[str, Any],
        output_dir: str
) -> None:
    """
    Vectorized rendering of Category Response Functions P(Y=c|theta).
    """
    question_idx_map = struct_maps["question_idx_map"]
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]
    K = len(question_idx_map)

    a_samples = idata.posterior["a"].values.reshape(-1, K)[:, q_idx]
    cut_samples = idata.posterior["cutpoints"].values.reshape(-1, K, config.N_CATEGORIES_RESPONSE - 1)[:, q_idx, :]

    thin_idx = np.linspace(0, len(a_samples) - 1, 150, dtype=int)
    a_samples = a_samples[thin_idx]
    cut_samples = cut_samples[thin_idx, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)

    eta = a_samples[:, None] * theta_vals[None, :]
    cum_prob = sps.expit(cut_samples[:, None, :] - eta[:, :, None])

    probs_all = np.zeros((150, 100, config.N_CATEGORIES_RESPONSE))
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
    ax.set_title(f"Category Response Functions: {question_id}", weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, f"crf_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_predicted_vs_empirical_dist(
        question_id: str,
        idata: az.InferenceData,
        df_long: pd.DataFrame,
        struct_maps: Dict[str, Any],
        output_dir: str,
        ci_level: float = 0.95
) -> None:
    """
    Plots predictive check proportions against empirical frequencies with posterior HDIs.
    """
    question_idx_map = struct_maps["question_idx_map"]
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]
    obs_indices = np.where(df_long["question_idx"] == q_idx)[0]
    if len(obs_indices) == 0:
        return

    empirical_responses = df_long.iloc[obs_indices]["response"].values
    groups_for_q = df_long.iloc[obs_indices]["group_idx"].values

    q_cat_idx = struct_maps["question_to_category"][q_idx]
    K = len(question_idx_map)
    J = len(struct_maps["group_idx_map"])
    C_cat = len(struct_maps["categories"])

    # Reshape posterior parameters
    a_samples = idata.posterior["a"].values.reshape(-1, K)[:, q_idx]
    cut_samples = idata.posterior["cutpoints"].values.reshape(-1, K, config.N_CATEGORIES_RESPONSE - 1)[:, q_idx, :]
    theta_cat_samples = idata.posterior["theta_cat"].values.reshape(-1, J, C_cat)

    # Thin to 200 samples to optimize performance while preserving shape stability
    sample_indices = np.linspace(0, len(a_samples) - 1, 150, dtype=int)
    predicted_proportions = np.zeros((len(sample_indices), config.N_CATEGORIES_RESPONSE))

    for idx, s_idx in enumerate(sample_indices):
        a_s = a_samples[s_idx]
        cuts_s = cut_samples[s_idx]
        theta_s = theta_cat_samples[s_idx, groups_for_q, q_cat_idx]

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

    plt.savefig(os.path.join(output_dir, f"item_fit_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_item_information_function(
        question_id: str,
        idata: az.InferenceData,
        struct_maps: Dict[str, Any],
        output_dir: str,
        ci_level: float = 0.95
) -> None:
    """
    Renders Item Information Function (IIF) curves incorporating shaded HDI bounds.
    """
    question_idx_map = struct_maps["question_idx_map"]
    if question_id not in question_idx_map:
        return
    q_idx = question_idx_map[question_id]
    K = len(question_idx_map)

    a_samples = idata.posterior["a"].values.reshape(-1, K)[:, q_idx]
    cut_samples = idata.posterior["cutpoints"].values.reshape(-1, K, config.N_CATEGORIES_RESPONSE - 1)[:, q_idx, :]

    thin_idx = np.linspace(0, len(a_samples) - 1, 150, dtype=int)
    a_samples = a_samples[thin_idx]
    cut_samples = cut_samples[thin_idx, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)

    eta = a_samples[:, None] * theta_vals[None, :]
    logit = cut_samples[:, None, :] - eta[:, :, None]
    P_star = sps.expit(logit)

    info_samples = np.sum(P_star * (1.0 - P_star), axis=2) * (a_samples[:, None] ** 2)

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

    plt.savefig(os.path.join(output_dir, f"iif_{filename_escape(question_id)}.png"), bbox_inches="tight")
    plt.close()


def plot_test_information_function(
        idata: az.InferenceData,
        K: int,
        output_dir: str,
        ci_level: float = 0.95
) -> None:
    """
    Renders Test Information Functions (TIF) and Standard Error of Measurement (SEM).
    """
    a_samples = idata.posterior["a"].values.reshape(-1, K)
    cut_samples = idata.posterior["cutpoints"].values.reshape(-1, K, config.N_CATEGORIES_RESPONSE - 1)

    thin_idx = np.linspace(0, len(a_samples) - 1, 100, dtype=int)
    a_samples = a_samples[thin_idx, :]
    cut_samples = cut_samples[thin_idx, :, :]

    theta_vals = np.linspace(-3.5, 3.5, 100)

    eta = a_samples[:, :, None] * theta_vals[None, None, :]
    logit = cut_samples[:, :, None, :] - eta[:, :, :, None]
    P_star = sps.expit(logit)

    info_per_question = np.sum(P_star * (1.0 - P_star), axis=3) * (a_samples[:, :, None] ** 2)
    tif_samples = np.sum(info_per_question, axis=1)

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
    plt.savefig(os.path.join(output_dir, "tif_sem_plot.png"), bbox_inches="tight")
    plt.close()
