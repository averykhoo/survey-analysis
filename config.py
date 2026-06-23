# -*- coding: utf-8 -*-
"""
config.py

Holds all global constants, directory targets, hierarchical mapping tables,
and statistical thresholds for the hierarchical DORA survey capability analysis.
"""

import os
from typing import Dict
from typing import List

# --- General Run Settings ---
RANDOM_SEED: int = 42
BASE_DIR: str = "dora_analysis_output_pymc"

# --- Compartmentalized Subdirectories ---
MODEL_DIR: str = os.path.join(BASE_DIR, "model")
DIAGNOSTICS_DIR: str = os.path.join(BASE_DIR, "diagnostics")
CSV_DIR: str = os.path.join(BASE_DIR, "estimates_csv")
PLOTS_DIR: str = os.path.join(BASE_DIR, "plots")

# Auto-generate folder structure on system load
for d in [MODEL_DIR, DIAGNOSTICS_DIR, CSV_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- MCMC Sampler Configurations ---
ITER_WARMUP: int = 1000
ITER_SAMPLING: int = 1500
CHAINS: int = 4 # 8 chains takes too much vram when running posterior checks
TARGET_ACCEPT: float = 0.95  # High target accept to prevent divergent transitions

# --- Statistical Diagnostic Thresholds ---
RHAT_THRESHOLD: float = 1.05
NEFF_RATIO_THRESHOLD: float = 0.1  # Warning flag if ESS/total draws < 10%
PARETO_K_THRESHOLD: float = 0.7  # For LOOIC diagnostic plotting [1]
LOW_DISCRIMINATION_THRESHOLD: float = 0.3
HIGH_CORR_THRESHOLD: float = 0.85  # Threshold for recommending merging of survey dimensions [1]
MIN_RESPONSES_PER_GROUP: int = 5

# --- Demographics & Survey Scales ---
RESPONSE_OPTIONS: List[int] = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE: int = len(RESPONSE_OPTIONS)
YEAR_COL: str = "year"
ID_VAR: str = "team_id"

# --------------------------------------------------------------------
# CHANGE ALL VALUES BELOW IN PROD!
# --------------------------------------------------------------------


# --- Hierarchical Survey Map (Sections -> Categories -> Questions) ---
SURVEY_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "Key Outcomes":           {
        "Software Delivery": ["q1", "q2"],
        "Org Performance":   ["q3", "q4"],
        "Team Performance":  ["q5"]  # 1-question category
    },
    "Technical Capabilities": {
        "Code Review":     ["q6"],  # 1-question category
        "Security":        ["q7", "q8"],
        "CI":              ["q9"],  # 1-question category
        "Version Control": ["q10", "q11"]
    },
    "AI and I":               {
        "AI Usage":    ["q12", "q13"],
        "Eng Culture": ["q14", "q15"]
    },
    "Wellbeing":              {
        "Flow":       ["q16"],  # 1-question category
        "Engagement": ["q17"]  # 1-question category
    }
}

# Categories strictly restricted to engineering personnel (filtered in demographics)
ENGG_ONLY_CATEGORIES: List[str] = [
    "Software Delivery",
    "Security",
    "CI",
    "Version Control"
]

# Non-engineering teams to filter out of engineering-only plots
NON_ENGG_TEAMS: List[str] = [
    "Product",
    "Marketing",
    "Operations"
]

# --- Year-by-Year Lineage Transition Map ---
REORG_LINEAGE_MAP: Dict[int, Dict[str, List[str]]] = {
    2026: {
        "Team_X": ["Team_A", "Team_B"],  # Team A and B merged to form X
        "Team_Y": ["Team_B", "Team_C"],  # Team B and C merged to form Y
        "Team_Z": ["Team_A", "Team_C", "Team_D"],
        "Team_E": ["Team_E"],  # Unchanged
        "Team_F": ["Team_F"],  # Unchanged
        "Team_G": []  # Completely new team in 2026
    },
    2025: {
        "Team_A": ["Team_A"],  # Unchanged between 2023 and 2025
        "Team_B": ["Team_B"],
        "Team_C": ["Team_C"],
        "Team_D": ["Team_D"],
        "Team_E": ["Team_E"],
        "Team_F": ["Team_F"]
    }
}
