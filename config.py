# -*- coding: utf-8 -*-
"""
config.py

Holds all global constants, directory targets, mapping tables,
and statistical thresholds for the DORA survey capability analysis.
"""

import os
from typing import Dict
from typing import List

# --- General Run Settings ---
RANDOM_SEED: int = 42
OUTPUT_DIR: str = "dora_analysis_output_pymc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MCMC Sampler Configurations ---
ITER_WARMUP: int = 1000
ITER_SAMPLING: int = 1500
CHAINS: int = 4
TARGET_ACCEPT: float = 0.95  # Raised to 0.95 to resolve divergent transitions

# --- Statistical Diagnostic Thresholds ---
RHAT_THRESHOLD: float = 1.05
NEFF_RATIO_THRESHOLD: float = 0.1  # Warning flag if ESS/total draws < 10%
PARETO_K_THRESHOLD: float = 0.7  # Flag for highly influential/poorly fit observations
LOW_DISCRIMINATION_THRESHOLD: float = 0.3
HIGH_CORR_THRESHOLD: float = 0.85
MIN_RESPONSES_PER_GROUP: int = 5

# --- Demographics & Survey Scales ---
RESPONSE_OPTIONS: List[int] = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE: int = len(RESPONSE_OPTIONS)
YEAR_COL: str = "year"
ID_VAR: str = "team_id"

# --- Category / Dimension Mapping ---
# Maps specific question codes to latent DORA capabilities
CATEGORY_MAPPING_INITIAL: Dict[str, List[str]] = {
    "Deployment & Release":       ["q1", "q2", "q3"],
    "Monitoring & Observability": ["q4", "q5"],
    "Technical Practices":        ["q6", "q7", "q8"],
    "Team Collaboration":         ["q9", "q10"],
    "Process Efficiency":         ["q11"],
    "Learning & Development":     ["q12", "q13"],
    "System Reliability":         ["q14", "q15"],
    "Change Management":          ["q16"]
}

# Categories strictly restricted to engineering personnel or groups
ENGG_ONLY_CATEGORIES: List[str] = [
    "Deployment & Release",
    "Monitoring & Observability",
    "Technical Practices",
    "System Reliability"
]

# Non-engineering teams to filter out of engineering-only plots
NON_ENGG_TEAMS: List[str] = [
    "Product",
    "Marketing",
    "Operations"
]

# --- Reorganization Map (Y_Last to Y1 Parent Lineage) ---
REORG_MAPPING_Y_LAST_TO_Y1: Dict[str, List[str]] = {
    "Team_X": ["Team_A", "Team_B"],
    "Team_Y": ["Team_B", "Team_C"],
    "Team_Z": ["Team_A", "Team_C", "Team_D"],
    "Team_E": ["Team_E"],
    "Team_F": ["Team_F"],
    "Team_G": []
}
