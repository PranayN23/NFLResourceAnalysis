"""
Shared grade tier labels and blending of population age curves with a player's
recent year-over-year grade trajectory (same player, consecutive seasons).
"""
from __future__ import annotations

import statistics
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Four-tier scale (aligned across all FA agents)
TIER_ELITE_MIN = 80.0
TIER_GOOD_MIN = 74.0
TIER_STARTER_MIN = 62.0


def grade_to_tier_universal(grade: float) -> str:
    """Elite / Good / Starter / Rotation/backup — single source of truth for FA agents."""
    g = float(grade)
    if g >= TIER_ELITE_MIN:
        return "Elite"
    if g >= TIER_GOOD_MIN:
        return "Good"
    if g >= TIER_STARTER_MIN:
        return "Starter"
    return "Rotation/backup"


def player_recent_grade_yoy(history: Optional[pd.DataFrame], grade_col: str) -> Optional[float]:
    """
    Median of the player's last up to 3 consecutive-season grade deltas (curr - prev).
    Returns None if insufficient history.
    """
    if history is None or history.empty or grade_col not in history.columns:
        return None
    h = history.sort_values("Year").copy()
    h[grade_col] = pd.to_numeric(h[grade_col], errors="coerce")
    h["Year"] = pd.to_numeric(h["Year"], errors="coerce")
    h = h.dropna(subset=[grade_col, "Year"])
    if len(h) < 2:
        return None

    deltas: list[float] = []
    prev_yr: Optional[int] = None
    prev_g: Optional[float] = None
    for _, row in h.iterrows():
        yr = int(round(float(row["Year"])))
        g = float(row[grade_col])
        if np.isnan(g):
            continue
        if prev_yr is not None and prev_g is not None and yr == prev_yr + 1:
            deltas.append(g - prev_g)
        prev_yr, prev_g = yr, g

    if not deltas:
        return None
    tail = deltas[-3:]
    return float(statistics.median(tail))


def blend_age_curve_delta(
    base_delta: float,
    age_transition: int,
    player_yoy: Optional[float],
) -> float:
    """
    Mix population age curve with the player's recent YoY grade trend.
    Older ages lean slightly more on the population curve; strong personal
    trajectories still pull the blended delta (e.g. late-30s QB still ascending).
    """
    if player_yoy is None or (isinstance(player_yoy, float) and np.isnan(player_yoy)):
        return float(base_delta)

    age_transition = int(age_transition)
    if age_transition < 29:
        w_player = 0.40
    elif age_transition < 33:
        w_player = 0.30
    elif age_transition < 36:
        w_player = 0.22
    else:
        w_player = 0.16

    blended = (1.0 - w_player) * float(base_delta) + w_player * float(player_yoy)
    blended = max(-12.0, min(12.0, blended))
    return round(blended, 3)


def apply_yearly_grade_step(
    grade: float,
    age_transition: int,
    player_yoy: Optional[float],
    annual_delta_fn: Callable[[int], float],
) -> float:
    """One contract year: population delta via *annual_delta_fn*, then blend with player YoY."""
    base = float(annual_delta_fn(age_transition))
    d = blend_age_curve_delta(base, age_transition, player_yoy)
    return max(45.0, min(99.0, float(grade) + d))
