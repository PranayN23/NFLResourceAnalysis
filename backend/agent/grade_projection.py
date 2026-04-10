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

    snap_candidates = (
        "passing_snaps",
        "dropbacks",
        "pass_block_snaps",
        "snap_counts_offense",
        "snap_counts_defense",
        "total_snaps",
        "routes",
        "targets",
        "attempts",
    )
    snap_col = next((c for c in snap_candidates if c in h.columns), None)
    if snap_col:
        h[snap_col] = pd.to_numeric(h[snap_col], errors="coerce").fillna(0.0)
    else:
        h["_snap_fallback"] = 1.0
        snap_col = "_snap_fallback"

    deltas: list[tuple[float, float]] = []
    prev_yr: Optional[int] = None
    prev_g: Optional[float] = None
    prev_snap: Optional[float] = None
    for _, row in h.iterrows():
        yr = int(round(float(row["Year"])))
        g = float(row[grade_col])
        s = float(row.get(snap_col, 0.0) or 0.0)
        if np.isnan(g):
            continue
        if prev_yr is not None and prev_g is not None and yr == prev_yr + 1:
            opp = max(1.0, min(float(prev_snap or 1.0), s))
            deltas.append((g - prev_g, opp))
        prev_yr, prev_g, prev_snap = yr, g, s

    if not deltas:
        return None
    tail = deltas[-3:]
    # Recency + opportunity weighting: down-weight tiny-sample deltas.
    recency = [0.24, 0.33, 0.43][-len(tail):]
    opp_baseline = 500.0 if snap_col in ("passing_snaps", "dropbacks") else 350.0
    weighted_num = 0.0
    weighted_den = 0.0
    for (delta, opp), rw in zip(tail, recency):
        opp_factor = max(0.30, min(1.0, np.sqrt(float(opp) / opp_baseline)))
        w = rw * opp_factor
        weighted_num += float(delta) * w
        weighted_den += w
    if weighted_den <= 0:
        return float(statistics.median([d for d, _ in tail]))
    return float(weighted_num / weighted_den)


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


def projection_trend_multiplier(
    position_key: str,
    age: int,
    year_idx: int,
    player_yoy: Optional[float],
) -> float:
    """
    Position-specific production multiplier layered on top of grade scaling.
    Positive recent trends for younger players get a larger boost in early years;
    negative trends in post-prime years get a larger dampener.
    """
    yoy = 0.0 if player_yoy is None or (isinstance(player_yoy, float) and np.isnan(player_yoy)) else float(player_yoy)
    yoy = max(-8.0, min(10.0, yoy))
    pos = (position_key or "").upper()
    # (develop_end_age, prime_end_age, max_growth_boost, max_decline_penalty)
    profiles = {
        "QB": (27, 33, 0.18, 0.14),
        # Skill RBs: hotter youth ramp; decline still meaningful for RB aging.
        "HB": (24, 27, 0.30, 0.18),
        "RB": (24, 27, 0.30, 0.18),
        # WRs: extend development window slightly + stronger trend-to-production lift.
        "WR": (26, 29, 0.28, 0.16),
        "TE": (26, 30, 0.16, 0.14),
        "T": (25, 31, 0.15, 0.13),
        "G": (25, 31, 0.14, 0.12),
        "C": (26, 32, 0.13, 0.12),
        "ED": (25, 29, 0.18, 0.18),
        "DI": (25, 30, 0.16, 0.18),
        "LB": (24, 28, 0.18, 0.18),
        # Secondaries: noisy year-to-year grades — softer decline pull on counting stats.
        "CB": (24, 28, 0.19, 0.10),
        "S": (25, 30, 0.15, 0.09),
    }
    develop_end, prime_end, boost_cap, decline_cap = profiles.get(pos, (25, 29, 0.14, 0.12))
    age_f = float(age)
    skill_youth = pos in {"WR", "HB", "RB"}
    secondary_soft_decline = pos in {"CB", "S"}

    if yoy >= 0:
        if age_f >= prime_end + 3:
            return 1.0
        # Strongest during development, moderate in prime, fades after.
        if age_f <= develop_end:
            age_weight = 1.0
        elif age_f <= prime_end:
            age_weight = max(0.55, 1.0 - 0.12 * (age_f - develop_end))
        else:
            age_weight = max(0.25, 0.55 - 0.10 * (age_f - prime_end))
        # WR/RB: allow a bit more year-over-year build inside the contract.
        yw_hi = 1.34 if skill_youth else 1.25
        yw_lo = 0.86 if skill_youth else 0.85
        yw_slope = 0.095 if skill_youth else 0.08
        year_weight = max(yw_lo, min(yw_hi, 0.88 + yw_slope * float(year_idx)))
        lift = min(boost_cap, (yoy / 10.0) * boost_cap * age_weight * year_weight)
        if skill_youth and age_f <= prime_end + 1:
            lift *= 1.07
        return 1.0 + max(0.0, lift)

    # Negative trend: amplify only after prime.
    if age_f <= prime_end:
        age_weight = max(0.25, 0.35 + 0.06 * max(0.0, age_f - develop_end))
    else:
        age_weight = min(1.0, 0.65 + 0.10 * (age_f - prime_end))
    # CB/S: dampen compounding of late-contract + negative YoY (coverage volatility).
    yw_hi = 1.22 if secondary_soft_decline else 1.35
    yw_slope = 0.05 if secondary_soft_decline else 0.08
    year_weight = max(0.92, min(yw_hi, 0.97 + yw_slope * float(year_idx)))
    hit = min(decline_cap, (abs(yoy) / 8.0) * decline_cap * age_weight * year_weight)
    if secondary_soft_decline:
        hit *= 0.58
    floor_mult = 0.78 if secondary_soft_decline else 0.70
    return max(floor_mult, 1.0 - max(0.0, hit))
