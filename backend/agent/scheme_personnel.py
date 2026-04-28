"""
Team offensive scheme + personnel (from backend/ML/scheme) for positional-need tuning.

Uses per-team personnel usage rates (11, 12, 13, etc.) to slightly adjust blend
weights in ``compute_positional_need`` for **HB, WR, and TE only** (the positions
this dataset supports well).
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Any

import pandas as pd

from backend.agent.api_year_utils import clamp_analysis_year
from backend.agent.team_summary import TEAM_TO_ABBR

_SCHEME_DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "ML",
    "scheme",
    "data",
)

# Positions using personnel/scheme-based adjustments.
# QB: shotgun rate adjusts expected dropback volume and starter-strength weight.
# T: pass-heavy packages (low heavy-run rate) elevate pass-block value weight.
# HB/WR/TE: original personnel-rate adjustments (unchanged).
SCHEME_PERSONNEL_POSITION_KEYS = frozenset({"HB", "WR", "TE", "QB", "T"})


def _abbr_for_team(team_name: str) -> str | None:
    if not team_name:
        return None
    t = team_name.strip()
    if t in TEAM_TO_ABBR:
        return TEAM_TO_ABBR[t]
    tl = t.lower()
    for full, abbr in TEAM_TO_ABBR.items():
        if full.lower() == tl:
            return abbr
    return None


def _pct(val: Any) -> float:
    try:
        f = float(val)
        if pd.isna(f):
            return 0.0
        return max(0.0, min(1.0, f / 100.0))
    except (TypeError, ValueError):
        return 0.0


@lru_cache(maxsize=1)
def _personnel_file_years() -> tuple[int, ...]:
    """Years for which ``{year}_schemes_with_personnel.csv`` exists, sorted ascending."""
    years: list[int] = []
    pattern = os.path.join(_SCHEME_DATA_DIR, "*_schemes_with_personnel.csv")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        prefix = base.split("_")[0]
        try:
            years.append(int(prefix))
        except ValueError:
            continue
    return tuple(sorted(years))


def _effective_personnel_year(requested_year: int | None) -> int | None:
    """
    Pick the personnel file year to use: latest year available that is <= requested,
    matching ``effective_year_for_df`` behavior. If the request is before any file,
    use the earliest available year.
    """
    avail = _personnel_file_years()
    if not avail:
        return None
    y_req = clamp_analysis_year(requested_year)
    eligible = [yy for yy in avail if yy <= y_req]
    if eligible:
        return max(eligible)
    return min(avail)


@lru_cache(maxsize=32)
def _load_schemes_with_personnel(year: int) -> pd.DataFrame | None:
    path = os.path.join(_SCHEME_DATA_DIR, f"{year}_schemes_with_personnel.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_team_scheme_personnel_row(team_name: str, reference_year: int | None) -> dict[str, Any] | None:
    """
    One merged scheme + personnel row for the franchise (offense-side stats),
    or None if unavailable. Uses the best-matching year among all
    ``*_schemes_with_personnel.csv`` files on disk.
    """
    abbr = _abbr_for_team(team_name)
    if not abbr:
        return None
    try_y = _effective_personnel_year(reference_year)
    if try_y is None:
        return None
    df = _load_schemes_with_personnel(try_y)
    if df is None or df.empty or "team_abbr" not in df.columns:
        return None
    sub = df[df["team_abbr"].astype(str).str.upper() == abbr.upper()]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def adjust_positional_need_blend_weights(
    w_star: float,
    w_starter: float,
    w_prod: float,
    w_depth: float,
    w_age: float,
    position_key: str | None,
    scheme_row: dict[str, Any] | None,
) -> tuple[float, float, float, float, float]:
    """
    Nudge the five blend weights using team personnel / scheme when applicable.

    **HB**: heavy run packages → slightly more depth weight.
    **WR**: high 11 → slightly more depth (multiple WR snaps).
    **TE**: high 12/13 usage → more weight on starter/TE quality.
    """
    if not scheme_row or not position_key or position_key not in SCHEME_PERSONNEL_POSITION_KEYS:
        return w_star, w_starter, w_prod, w_depth, w_age

    p11 = _pct(scheme_row.get("personnel_11_rate", 61))
    p12 = _pct(scheme_row.get("personnel_12_rate", 0))
    p13 = _pct(scheme_row.get("personnel_13_rate", 0))
    p20 = _pct(scheme_row.get("personnel_20_rate", 0))
    p21 = _pct(scheme_row.get("personnel_21_rate", 0))
    p22 = _pct(scheme_row.get("personnel_22_rate", 0))
    shotgun = _pct(scheme_row.get("shotgun_rate", 60))

    two_te = p12 + p13
    heavy_run = p20 + p21 + p22 + 0.35 * two_te

    w = [w_star, w_starter, w_prod, w_depth, w_age]

    if position_key == "QB":
        # High shotgun rate → more pass volume → starter-strength and star-power matter more.
        # League average shotgun ~60%; teams above 70% are clearly pass-first.
        if shotgun > 0.60:
            sq_idx = min(1.0, (shotgun - 0.60) / 0.25)
            shift = 0.05 * sq_idx
            w[1] += shift * 0.60   # starter_strength
            w[0] += shift * 0.40   # star_power
            w[3] = max(0.02, w[3] - shift * 0.55)   # reduce depth weight
            w[4] = max(0.02, w[4] - shift * 0.45)   # reduce age weight

    elif position_key == "T":
        # Pass-heavy teams (low heavy-run, high 11 personnel) elevate the value of a quality
        # pass-protector — up-weight star/starter, down-weight depth.
        pass_heavy_idx = max(0.0, min(1.0, (p11 - 0.55) / 0.30)) if p11 > 0.55 else 0.0
        if pass_heavy_idx > 0:
            shift = 0.04 * pass_heavy_idx
            w[0] += shift * 0.55   # star_power
            w[1] += shift * 0.45   # starter_strength
            w[3] = max(0.02, w[3] - shift)   # reduce depth

    elif position_key == "TE":
        te_idx = max(0.0, min(1.0, (two_te - 0.22) / 0.38))
        shift = 0.06 * te_idx
        w[1] += shift * 0.65
        w[0] += shift * 0.35
        w[3] = max(0.02, w[3] - shift * 0.45)

    elif position_key == "HB":
        if heavy_run > 0.22:
            hr = min(1.0, (heavy_run - 0.22) / 0.38)
            shift = 0.06 * hr
            w[3] += shift
            w[1] = max(0.02, w[1] - shift * 0.55)
            w[0] = max(0.02, w[0] - shift * 0.15)

    elif position_key == "WR":
        if p11 > 0.56:
            shift = 0.05 * min(1.0, (p11 - 0.56) / 0.28)
            w[3] += shift
            w[1] = max(0.02, w[1] - shift * 0.45)
            w[0] = max(0.02, w[0] - shift * 0.15)

    s = sum(w)
    if s <= 0:
        return w_star, w_starter, w_prod, w_depth, w_age
    return tuple(x / s for x in w)


def compact_scheme_personnel_for_api(team_name: str, reference_year: int | None) -> dict[str, Any] | None:
    """Small payload for team-roster responses (HB / WR / TE only)."""
    row = get_team_scheme_personnel_row(team_name, reference_year)
    if not row:
        return None
    season = row.get("season", reference_year)
    try:
        season = int(season)
    except (TypeError, ValueError):
        season = reference_year
    eff_y = _effective_personnel_year(reference_year)
    return {
        "personnel_file_year": eff_y,
        "season": season,
        "personnel_11_pct": round(float(row.get("personnel_11_rate", 0) or 0), 2),
        "personnel_12_pct": round(float(row.get("personnel_12_rate", 0) or 0), 2),
        "personnel_13_pct": round(float(row.get("personnel_13_rate", 0) or 0), 2),
        "shotgun_pct": round(float(row.get("shotgun_rate", 0) or 0), 2),
        "note": "Positional need weights use this team's offensive personnel mix (HB/WR/TE).",
    }
