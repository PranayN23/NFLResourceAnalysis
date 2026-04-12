"""
Team Context — shared utilities for team-mode simulation.

Provides roster extraction, positional need scoring, cap-percentage
analysis, and AAV-to-cap-% conversion using the same CAP_GROWTH_RATE
(6.5 %/yr) that the agent valuation graphs already use.
"""

import pandas as pd
import numpy as np
import os
from typing import Callable, List, Tuple, Optional

from backend.agent.scheme_personnel import (
    SCHEME_PERSONNEL_POSITION_KEYS,
    adjust_positional_need_blend_weights,
    get_team_scheme_personnel_row,
)
from backend.agent.api_year_utils import age_during_season

CAP_GROWTH_RATE = 0.065
BASE_CAP_YEAR = 2024
# Keep in sync with frontend `leagueCapMillions` (freeAgencyPositionConfig.js).
# Historical caps: official NFL figures; 2021 reduced from 2020 due to COVID TV-revenue clawback.
_LEAGUE_CAP_BY_YEAR: dict[int, float] = {
    2010: 102.0,
    2011: 120.0,
    2012: 120.6,
    2013: 123.0,
    2014: 133.0,
    2015: 143.28,
    2016: 155.27,
    2017: 167.0,
    2018: 177.2,
    2019: 188.2,
    2020: 198.2,
    2021: 182.5,
    2022: 208.2,
    2023: 224.8,
    2024: 255.4,
    2025: 279.2,
    2026: 301.2,
}

# Value anchors in all agent graphs are calibrated to this year's OTC contracts.
VALUE_ANCHOR_CALIBRATION_YEAR = 2026

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAP_DATA_PATH = os.path.join(_BASE, "ML", "cap_data.csv")

_cap_df: pd.DataFrame = pd.DataFrame()


def _load_cap_data() -> pd.DataFrame:
    global _cap_df
    if _cap_df.empty and os.path.exists(CAP_DATA_PATH):
        _cap_df = pd.read_csv(CAP_DATA_PATH)
        if "year" in _cap_df.columns:
            _cap_df["year"] = _cap_df["year"].astype(int)
    return _cap_df


def league_cap_millions(year: int) -> float:
    """
    NFL league-year salary cap in $M (not adjusted for in-season moves).
    Uses published figures where available; otherwise projects from nearest
    season at CAP_GROWTH_RATE. Add new years to _LEAGUE_CAP_BY_YEAR when the
    league announces them.
    """
    y = int(year)
    if y in _LEAGUE_CAP_BY_YEAR:
        return float(_LEAGUE_CAP_BY_YEAR[y])
    hi = max(_LEAGUE_CAP_BY_YEAR)
    lo = min(_LEAGUE_CAP_BY_YEAR)
    if y > hi:
        return round(float(_LEAGUE_CAP_BY_YEAR[hi]) * ((1.0 + CAP_GROWTH_RATE) ** (y - hi)), 2)
    return round(float(_LEAGUE_CAP_BY_YEAR[lo]) / ((1.0 + CAP_GROWTH_RATE) ** (lo - y)), 2)


def cap_scale_for_year(year: int) -> float:
    """
    Multiplicative factor to convert value-anchor dollars (calibrated to
    VALUE_ANCHOR_CALIBRATION_YEAR) into *year*-appropriate dollars.

    A grade-80 QB worth $X in 2026 is worth ``X * cap_scale_for_year(2017)``
    in 2017 dollars, since both player value and the absolute salary market
    scale roughly proportionally with the league cap.
    """
    return league_cap_millions(year) / league_cap_millions(VALUE_ANCHOR_CALIBRATION_YEAR)


def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _effective_year_from_df(df: pd.DataFrame, requested_year: int | None, col: str) -> int | None:
    """Pick the latest available year <= requested_year (or absolute latest if omitted)."""
    if col not in df.columns:
        return None
    years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    if years.empty:
        return None
    if requested_year is None:
        return int(years.max())
    elig = years[years <= int(requested_year)]
    return int(elig.max()) if not elig.empty else int(years.min())


def _roster_year_from_df(df: pd.DataFrame, requested_year: int | None, col: str) -> int | None:
    """Pick the latest available year STRICTLY BEFORE requested_year for roster context.

    Free agency and signings happen before the season starts, so when evaluating
    players for analysis_year N, the 'current roster' should reflect who was on
    the team at the END of season N-1 — not the current-season stats, which would
    already show players as signed.
    """
    if col not in df.columns:
        return None
    years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    if years.empty:
        return None
    if requested_year is None:
        return int(years.max())
    elig = years[years < int(requested_year)]
    return int(elig.max()) if not elig.empty else int(years.min())


def _coerce_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Coerce messy numeric columns (often read as strings) into floats.
    Handles common formatting like commas; non-parsable values become NaN.
    """
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == object:
            s = s.astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce")


# ─────────────────────────────────────────────
# Roster extraction from the position CSV
# ─────────────────────────────────────────────
def get_team_roster(
    team: str,
    position_df: pd.DataFrame,
    exclude_player: str = "",
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    reference_year: int | None = None,
) -> List[dict]:
    """
    Return the team's players at this position from the season BEFORE
    reference_year, sorted by snap count descending.

    Using the prior season's data correctly reflects the roster state at
    the start of free agency (before any new signings in reference_year).

    If *exclude_player* is provided, that player is omitted from the
    roster — used for re-signing scenarios where we need to see what
    the roster looks like WITHOUT the player.
    """
    max_year = _roster_year_from_df(position_df, reference_year, "Year")
    if max_year is None:
        return []
    team_df = position_df[
        (position_df["Team"] == team) & (position_df["Year"] == max_year)
    ].copy()

    exclude_norm = exclude_player.strip().lower()

    # Fall back to alternative column names if specified ones not present
    _grade_col = grade_col if grade_col in team_df.columns else (
        "grades_defense" if "grades_defense" in team_df.columns else
        "grades_offense" if "grades_offense" in team_df.columns else None
    )
    _snap_col = snap_col if snap_col in team_df.columns else (
        "snap_counts_defense" if "snap_counts_defense" in team_df.columns else
        "snap_counts_offense" if "snap_counts_offense" in team_df.columns else
        "total_snaps" if "total_snaps" in team_df.columns else None
    )

    roster = []
    my = int(max_year)
    for _, row in team_df.iterrows():
        grade = _safe_float(row.get(_grade_col)) if _grade_col else 0.0
        snaps = _safe_float(row.get(_snap_col)) if _snap_col else 0.0
        cap_pct = _safe_float(row.get("Cap_Space"))
        age = _safe_float(row.get("age"))
        name = row.get("player", "Unknown")

        if name.strip().lower() == exclude_norm and exclude_norm:
            continue

        if snaps < 1 and grade < 1:
            continue

        if "age" in position_df.columns and "Year" in position_df.columns:
            nm = str(name).strip().lower()
            p_hist = position_df[
                position_df["player"].astype(str).str.strip().str.lower() == nm
            ]
            adj = age_during_season(p_hist, my)
            if adj is not None:
                age = float(adj)

        roster.append({
            "player": name,
            "age": int(age) if age else 0,
            "grade": round(grade, 1),
            "snaps": int(snaps),
            "cap_pct": round(cap_pct, 2),
        })

    roster.sort(key=lambda r: r["snaps"], reverse=True)
    return roster


def is_player_on_team(
    player_name: str,
    team: str,
    position_df: pd.DataFrame,
    reference_year: int | None = None,
) -> bool:
    """Check if a player was on the team's roster at the END of the prior season.

    Uses the season before reference_year so that free agency signings in
    reference_year don't incorrectly appear as re-signings.
    """
    max_year = _roster_year_from_df(position_df, reference_year, "Year")
    if max_year is None:
        return False
    team_df = position_df[
        (position_df["Team"] == team) & (position_df["Year"] == max_year)
    ]
    return any(
        team_df["player"].str.strip().str.lower() == player_name.strip().lower()
    )


def get_roster_without_player(roster: List[dict], player_name: str) -> List[dict]:
    """Return a copy of the roster excluding the named player."""
    norm = player_name.strip().lower()
    return [p for p in roster if p["player"].strip().lower() != norm]


# ─────────────────────────────────────────────
# Positional need scoring (league-relative)
# ─────────────────────────────────────────────

# Production stats used to build per-player composite scores.
# Each is normalised to a per-snap rate so low-snap players aren't
# penalised for volume, then the rate is percentile-ranked league-wide.
_PROD_STATS_DEF = ["sacks", "total_pressures", "stops", "hits", "hurries", "tackles"]
_PROD_STATS_OFF = ["yards", "receptions", "touchdowns", "first_downs"]
_PROD_STATS_OL  = ["sacks_allowed", "hurries_allowed", "block_percent"]

# Legacy alias
_PROD_STATS = _PROD_STATS_DEF

# QB / HB / TE: one (or ~two for TE) primary roles on the field — positional
# strength should track the snap-weighted room, not the mean of the top-2
# composites (which overweights a low-snap backup). Other positions use the
# top-2 composite mean and slightly higher depth weight in the blend.
_STARTER_DOMINANT_POSITIONS = frozenset({"QB", "HB", "TE"})


def _compute_starter_strength_value(
    g: pd.DataFrame,
    position_key: str | None,
    snap_col: str | None,
) -> float:
    """Per-team aggregate: snap-weighted composite (starter-dominant) or mean of top-2 composites."""
    comps = g["_composite"]
    if comps.empty:
        return 50.0
    sc = snap_col if snap_col and snap_col in g.columns else None
    if position_key in _STARTER_DOMINANT_POSITIONS and sc:
        snaps = pd.to_numeric(g[sc], errors="coerce").fillna(0).clip(lower=0)
        tot = float(snaps.sum())
        if tot < 1e-9:
            return float(comps.max())
        return float((comps * snaps).sum() / tot)
    return float(comps.nlargest(min(2, len(comps))).mean())


def _positional_need_blend_weights(position_key: str | None) -> tuple[float, float, float, float, float]:
    """(star, starter_strength, production, depth, age); sums to 1.0."""
    if position_key in _STARTER_DOMINANT_POSITIONS:
        return (0.25, 0.32, 0.28, 0.08, 0.07)
    return (0.20, 0.28, 0.24, 0.23, 0.05)


def _player_composite(
    latest: pd.DataFrame,
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    prod_stat_cols: list = None,
) -> pd.Series:
    """
    Build a composite score per player that blends PFF grade (40%)
    with per-snap production rate percentiles (60%).

    This means star power / starter quality reflect both how well
    PFF graded a player AND how productive they actually were.
    """
    if prod_stat_cols is None:
        prod_stat_cols = _PROD_STATS_DEF

    if grade_col not in latest.columns:
        # fall back to any grade column
        for alt in ("grades_defense", "grades_offense", "grades_pass"):
            if alt in latest.columns:
                grade_col = alt
                break

    if snap_col not in latest.columns:
        for alt in ("snap_counts_defense", "snap_counts_offense", "total_snaps", "passing_snaps"):
            if alt in latest.columns:
                snap_col = alt
                break

    available = [c for c in (prod_stat_cols or []) if c in latest.columns]
    _coerce_numeric_inplace(latest, [c for c in [grade_col, snap_col, *available] if c])

    snaps = latest[snap_col].clip(lower=1) if snap_col in latest.columns else pd.Series(1, index=latest.index)
    if not available:
        return latest[grade_col] if grade_col in latest.columns else pd.Series(50.0, index=latest.index)

    rate_pctiles = pd.DataFrame(index=latest.index)
    for col in available:
        rate = latest[col] / snaps
        rate_pctiles[col] = rate.rank(pct=True)

    avg_prod_pctile = rate_pctiles.mean(axis=1)
    prod_score = avg_prod_pctile * 100  # scale to 0-100

    grade_pctile = latest[grade_col].rank(pct=True) * 100

    return 0.40 * grade_pctile + 0.60 * prod_score


def _compute_league_percentiles(
    position_df: pd.DataFrame,
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    prod_stat_cols: list = None,
    reference_year: int | None = None,
    position_key: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-team metrics for the most recent year and percentile-
    rank them across the league.

    Star power and starter quality now use a composite that blends
    PFF grade with individual production stats (sacks, pressures,
    stops, hits, hurries, tackles on a per-snap basis), so a player
    who dominates statistically is properly credited even if their
    PFF grade is only "good".

    *starter_strength* is position-aware: QB/HB/TE use snap-weighted
    composite (starter drives the number); other positions use the mean
    of the top-2 composites (typical two-starter / rotation roles).
    """
    max_year = _effective_year_from_df(position_df, reference_year, "Year")
    if max_year is None:
        return pd.DataFrame()
    latest = position_df[position_df["Year"] == max_year].copy()

    if prod_stat_cols is None:
        prod_stat_cols = _PROD_STATS_DEF

    _snap_col = snap_col if snap_col in latest.columns else (
        "snap_counts_defense" if "snap_counts_defense" in latest.columns else
        "snap_counts_offense" if "snap_counts_offense" in latest.columns else
        "total_snaps" if "total_snaps" in latest.columns else None
    )
    _grade_col = grade_col if grade_col in latest.columns else (
        "grades_defense" if "grades_defense" in latest.columns else
        "grades_offense" if "grades_offense" in latest.columns else None
    )

    # Robustness: some position CSVs have numeric columns as strings.
    # If we don't coerce them, groupby max/sum can throw (e.g., 'str' vs float).
    maybe_numeric = [c for c in [_snap_col, _grade_col] if c]
    if prod_stat_cols:
        maybe_numeric.extend([c for c in prod_stat_cols if c in latest.columns])
    _coerce_numeric_inplace(latest, maybe_numeric)

    latest["_composite"] = _player_composite(latest, grade_col=grade_col, snap_col=snap_col, prod_stat_cols=prod_stat_cols)

    agg_dict = {
        "best_composite":    ("_composite", "max"),
        "avg_composite_top2": ("_composite", lambda x: x.nlargest(2).mean()),
        "n_quality":         ("_composite", lambda x: int((x >= x.quantile(0.55)).sum())),
    }
    if _grade_col:
        agg_dict["best_grade"] = (_grade_col, "max")
    if _snap_col:
        agg_dict["total_snaps"] = (_snap_col, "sum")

    agg = latest.groupby("Team").agg(**agg_dict)

    starter_map = {
        team: _compute_starter_strength_value(g, position_key, _snap_col)
        for team, g in latest.groupby("Team")
    }
    agg["starter_strength"] = [starter_map[t] for t in agg.index]

    prod_agg_stats = prod_stat_cols if prod_stat_cols else ["sacks", "total_pressures", "stops"]
    for stat_col in prod_agg_stats:
        if stat_col in latest.columns:
            agg[f"total_{stat_col}"] = latest.groupby("Team")[stat_col].sum()

    for col in list(agg.columns):
        agg[f"{col}_pctile"] = agg[col].rank(pct=True)

    return agg


_league_cache: dict = {}


def _get_league_stats(
    position_df: pd.DataFrame,
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    prod_stat_cols: list = None,
    reference_year: int | None = None,
    position_key: str | None = None,
) -> pd.DataFrame:
    key = (
        id(position_df),
        grade_col,
        snap_col,
        tuple(prod_stat_cols) if prod_stat_cols is not None else (),
        reference_year,
        position_key,
    )
    if key not in _league_cache:
        _league_cache[key] = _compute_league_percentiles(
            position_df,
            grade_col=grade_col,
            snap_col=snap_col,
            prod_stat_cols=prod_stat_cols,
            reference_year=reference_year,
            position_key=position_key,
        )
    return _league_cache[key]


def _recompute_team_row(
    position_df: pd.DataFrame,
    team: str,
    exclude_player: str = "",
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    prod_stat_cols: list = None,
    reference_year: int | None = None,
    position_key: str | None = None,
) -> pd.Series:
    """
    Recompute a single team's raw aggregate metrics from the CSV,
    optionally excluding a player (for re-signing scenarios).
    Returns a Series with the same raw columns as the league table.
    """
    if prod_stat_cols is None:
        prod_stat_cols = _PROD_STATS_DEF

    max_year = _effective_year_from_df(position_df, reference_year, "Year")
    if max_year is None:
        return None
    team_rows = position_df[
        (position_df["Team"] == team) & (position_df["Year"] == max_year)
    ].copy()

    if exclude_player:
        norm = exclude_player.strip().lower()
        team_rows = team_rows[team_rows["player"].str.strip().str.lower() != norm]

    if team_rows.empty:
        return None

    team_rows["_composite"] = _player_composite(team_rows, grade_col=grade_col, snap_col=snap_col, prod_stat_cols=prod_stat_cols)

    composites = team_rows["_composite"]
    snaps_col = snap_col if snap_col in team_rows.columns else (
        "snap_counts_defense" if "snap_counts_defense" in team_rows.columns else
        "snap_counts_offense" if "snap_counts_offense" in team_rows.columns else
        "passing_snaps" if "passing_snaps" in team_rows.columns else
        "total_snaps" if "total_snaps" in team_rows.columns else None
    )
    grade_c = grade_col if grade_col in team_rows.columns else "grades_defense"
    row = {
        "best_composite":    composites.max(),
        "avg_composite_top2": composites.nlargest(2).mean(),
        "starter_strength": _compute_starter_strength_value(team_rows, position_key, snaps_col),
        "best_grade":        team_rows[grade_c].max() if grade_c in team_rows.columns else 50.0,
        "n_quality":         int((composites >= composites.quantile(0.55)).sum()),
        "total_snaps":       team_rows[snaps_col].sum() if snaps_col in team_rows.columns else 0,
    }
    for stat_col in prod_stat_cols:
        if stat_col in team_rows.columns:
            row[f"total_{stat_col}"] = team_rows[stat_col].sum()

    return pd.Series(row)


def _rank_team_against_league(
    league: pd.DataFrame,
    team: str,
    team_row: pd.Series,
) -> dict:
    """
    Temporarily replace a team's row in the league table, re-rank,
    and return the team's new percentiles.
    """
    modified = league.copy()
    raw_cols = [c for c in modified.columns if not c.endswith("_pctile")]
    for col in raw_cols:
        if col in team_row.index:
            modified.loc[team, col] = team_row[col]

    pctile_map = {}
    for col in raw_cols:
        pctile_col = f"{col}_pctile"
        modified[pctile_col] = modified[col].rank(pct=True)
        pctile_map[pctile_col] = modified.loc[team, pctile_col]

    return pctile_map


def compute_positional_need(
    roster: List[dict],
    position_df: pd.DataFrame = None,
    team: str = "",
    exclude_player: str = "",
    grade_col: str = "grades_defense",
    snap_col: str = "snap_counts_defense",
    prod_stat_cols: list = None,
    reference_year: int | None = None,
    position_key: str | None = None,
) -> Tuple[float, str]:
    """
    Score 0-100 how strong a team is at this position, ranked against
    the rest of the league. Blends five factors (weights depend on
    *position_key*):

    1. **Star power** — best player's composite percentile.
    2. **Starter strength** — for QB/HB/TE, snap-weighted composite
       (starters dominate); for other positions, mean of top-2 composites.
    3. **Team production** — league-relative totals / rates.
    4. **Depth** — count of above-threshold composite players (weighted
       higher for typical two-deep positions).
    5. **Age risk** — for QB/HB/TE, snap-weighted roster age; else top-2
       by grade.

    For **HB, WR, and TE**, blend weights are further adjusted using each
    team's **offensive personnel** file under
    ``backend/ML/scheme/data/*_schemes_with_personnel.csv`` when present
    (latest available year ≤ analysis year).

    When *exclude_player* is set (re-signing), the team's stats are
    recomputed without that player and re-ranked vs. the league.
    """
    if not roster:
        return 0.0, "Weak"

    if prod_stat_cols is None:
        prod_stat_cols = _PROD_STATS_DEF

    if position_df is not None and team:
        league = _get_league_stats(
            position_df,
            grade_col=grade_col,
            snap_col=snap_col,
            prod_stat_cols=prod_stat_cols,
            reference_year=reference_year,
            position_key=position_key,
        )
        if team in league.index:
            if exclude_player:
                team_row = _recompute_team_row(
                    position_df,
                    team,
                    exclude_player,
                    grade_col=grade_col,
                    snap_col=snap_col,
                    prod_stat_cols=prod_stat_cols,
                    reference_year=reference_year,
                    position_key=position_key,
                )
                if team_row is None:
                    return 0.0, "Weak"
                pctiles = _rank_team_against_league(league, team, team_row)
            else:
                row = league.loc[team]
                pctiles = {c: row[c] for c in row.index if c.endswith("_pctile")}

            star_pctile = pctiles.get("best_composite_pctile", 0.5)
            starter_pctile = pctiles.get(
                "starter_strength_pctile",
                pctiles.get("avg_composite_top2_pctile", 0.5),
            )
            depth_pctile = pctiles.get("n_quality_pctile", 0.5)

            prod_cols = [c for c in pctiles if c.startswith("total_") and c.endswith("_pctile")
                         and c != "total_snaps_pctile"]
            production_pctile = np.mean([pctiles[c] for c in prod_cols]) if prod_cols else 0.5

            if position_key in _STARTER_DOMINANT_POSITIONS:
                age_snap = [(p["age"], max(0, int(p["snaps"]))) for p in roster if p.get("age", 0) > 0 and p.get("snaps", 0) > 0]
                if age_snap:
                    ages_a, wts = zip(*age_snap)
                    sw = sum(wts)
                    avg_age_top2 = float(np.average(ages_a, weights=wts)) if sw > 0 else 28
                else:
                    avg_age_top2 = 28
            else:
                qualified = [p for p in roster if p["snaps"] >= 40]
                top_2 = sorted(qualified, key=lambda r: r["grade"], reverse=True)[:2]
                if not top_2:
                    top_2 = sorted(roster, key=lambda r: r["grade"], reverse=True)[:2]
                avg_age_top2 = np.mean([p["age"] for p in top_2 if p["age"] > 0]) if top_2 else 28

            if avg_age_top2 >= 32:
                age_factor = 0.15
            elif avg_age_top2 >= 30:
                age_factor = 0.40
            elif avg_age_top2 >= 28:
                age_factor = 0.65
            else:
                age_factor = 0.90

            w_star, w_starter, w_prod, w_depth, w_age = _positional_need_blend_weights(position_key)
            if position_key in SCHEME_PERSONNEL_POSITION_KEYS:
                scheme_row = get_team_scheme_personnel_row(team, reference_year)
                w_star, w_starter, w_prod, w_depth, w_age = adjust_positional_need_blend_weights(
                    w_star, w_starter, w_prod, w_depth, w_age, position_key, scheme_row
                )
            composite = (
                w_star * star_pctile
                + w_starter * starter_pctile
                + w_prod * production_pctile
                + w_depth * depth_pctile
                + w_age * age_factor
            )

            strength = round(max(0, min(100, composite * 100)), 1)

            if strength >= 75:
                label = "Well-Stocked"
            elif strength >= 40:
                label = "Average"
            else:
                label = "Weak"

            return strength, label

    # Fallback: roster-only scoring (no league context)
    qualified = [p for p in roster if p["snaps"] >= 40]
    quality_players = [p for p in roster if p["grade"] >= 60 and p["snaps"] >= 100]
    top_2 = sorted(qualified, key=lambda r: r["grade"], reverse=True)[:2]
    if not top_2:
        top_2 = sorted(roster, key=lambda r: r["grade"], reverse=True)[:2]
    top_2_avg_grade = np.mean([p["grade"] for p in top_2]) if top_2 else 45.0
    depth_count = len(quality_players)
    best_grade = max((p["grade"] for p in roster), default=45.0)
    avg_age_top2 = np.mean([p["age"] for p in top_2 if p["age"] > 0]) if top_2 else 28

    star_bonus = max(0, (best_grade - 80) * 1.5) if best_grade >= 80 else 0

    if top_2_avg_grade >= 85:
        grade_strength = 50
    elif top_2_avg_grade >= 75:
        grade_strength = 35
    elif top_2_avg_grade >= 65:
        grade_strength = 15
    else:
        grade_strength = 0

    if depth_count >= 4:
        depth_strength = 25
    elif depth_count >= 3:
        depth_strength = 17
    elif depth_count >= 2:
        depth_strength = 10
    else:
        depth_strength = 0

    if avg_age_top2 >= 32:
        age_strength = 0
    elif avg_age_top2 >= 30:
        age_strength = 8
    elif avg_age_top2 >= 28:
        age_strength = 15
    else:
        age_strength = 20

    strength = min(100.0, max(0, float(grade_strength + depth_strength + age_strength + star_bonus)))

    if strength >= 75:
        label = "Well-Stocked"
    elif strength >= 40:
        label = "Average"
    else:
        label = "Weak"

    return round(strength, 1), label


# ─────────────────────────────────────────────
# Team cap from cap_data.csv (% of cap)
# ─────────────────────────────────────────────
def get_team_cap(team: str, reference_year: int | None = None) -> Tuple[float, float]:
    """
    Sum Cap_Space percentages for the team's most recent year.
    Returns (allocated_pct, available_pct).
    """
    cap_df = _load_cap_data()
    if cap_df.empty:
        return 85.0, 15.0

    effective_year = _effective_year_from_df(cap_df, reference_year, "year")
    if effective_year is None:
        return 85.0, 15.0
    requested_year = int(reference_year) if reference_year is not None else effective_year
    team_rows = cap_df[(cap_df["Team"] == team) & (cap_df["year"] == effective_year)]

    if team_rows.empty:
        return 85.0, 15.0

    allocated = float(team_rows["Cap_Space"].sum())

    # If requesting a future year beyond available cap snapshots, project cap burden
    # by holding implied dollars constant and growing league cap (~6.5%/yr).
    if requested_year > effective_year:
        cap_then = league_cap_millions(effective_year)
        cap_req = league_cap_millions(requested_year)
        if cap_req > 0:
            implied_dollars = (allocated / 100.0) * cap_then
            allocated = (implied_dollars / cap_req) * 100.0

    allocated = min(allocated, 100.0)
    available = round(100.0 - allocated, 1)
    return round(allocated, 1), max(0.0, available)


def get_all_teams(cap_df: pd.DataFrame = None, reference_year: int | None = None) -> List[str]:
    """Return sorted list of all team names from cap_data."""
    if cap_df is None:
        cap_df = _load_cap_data()
    if cap_df.empty:
        return []
    max_year = _effective_year_from_df(cap_df, reference_year, "year")
    if max_year is None:
        return []
    teams = cap_df[cap_df["year"] == max_year]["Team"].dropna().unique().tolist()
    return sorted(teams)


# ─────────────────────────────────────────────
# AAV → per-year cap %
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Replacement / snap overlap — signing often displaces an incumbent, not pure addition
# ─────────────────────────────────────────────
# Weight on incumbent's market value to subtract from headline fair AAV when making
# SIGN/PASS surplus (stronger for true one-starter roles).
REPLACEMENT_INCUMBENT_WEIGHT: dict[str, float] = {
    "QB": 0.88,
    "HB": 0.82,
    "TE": 0.52,
    "WR": 0.36,
    "T": 0.26,
    "G": 0.26,
    "C": 0.26,
    "ED": 0.38,
    "DI": 0.38,
    "LB": 0.40,
    "CB": 0.38,
    "S": 0.36,
}

# Approximate starters by position for depth-chart replacement inference.
POSITION_STARTER_SLOTS: dict[str, int] = {
    "QB": 1,
    "HB": 1,
    "WR": 3,
    "TE": 1,
    "T": 2,
    "G": 2,
    "C": 1,
    "ED": 2,
    "DI": 2,
    "LB": 2,
    "CB": 3,
    "S": 2,
}


def _infer_replacement_target(
    roster: List[dict],
    signee_composite_grade: float,
    position_key: str,
) -> Tuple[Optional[dict], int, str]:
    """
    Infer which incumbent depth slot this signee would displace.
    Returns: (incumbent_row_or_none, slot_idx_1_based, role_label)
    """
    if not roster:
        return None, 1, "starter"

    depth_sorted = sorted(
        roster,
        key=lambda r: (
            -_safe_float(r.get("snaps"), 0.0),
            -_safe_float(r.get("grade"), 0.0),
        ),
    )
    sig_g = float(signee_composite_grade)
    # Number of incumbents clearly ahead on grade approximates signee's depth slot.
    ahead = sum(1 for r in depth_sorted if _safe_float(r.get("grade"), 0.0) > (sig_g + 0.75))
    slot = max(1, min(len(depth_sorted), ahead + 1))
    incumbent = depth_sorted[slot - 1] if depth_sorted else None
    starter_slots = int(POSITION_STARTER_SLOTS.get(position_key, 1))
    role = "starter" if slot <= starter_slots else "backup/depth"
    return incumbent, slot, role


def decision_fair_aav_with_replacement(
    pv_fair_aav: float,
    grade_to_mv: Callable[[float], float],
    signee_composite_grade: float,
    roster: Optional[List[dict]],
    position_key: str,
    incumbent_grade_floor: float = 54.0,
) -> Tuple[float, str]:
    """
    When team mode supplies a roster, reduce the fair AAV used for surplus_pct so
    overlapping value vs a decent incumbent is not double-counted.

    Returns (adjusted_fair_aav, note_suffix). If no adjustment, note is "".
    """
    if not roster:
        return float(pv_fair_aav), ""
    incumbent, slot, role = _infer_replacement_target(roster, signee_composite_grade, position_key)
    if not incumbent:
        return float(pv_fair_aav), ""
    inc_g = float(incumbent.get("grade") or 0.0)
    inc_name = str(incumbent.get("player") or "incumbent")
    starter_slots = int(POSITION_STARTER_SLOTS.get(position_key, 1))
    if inc_g < incumbent_grade_floor:
        return float(pv_fair_aav), ""

    w = REPLACEMENT_INCUMBENT_WEIGHT.get(position_key, 0.34)
    # Backup/depth displacement should have a lighter overlap effect than QB1/CB1-type replacement.
    if role != "starter":
        w *= 0.55
    mv_i = float(grade_to_mv(max(45.0, min(100.0, inc_g))))
    overlap = w * mv_i
    base = max(float(pv_fair_aav), 0.01)
    adj = max(0.10 * base, base - overlap)
    role_note = (
        f"starter slot (#{slot} of {starter_slots})"
        if role == "starter"
        else f"backup/depth slot (#{slot})"
    )
    note = (
        f" Roster replacement: projected to displace {inc_name} "
        f"({role_note}, ~{inc_g:.0f} grade). "
        f"~{int(round(w * 100))}% of that incumbent fair AAV (${overlap:.1f}M/yr) overlaps with this role; "
        f"decision surplus uses adjusted fair ${adj:.1f}M/yr vs ${base:.1f}M/yr headline."
    )
    return adj, note


def aav_to_cap_pcts(
    aav_dollars: float,
    contract_years: int,
    first_league_year: int | None = None,
) -> List[float]:
    """
    Convert a fixed $M AAV to year-by-year cap percentages.
    Each contract year t uses the NFL salary cap for league season (first_league_year + t).
    If first_league_year is omitted, defaults to BASE_CAP_YEAR (legacy behavior anchor).
    """
    y0 = int(first_league_year) if first_league_year is not None else BASE_CAP_YEAR
    pcts = []
    for t in range(contract_years):
        cap_y = league_cap_millions(y0 + t)
        pcts.append(round(aav_dollars / max(cap_y, 1e-6) * 100, 2))
    return pcts


# ─────────────────────────────────────────────
# Team-fit decision adjustment — full matrix
# ─────────────────────────────────────────────
# Every (need_label, base_value_decision) pair maps to a combined
# team-aware tier + explanation.  This is the core of the GM logic:
# value alone isn't enough — team context changes the verdict.

_TEAM_TIER_MATRIX = {
    # ── Position of NEED (Weak) ──
    # Even overpays become more justifiable; great value is a slam dunk
    ("Weak", "Exceptional Value"): ("Must Sign — Elite Value + Need",
        "Elite value at a position of dire need — don't let this player leave."),
    ("Weak", "Good Signing"):      ("Priority Target",
        "Good value at a weak position — this should be a top priority."),
    ("Weak", "Fair Deal"):         ("Fill the Gap",
        "Market-rate deal, but the positional weakness makes it worthwhile."),
    ("Weak", "Slight Overpay"):    ("Justifiable Overpay",
        "Slight premium, but the team badly needs help here."),
    ("Weak", "Overpay"):           ("Overpay — But Consider",
        "Significant overpay, though the positional weakness may justify the premium."),
    ("Weak", "Poor Signing"):      ("Desperation Overpay",
        "Severe overpay even accounting for need — explore cheaper alternatives."),

    # ── AVERAGE positional need ──
    # Tiers stay close to the pure-value assessment
    ("Average", "Exceptional Value"): ("Exceptional Value",
        "Elite value at a position with moderate need — strong signing."),
    ("Average", "Good Signing"):      ("Good Signing",
        "Solid value at a position with room for improvement."),
    ("Average", "Fair Deal"):         ("Fair Deal",
        "Market-rate deal at a position of moderate need."),
    ("Average", "Slight Overpay"):    ("Slight Overpay",
        "Modest premium with only moderate positional need."),
    ("Average", "Overpay"):           ("Overpay",
        "Overpaying at a position that isn't a priority."),
    ("Average", "Poor Signing"):      ("Poor Signing",
        "Severe overpay with no pressing need to justify it."),

    # ── Position already STRONG (Well-Stocked) ──
    # Good value is a luxury add; market rate or worse is wasteful
    ("Well-Stocked", "Exceptional Value"): ("Luxury Add — Great Value",
        "Excellent value, but a luxury — this position is already strong."),
    ("Well-Stocked", "Good Signing"):      ("Luxury Add",
        "Good value as depth, but cap dollars may be better spent elsewhere."),
    ("Well-Stocked", "Fair Deal"):         ("Unnecessary Spend",
        "Paying market rate at an already-strong position — poor allocation."),
    ("Well-Stocked", "Slight Overpay"):    ("Wasteful Overpay",
        "Overpaying for depth the team doesn't need."),
    ("Well-Stocked", "Overpay"):           ("Poor Signing",
        "Significant overpay at an already-stacked position group."),
    ("Well-Stocked", "Poor Signing"):      ("Cap Mismanagement",
        "Severe overpay with no positional justification — avoid."),
}


def assess_team_fit(
    base_decision: str,
    surplus_pct: float,
    need_score: float,
    need_label: str,
    signing_cap_pcts: List[float],
    available_cap_pct: float,
    roster: List[dict],
    player_name: str,
) -> Tuple[str, str, str]:
    """
    Combine the pure-value decision with team context (positional
    strength, cap space) to produce a team-aware tier.

    Returns (adjusted_decision, fit_summary, team_reasoning).
    """
    yr1_cap_pct = signing_cap_pcts[0] if signing_cap_pcts else 0.0

    # Hard cap check — overrides everything
    if yr1_cap_pct > available_cap_pct:
        fit_summary = (
            f"Base value verdict: {base_decision}. Team-context verdict: Exceeds Cap. "
            f"Signing requires {yr1_cap_pct:.1f}% of cap but only {available_cap_pct:.1f}% is available."
        )
        cap_gap = yr1_cap_pct - available_cap_pct
        team_reason = _build_team_reasoning(
            player_name, roster, need_label, need_score,
            yr1_cap_pct, signing_cap_pcts, available_cap_pct,
            (
                f"Valuation bucket is still {base_decision}, but Year 1 asks for {yr1_cap_pct:.1f}% of cap "
                f"with {available_cap_pct:.1f}% free — roughly {cap_gap:.1f}% short. "
                f"That only works if the hit is deferred, trimmed, or the room picture changes."
            ),
        )
        return "Exceeds Cap", fit_summary, team_reason

    cap_burden_ratio = yr1_cap_pct / max(available_cap_pct, 0.01)

    # Look up the full matrix
    key = (need_label, base_decision)
    if key in _TEAM_TIER_MATRIX:
        adjusted, note = _TEAM_TIER_MATRIX[key]
    else:
        adjusted = base_decision
        note = f"Positional strength: {need_label}."

    # Aggressive cap usage can further downgrade
    if cap_burden_ratio >= 0.50 and adjusted not in (
        "Exceeds Cap", "Poor Signing", "Cap Mismanagement", "Desperation Overpay"
    ):
        note += (
            f" WARNING: This signing consumes {cap_burden_ratio*100:.0f}% "
            f"of remaining cap room."
        )
        # Bump down one severity level for extreme cap burden
        _cap_downgrades = {
            "Fill the Gap": "Justifiable Overpay",
            "Justifiable Overpay": "Overpay — But Consider",
            "Fair Deal": "Slight Overpay",
            "Slight Overpay": "Overpay",
            "Unnecessary Spend": "Wasteful Overpay",
            "Wasteful Overpay": "Poor Signing",
            "Overpay": "Poor Signing",
            "Overpay — But Consider": "Desperation Overpay",
        }
        if adjusted in _cap_downgrades:
            adjusted = _cap_downgrades[adjusted]

    who = (player_name or "").strip() or "This player"
    if adjusted == base_decision:
        combined_note = f"Contract tier and team read both land on {adjusted}. {note}".strip()
    else:
        combined_note = (
            f"{who} charts as a {base_decision} on the contract alone; "
            f"layer in a {need_label} positional room ({need_score:.0f}/100 need score) "
            f"plus this cap share and the tag moves to {adjusted}. {note}"
        ).strip()

    fit_summary = combined_note

    team_reason = _build_team_reasoning(
        player_name, roster, need_label, need_score,
        yr1_cap_pct, signing_cap_pcts, available_cap_pct,
        combined_note,
    )

    return adjusted, fit_summary, team_reason


def _build_team_reasoning(
    player_name: str,
    roster: List[dict],
    need_label: str,
    need_score: float,
    yr1_cap_pct: float,
    signing_cap_pcts: List[float],
    available_cap_pct: float,
    extra_note: str,
) -> str:
    """Build the team-context paragraph appended to reasoning."""
    top_players = roster[:3]
    roster_str = ", ".join(
        f"{p['player']} ({p['grade']}, {p['cap_pct']:.1f}% cap)"
        for p in top_players
    )
    if not roster_str:
        roster_str = "no notable players"

    cap_trajectory = ""
    if len(signing_cap_pcts) > 1:
        last_pct = signing_cap_pcts[-1]
        cap_trajectory = (
            f" The fixed AAV shrinks from {yr1_cap_pct:.1f}% of cap in Year 1 "
            f"to {last_pct:.1f}% by Year {len(signing_cap_pcts)} as the cap "
            f"grows at ~6.5%/yr."
        )

    cap_after = round(available_cap_pct - yr1_cap_pct, 1)

    return (
        f"TEAM CONTEXT: Current roster at this position includes {roster_str}. "
        f"Positional strength: {need_label} ({need_score:.0f}/100). "
        f"This signing would consume {yr1_cap_pct:.1f}% of the salary cap, "
        f"leaving {cap_after:.1f}% of {available_cap_pct:.1f}% available room "
        f"({yr1_cap_pct/max(available_cap_pct,0.01)*100:.0f}% of remaining cap).{cap_trajectory} "
        f"{extra_note}".strip()
    )
