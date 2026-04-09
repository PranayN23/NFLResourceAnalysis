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

CAP_GROWTH_RATE = 0.065
BASE_CAP_YEAR = 2024
BASE_CAP_DOLLARS = 255.4  # 2024 NFL salary cap in $M

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
    Return the team's players at this position from the most recent year
    in the dataframe, sorted by snap count descending.

    If *exclude_player* is provided, that player is omitted from the
    roster — used for re-signing scenarios where we need to see what
    the roster looks like WITHOUT the player.
    """
    max_year = _effective_year_from_df(position_df, reference_year, "Year")
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
    """Check if a player is currently on the team's roster."""
    max_year = _effective_year_from_df(position_df, reference_year, "Year")
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
) -> pd.DataFrame:
    """
    Aggregate per-team metrics for the most recent year and percentile-
    rank them across the league.

    Star power and starter quality now use a composite that blends
    PFF grade with individual production stats (sacks, pressures,
    stops, hits, hurries, tackles on a per-snap basis), so a player
    who dominates statistically is properly credited even if their
    PFF grade is only "good".
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
) -> pd.DataFrame:
    key = (id(position_df), grade_col, snap_col, reference_year)
    if key not in _league_cache:
        _league_cache[key] = _compute_league_percentiles(
            position_df,
            grade_col=grade_col,
            snap_col=snap_col,
            prod_stat_cols=prod_stat_cols,
            reference_year=reference_year,
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
    snaps_col = snap_col if snap_col in team_rows.columns else "snap_counts_defense"
    grade_c = grade_col if grade_col in team_rows.columns else "grades_defense"
    row = {
        "best_composite":    composites.max(),
        "avg_composite_top2": composites.nlargest(2).mean(),
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
) -> Tuple[float, str]:
    """
    Score 0-100 how strong a team is at this position, ranked against
    the rest of the league. Blends five factors:

    1. **Star power** (25%) — best player's composite percentile.
    2. **Starter quality** (25%) — avg composite of top-2 players.
    3. **Team production** (25%) — total sacks + pressures + stops.
    4. **Depth** (15%) — count of above-median composite players.
    5. **Age risk** (10%) — top-2 starters' age.

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
                )
                if team_row is None:
                    return 0.0, "Weak"
                pctiles = _rank_team_against_league(league, team, team_row)
            else:
                row = league.loc[team]
                pctiles = {c: row[c] for c in row.index if c.endswith("_pctile")}

            star_pctile = pctiles.get("best_composite_pctile", 0.5)
            starter_pctile = pctiles.get("avg_composite_top2_pctile", 0.5)
            depth_pctile = pctiles.get("n_quality_pctile", 0.5)

            prod_cols = [c for c in pctiles if c.startswith("total_") and c.endswith("_pctile")
                         and c != "total_snaps_pctile"]
            production_pctile = np.mean([pctiles[c] for c in prod_cols]) if prod_cols else 0.5

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

            composite = (
                0.25 * star_pctile
                + 0.25 * starter_pctile
                + 0.25 * production_pctile
                + 0.15 * depth_pctile
                + 0.10 * age_factor
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

    max_year = _effective_year_from_df(cap_df, reference_year, "year")
    if max_year is None:
        return 85.0, 15.0
    team_rows = cap_df[(cap_df["Team"] == team) & (cap_df["year"] == max_year)]

    if team_rows.empty:
        return 85.0, 15.0

    allocated = team_rows["Cap_Space"].sum()
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
    top = roster[0]
    top_g = float(top.get("grade") or 0.0)
    if top_g < incumbent_grade_floor:
        return float(pv_fair_aav), ""

    w = REPLACEMENT_INCUMBENT_WEIGHT.get(position_key, 0.34)
    mv_i = float(grade_to_mv(max(45.0, min(100.0, top_g))))
    overlap = w * mv_i
    base = max(float(pv_fair_aav), 0.01)
    adj = max(0.10 * base, base - overlap)
    note = (
        f" Roster replacement: top incumbent ~{top_g:.0f} grade — "
        f"~{int(round(w * 100))}% of their fair AAV (${overlap:.1f}M/yr) overlaps with this role; "
        f"decision surplus uses adjusted fair ${adj:.1f}M/yr vs ${base:.1f}M/yr headline."
    )
    return adj, note


def aav_to_cap_pcts(aav_dollars: float, contract_years: int) -> List[float]:
    """
    Convert a fixed $M AAV to year-by-year cap percentages.
    Cap grows at CAP_GROWTH_RATE per year, so a fixed dollar amount
    shrinks as a fraction of the cap over time.
    """
    pcts = []
    for yr in range(contract_years):
        future_cap = BASE_CAP_DOLLARS * ((1.0 + CAP_GROWTH_RATE) ** yr)
        pcts.append(round(aav_dollars / future_cap * 100, 2))
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
        team_reason = _build_team_reasoning(
            player_name, roster, need_label, need_score,
            yr1_cap_pct, signing_cap_pcts, available_cap_pct,
            f"Base value verdict: {base_decision}. Team-context verdict: Exceeds Cap. "
            "This can still be a fair player price, but it does not fit current cap room."
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

    if adjusted == base_decision:
        combined_note = f"Base value verdict: {base_decision}. Team-context verdict: {adjusted}. {note}".strip()
    else:
        combined_note = (
            f"Base value verdict: {base_decision}. Team-context verdict: {adjusted}. "
            f"This can be a fair player price, but roster/cap context changes the recommendation. {note}"
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
