"""
QB GM Agent — Quarterback Free Agency Evaluator

LangGraph agent that evaluates a QB free agent and returns a
SIGN / PASS recommendation with full year-by-year contract breakdown.

Stats grade weighted by: passer_rating 35%, ypa 30%, btt_rate 15%,
completion_pct 10%, epa_per_dropback 10%.
Composite grade: model pass grade 45%, stats grade 55%.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.team_context import (
    assess_team_fit as _assess_team_fit_logic,
    aav_to_cap_pcts,
    decision_fair_aav_with_replacement,
    cap_scale_for_year,
)
from backend.agent.grade_projection import (
    grade_to_tier_universal,
    player_recent_grade_yoy,
    apply_yearly_grade_step,
    projection_trend_multiplier,
)
from backend.agent.stat_projection_utils import (
    qb_full_role_dropbacks_17,
    qb_stabilized_int_rate_per_db,
    qb_stabilized_td_rate_per_db,
    infer_qb_signing_role,
    qb_dropbacks_17_for_role,
    inactivity_retirement_penalty,
    apply_inactivity_to_projection_list,
    apply_projection_plausibility_caps,
    snap_value_reliability_factor,
)
import pandas as pd
import numpy as np
import os, datetime

from backend.agent.api_year_utils import resolve_player_age_for_evaluation

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QB_CSV_PATH = os.path.join(_BASE, "ML", "QB.csv")

# ─────────────────────────────────────────────
# Grade → Market Value (2026 OTC calibrated — QBs command highest AAV)
# ─────────────────────────────────────────────
_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [1.14, 4.55, 26.14, 31.82, 38.64, 47.73, 62.50, 65.91, 68.18, 70.45, 72.73, 75.00]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# ─────────────────────────────────────────────
# Stats-based grade
# Empirical anchors (snap-normalised, league-wide, 17-game basis):
#   passer_rating: Reserve~72, Rotation~82, Starter~92, Elite~105
#   ypa: Reserve~5.8, Rotation~6.8, Starter~7.5, Elite~8.5
#   btt_rate: Reserve~3%, Rotation~4%, Starter~5%, Elite~7%
#   completion_pct: Reserve~60%, Rotation~64%, Starter~67%, Elite~70%
#   epa_per_db: Reserve~-0.1, Rotation~0.0, Starter~0.10, Elite~0.20
# ─────────────────────────────────────────────
_QBR_ANCHORS   = [0.0,  72.0,  82.0,  92.0, 105.0, 130.0]
_YPA_ANCHORS   = [0.0,   5.8,   6.8,   7.5,   8.5,  11.0]
_BTT_ANCHORS   = [0.0,  0.03,  0.04,  0.05,  0.07,  0.10]
_CMP_ANCHORS   = [0.0,  60.0,  64.0,  67.0,  70.0,  78.0]
_EPA_DB_ANCHORS= [-0.4, -0.10,  0.00,  0.10,  0.20,  0.35]
_STAT_GRD_SCALE= [45.0, 55.0,  65.0,  75.0,  85.0,  99.0]


def _stats_grade(passer_rating, ypa, btt_rate, cmp_pct, epa_per_db):
    pr = float(np.interp(passer_rating, _QBR_ANCHORS, _STAT_GRD_SCALE))
    ya = float(np.interp(ypa, _YPA_ANCHORS, _STAT_GRD_SCALE))
    bt = float(np.interp(btt_rate, _BTT_ANCHORS, _STAT_GRD_SCALE))
    cp = float(np.interp(cmp_pct, _CMP_ANCHORS, _STAT_GRD_SCALE))
    ep = float(np.interp(epa_per_db, _EPA_DB_ANCHORS, _STAT_GRD_SCALE))
    return round(0.35 * pr + 0.30 * ya + 0.15 * bt + 0.10 * cp + 0.10 * ep, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.45 * model_grade + 0.55 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# ─────────────────────────────────────────────
# Age-based annual grade delta (QB-specific; median YoY grades_offense change on
# consecutive ML/QB.csv seasons, age k → k+1 keyed by k). Regenerate via
# backend/agent/compute_position_age_curves.py
# ─────────────────────────────────────────────
_AGE_DELTAS = {
    20: +4.9, 21: +4.9, 22: +2.7, 23: -0.8, 24: -1.0, 25: -3.2, 26: +5.3,
    27: -3.3, 28: -0.2, 29: +2.9, 30: -2.1, 31: -1.6, 32: -0.9, 33: -4.8,
    34: +3.1, 35: -4.5, 36: -6.1, 37: +5.6, 38: -2.8,
}


def _annual_grade_delta(age):
    k = int(age)
    lo, hi = min(_AGE_DELTAS), max(_AGE_DELTAS)
    return _AGE_DELTAS[max(lo, min(hi, k))]


def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _max_games_for_year(year: int) -> float:
    return 17.0 if int(year) >= 2021 else 16.0


def _infer_games_played(row, max_g: float) -> float:
    """
    QB.csv often omits player_game_count. Defaulting missing values to max_g
    treats backup seasons as full 17-game samples and crushes per-game volume.
    Infer games from dropbacks (~38 DB/gm full-time) or attempts when missing.
    """
    pg = _safe_float(row.get("player_game_count"), float("nan"))
    if not np.isnan(pg) and pg > 0:
        return max(1.0, min(max_g, pg))
    db = max(0.0, _safe_float(row.get("dropbacks"), 0.0))
    att = max(0.0, _safe_float(row.get("attempts"), 0.0))
    if db <= 0 and att <= 0:
        return 1.0
    base = max(db, att * 0.98)
    inferred = max(1.0, min(max_g, base / 38.0))
    return inferred


# Seasons with at least this many dropbacks count toward "starter peak" load.
_MIN_DB_FOR_STARTER_PEAK = 100

# Last 1–3 seasons, chronological order (oldest → newest): emphasize the most recent year.
_QB_RECENCY_W_3 = (0.10, 0.25, 0.65)
_QB_RECENCY_W_2 = (0.28, 0.72)


def _qb_recency_weights_n(n: int) -> List[float]:
    if n <= 0:
        return [1.0]
    if n == 1:
        return [1.0]
    tpl = _QB_RECENCY_W_2 if n == 2 else _QB_RECENCY_W_3
    w = list(tpl[-n:])
    s = sum(w)
    return [x / s for x in w]


def _qb_ints_from_row(row) -> float:
    """Interceptions column name can vary across merged CSVs."""
    for k in ("interceptions", "INT"):
        if hasattr(row, "get"):
            v = row.get(k)
            if v is not None:
                try:
                    f = float(v)
                    if not np.isnan(f):
                        return f
                except Exception:
                    pass
    return 0.0


def _effective_qb_int_rate(last_stats: dict) -> float:
    r = float(last_stats.get("int_rate_per_db") or 0.0)
    if r > 1e-8:
        return r
    return qb_stabilized_int_rate_per_db(
        float(last_stats.get("career_c_ints", 0)),
        float(last_stats.get("career_c_dbs", 1)),
    )


def _effective_qb_td_rate(last_stats: dict) -> float:
    r = float(last_stats.get("td_rate_per_db") or 0.0)
    if r > 1e-8:
        return r
    return qb_stabilized_td_rate_per_db(
        float(last_stats.get("career_c_tds", 0)),
        float(last_stats.get("career_c_dbs", 1)),
    )


def _apply_qb_volume_from_rates(last_stats: dict, proj_dbs: float) -> None:
    """Recompute 17g counting stats from per-DB rates × load; fixes INT/TD when rates were rounded to 0."""
    d = max(1.0, float(proj_dbs))
    ir = _effective_qb_int_rate(last_stats)
    tr = _effective_qb_td_rate(last_stats)
    ypa = float(last_stats.get("ypa") or 0.0)
    sack_rate = max(0.02, min(0.18, float(last_stats.get("sack_rate_db") or 0.065)))
    cmp_pct = max(45.0, min(80.0, float(last_stats.get("completion_pct") or 62.0)))
    atts = d * (1.0 - sack_rate)
    comps = atts * (cmp_pct / 100.0)
    last_stats["proj_dbs_17g"] = round(d, 1)
    last_stats["proj_atts_17g"] = round(atts, 1)
    last_stats["proj_completions_17g"] = round(comps, 1)
    last_stats["ints_17g"] = round(min(35.0, ir * d), 1)
    last_stats["tds_17g"] = round(min(60.0, tr * d), 1)
    last_stats["yards_17g"] = round(ypa * atts)
    if float(last_stats.get("int_rate_per_db") or 0) < 1e-8:
        last_stats["int_rate_per_db"] = round(ir, 8)
    if float(last_stats.get("td_rate_per_db") or 0) < 1e-8:
        last_stats["td_rate_per_db"] = round(tr, 8)


def _has_valid_stats(row):
    for col in ("yards", "touchdowns", "dropbacks", "grades_offense", "grades_pass"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f):
                return True
        except Exception:
            pass
    return False


def _aggregate_qb_history_by_year(history: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate split-team seasons into one row per year so career tables and
    last-season snapshots reflect full-year production.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    rows = []
    src = history.sort_values("Year").copy()
    for yr, grp in src.groupby("Year", sort=True):
        max_g = _max_games_for_year(int(_safe_float(yr, 2024)))
        attempts = sum(_safe_float(r.get("attempts")) for _, r in grp.iterrows())
        completions = sum(_safe_float(r.get("completions")) for _, r in grp.iterrows())
        dropbacks = sum(_safe_float(r.get("dropbacks")) for _, r in grp.iterrows())
        yards = sum(_safe_float(r.get("yards")) for _, r in grp.iterrows())
        tds = sum(_safe_float(r.get("touchdowns")) for _, r in grp.iterrows())
        btts = sum(_safe_float(r.get("big_time_throws")) for _, r in grp.iterrows())
        ints = sum(_qb_ints_from_row(r) for _, r in grp.iterrows())
        epa = sum(_safe_float(r.get("Net EPA")) for _, r in grp.iterrows())
        games = sum(_safe_float(r.get("player_game_count")) for _, r in grp.iterrows())
        games = max(1.0, min(max_g, games if games > 0 else _infer_games_played(grp.iloc[-1], max_g)))

        w_att = max(attempts, 1.0)
        w_db = max(dropbacks, 1.0)
        qb_rating = sum(_safe_float(r.get("qb_rating")) * _safe_float(r.get("attempts")) for _, r in grp.iterrows()) / w_att
        pass_grade = sum(_safe_float(r.get("grades_pass")) * _safe_float(r.get("dropbacks")) for _, r in grp.iterrows()) / w_db
        off_grade = sum(_safe_float(r.get("grades_offense")) * _safe_float(r.get("dropbacks")) for _, r in grp.iterrows()) / w_db
        run_grade = sum(_safe_float(r.get("grades_run")) * _safe_float(r.get("dropbacks")) for _, r in grp.iterrows()) / w_db

        # Guard against known data anomalies (e.g., 6000+ pass-yard single season rows).
        season_yards = min(5800.0, max(0.0, yards))

        rows.append({
            "Year": int(_safe_float(yr, 2024)),
            "player_game_count": games,
            "age": _safe_float(grp.iloc[-1].get("age")),
            "attempts": attempts,
            "completions": completions,
            "dropbacks": dropbacks,
            "yards": season_yards,
            "touchdowns": tds,
            "interceptions": ints,
            "big_time_throws": btts,
            "completion_percent": (completions / w_att) * 100.0,
            "btt_rate": btts / w_db,
            "ypa": season_yards / w_att,
            "qb_rating": qb_rating,
            "grades_pass": pass_grade,
            "grades_offense": off_grade,
            "grades_run": run_grade,
            "Net EPA": epa,
        })
    return pd.DataFrame(rows).sort_values("Year")


def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    seasons = []
    yearly = _aggregate_qb_history_by_year(history)
    for _, row in yearly.iterrows():
        if not _has_valid_stats(row):
            continue
        year = int(_safe_float(row.get("Year"), 2024))
        max_g = _max_games_for_year(year)
        games = int(round(_infer_games_played(row, max_g)))
        dropbacks = max(1.0, _safe_float(row.get("dropbacks"), 1.0))

        seasons.append({
            "season":           year,
            "games_played":     int(games),
            "max_games":        int(max_g),
            "yards":            round(_safe_float(row.get("yards")), 0),
            "touchdowns":       round(_safe_float(row.get("touchdowns")), 0),
            "interceptions":    round(_qb_ints_from_row(row), 0),
            "ypa":              round(_safe_float(row.get("ypa")), 2),
            "qb_rating":        round(_safe_float(row.get("qb_rating")), 1),
            "completion_pct":   round(_safe_float(row.get("completion_percent")), 1),
            "btt_rate":         round(_safe_float(row.get("btt_rate")), 4),
            "big_time_throws":  round(_safe_float(row.get("big_time_throws")), 0),
            "epa":              round(_safe_float(row.get("Net EPA")), 2),
            "rushing_yards":    round(_safe_float(row.get("grades_run")), 1),
            "dropbacks":        round(dropbacks, 0),
            "pass_grade":       round(_safe_float(row.get("grades_pass")), 1),
            "overall_grade":    round(_safe_float(row.get("grades_offense")), 1),
        })
    return seasons


def extract_last_season_stats(history: pd.DataFrame) -> dict:
    sorted_hist = _aggregate_qb_history_by_year(history)
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist
    recent_rows = valid_rows.sort_values("Year").tail(3).reset_index(drop=True)

    row = valid_rows.iloc[-1]
    year = int(_safe_float(row.get("Year"), 2024))
    max_g = _max_games_for_year(year)
    games = float(max(1.0, min(max_g, _infer_games_played(row, max_g))))
    avail = round(games / max_g, 3)
    dropbacks = max(1.0, _safe_float(row.get("dropbacks"), 1.0))

    # Recent-weighted rates (last 3 seasons) so older prime years don't dominate.
    c_epa = c_dbs = c_yards = c_tds = c_ints = c_btts = c_games = 0.0
    c_cmp = c_att = c_sacks = 0.0
    recent_att_17_w = 0.0
    att_pg_vals: List[float] = []
    peak_dbs_17: List[float] = []
    proj_dbs_pace_w = 0.0
    n_recent = len(recent_rows)
    w_recent = _qb_recency_weights_n(n_recent)
    for (_, r), rw in zip(recent_rows.iterrows(), w_recent):
        yr = int(_safe_float(r.get("Year"), 2024))
        yr_g = _max_games_for_year(yr)
        g = _infer_games_played(r, yr_g)
        g = max(1.0, min(yr_g, g))
        db_raw = max(0.0, _safe_float(r.get("dropbacks"), 0.0))
        if db_raw >= _MIN_DB_FOR_STARTER_PEAK:
            peak_dbs_17.append(round((db_raw / max(g, 0.01)) * 17.0))
        proj_dbs_pace_w += (db_raw / g) * 17.0 * rw
        db = max(1.0, db_raw) if db_raw > 0 else 1.0
        c_epa   += _safe_float(r.get("Net EPA")) * rw
        c_dbs   += db * rw
        c_yards += _safe_float(r.get("yards")) * rw
        c_tds   += _safe_float(r.get("touchdowns")) * rw
        c_ints  += _qb_ints_from_row(r) * rw
        c_btts  += _safe_float(r.get("big_time_throws")) * rw
        c_cmp   += _safe_float(r.get("completions")) * rw
        c_att   += _safe_float(r.get("attempts")) * rw
        c_sacks += _safe_float(r.get("sacks")) * rw
        c_games += g * rw
        att_pg = _safe_float(r.get("attempts")) / max(g, 1.0)
        att_pg_vals.append(att_pg)
        recent_att_17_w += (att_pg * 17.0) * rw

    c_dbs  = max(c_dbs, 1.0)
    c_att  = max(c_att, 1.0)
    c_games = max(c_games, 1.0)

    career_epa_per_db  = c_epa / c_dbs
    career_ypa         = (c_yards / c_att) if c_att > 0 else _safe_float(row.get("ypa"))
    career_cmp_pct     = (c_cmp / c_att * 100) if c_att > 0 else _safe_float(row.get("completion_percent"))
    career_btt_rate    = (c_btts / c_dbs) if c_dbs > 0 else _safe_float(row.get("btt_rate"))
    career_sack_rate_db = (c_sacks / c_dbs) if c_dbs > 0 else 0.065
    td_rate_per_db = qb_stabilized_td_rate_per_db(c_tds, c_dbs)
    int_rate_per_db = qb_stabilized_int_rate_per_db(c_ints, c_dbs)
    att_trend_17 = 0.0
    if len(att_pg_vals) >= 2:
        att_trend_17 = (att_pg_vals[-1] - att_pg_vals[0]) * 17.0
    att_pg_sorted = sorted(att_pg_vals) if att_pg_vals else [max(1.0, c_att / max(c_games, 1.0))]
    m = len(att_pg_sorted)
    if m % 2 == 1:
        att_pg_med = att_pg_sorted[m // 2]
    else:
        att_pg_med = 0.5 * (att_pg_sorted[m // 2 - 1] + att_pg_sorted[m // 2])
    att_17_median = att_pg_med * 17.0
    # Recency-first attempt pace; small YoY momentum (oldest→newest in window).
    att_17_baseline = recent_att_17_w + 0.12 * att_trend_17
    att_17_baseline = max(180.0, min(680.0, att_17_baseline))

    proj_dbs_linear = max(1.0, proj_dbs_pace_w)
    peak_17_val = max(peak_dbs_17) if peak_dbs_17 else float(proj_dbs_linear)
    proj_dbs_starter_17g = round(
        qb_full_role_dropbacks_17(max(proj_dbs_linear, peak_17_val)), 1
    )

    out = {
        "season":         year,
        "games_played":   int(round(games)),
        "max_games":      int(max_g),
        "availability":   avail,
        "yards":          round(_safe_float(row.get("yards"))),
        "touchdowns":     round(_safe_float(row.get("touchdowns"))),
        "interceptions":  round(_qb_ints_from_row(row)),
        "big_time_throws":round(_safe_float(row.get("big_time_throws"))),
        "dropbacks":      round(dropbacks),
        "pass_grade":     round(_safe_float(row.get("grades_pass")), 1),
        "run_grade":      round(_safe_float(row.get("grades_run")), 1),
        # Career rates (used for stats grade)
        "qb_rating":      round(_safe_float(row.get("qb_rating")), 1),
        "ypa":            round(career_ypa, 2),
        "completion_pct": round(career_cmp_pct, 2),
        "btt_rate":       round(career_btt_rate, 4),
        "epa_per_db":     round(career_epa_per_db, 4),
        "sack_rate_db":   round(max(0.02, min(0.18, career_sack_rate_db)), 5),
        # Per-DB rates (8dp — avoids rounding to 0.0 and killing INT projections)
        "td_rate_per_db": round(td_rate_per_db, 8),
        "int_rate_per_db": round(int_rate_per_db, 8),
        "career_c_ints":  round(c_ints, 3),
        "career_c_tds":   round(c_tds, 3),
        "career_c_dbs":   round(c_dbs, 3),
        "proj_dbs_career_pace_17g": round(proj_dbs_linear, 1),
        "proj_dbs_peak_17g":        round(peak_17_val, 1),
        "proj_dbs_starter_17g":     proj_dbs_starter_17g,
        # "Typical" here = recency-weighted 17-game attempt pace (not median of seasons).
        "proj_atts_typical_17g":    round(recent_att_17_w, 1),
        "proj_atts_trend_17g":      round(att_17_baseline, 1),
        "proj_atts_median_17g":     round(att_17_median, 1),
    }
    _apply_qb_volume_from_rates(out, proj_dbs_starter_17g)
    return out


def _compute_health_factor(history: pd.DataFrame) -> tuple:
    recent = history.sort_values("Year").tail(3).reset_index(drop=True)
    n = len(recent)
    w = _qb_recency_weights_n(n)
    avail_list = []
    for _, row in recent.iterrows():
        yr = int(_safe_float(row.get("Year"), 2024))
        max_g = _max_games_for_year(yr)
        games = max(1.0, min(max_g, _infer_games_played(row, max_g)))
        avail_list.append(games / max_g)
    avg_avail = sum(a * wt for a, wt in zip(avail_list, w))
    adj = (avg_avail - 0.75) * 10.0
    adj = max(-5.0, min(2.5, adj))
    return round(adj, 2), round(avg_avail, 3)


def project_stats(
    last_stats: dict,
    composite_gr: float,
    current_age: int,
    contract_years: int,
    history: pd.DataFrame = None,
    grade_col: str = "grades_offense",
) -> List[dict]:
    projections = []
    grade = composite_gr
    player_yoy = player_recent_grade_yoy(history, grade_col)

    def _nfl_passer_rating(cmp_pct: float, ypa: float, td_rate_att: float, int_rate_att: float) -> float:
        a = max(0.0, min(2.375, (cmp_pct - 30.0) / 20.0))
        b = max(0.0, min(2.375, (ypa - 3.0) / 4.0))
        c = max(0.0, min(2.375, td_rate_att * 20.0))
        d = max(0.0, min(2.375, 2.375 - int_rate_att * 25.0))
        return ((a + b + c + d) / 6.0) * 100.0

    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        base_scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0
        trend_mult = projection_trend_multiplier("QB", age, yr, player_yoy)
        scale = max(0.25, min(1.8, base_scale * trend_mult))
        base_atts_17 = float(last_stats.get("proj_atts_17g") or 0.0)
        proj_atts_y = max(40.0, min(660.0, base_atts_17 * scale))
        base_ypa = float(last_stats.get("ypa") or 0.0)
        sack_rate_db = max(0.02, min(0.18, float(last_stats.get("sack_rate_db") or 0.065)))
        proj_dbs_y = max(1.0, proj_atts_y / max(1e-6, (1.0 - sack_rate_db)))
        # Turnovers: stabilized per-DB rates × load; grade decline vs composite raises INT risk
        int_rate = _effective_qb_int_rate(last_stats)
        td_rate = _effective_qb_td_rate(last_stats)
        gr_diff = float(composite_gr) - float(grade)
        risk_adj = max(0.68, min(1.38, 1.0 + gr_diff * 0.019))
        proj_ints = round(min(35.0, int_rate * proj_dbs_y * risk_adj), 1)
        td_eff = max(0.82, min(1.15, 1.0 + (grade - composite_gr) * 0.009))
        proj_tds = round(min(60.0, td_rate * proj_dbs_y * td_eff), 1)
        proj_atts = proj_atts_y
        proj_cmp_pct = max(52.0, min(78.0, float(last_stats["completion_pct"]) * (0.55 + 0.45 * scale)))
        proj_cmp = max(1.0, proj_atts * (proj_cmp_pct / 100.0))
        proj_ypa = max(4.8, min(9.8, base_ypa * (0.78 + 0.22 * scale)))
        proj_yards = round(min(5800.0, proj_atts * proj_ypa))
        td_rate_att = float(proj_tds) / max(proj_atts, 1.0)
        int_rate_att = float(proj_ints) / max(proj_atts, 1.0)
        proj_qbr = round(min(125.0, _nfl_passer_rating(proj_cmp_pct, proj_ypa, td_rate_att, int_rate_att)), 1)
        # Soft caps: scale with attempt volume so full-season starters are not pinned ~2500 yds.
        att_scale = max(0.88, min(1.12, float(proj_atts) / 485.0))
        if proj_qbr < 85.0:
            proj_yards = min(proj_yards, int(4100 * att_scale))
        elif proj_qbr < 90.0:
            proj_yards = min(proj_yards, int(4550 * att_scale))
        elif proj_qbr < 95.0:
            proj_yards = min(proj_yards, int(5050 * att_scale))
        projections.append({
            "year":           yr,
            "age":            age,
            "projected_grade": round(grade, 1),
            "attempts":       round(proj_atts),
            "completions":    round(proj_cmp),
            "sacks":          round(max(0.0, proj_dbs_y - proj_atts)),
            "yards":          proj_yards,
            "touchdowns":     proj_tds,
            "interceptions":  proj_ints,
            "ypa":            round(proj_ypa, 2),
            "qb_rating":      proj_qbr,
            "btt_rate":       round(min(0.12, last_stats["btt_rate"] * scale), 4),
            "completion_pct": round(proj_cmp_pct, 1),
            "pass_grade":     round(min(99, last_stats["pass_grade"] * scale), 1),
        })
    return projections


DISCOUNT_RATE   = 0.08
CAP_GROWTH_RATE = 0.065


def compute_contract_value(
    composite_gr,
    current_age,
    contract_years,
    salary_ask,
    history: pd.DataFrame = None,
    grade_col: str = "grades_offense",
    analysis_year: int = 2026,
):
    breakdown = []
    total_disc_value = total_disc_ask = total_nominal_value = 0.0
    weighted_fair_num = weighted_burden_num = weight_den = 0.0
    grade = float(composite_gr)
    player_yoy = player_recent_grade_yoy(history, grade_col)
    snap_rel, _ = snap_value_reliability_factor(history)
    cap_scale = cap_scale_for_year(analysis_year)
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        cap_factor    = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))
        base_value    = grade_to_market_value(grade) * snap_rel * cap_scale
        nominal_value = base_value * cap_factor
        disc_value    = nominal_value * time_discount
        cap_adj_ask   = salary_ask / cap_factor
        disc_ask      = cap_adj_ask * time_discount
        year_surplus  = round(base_value - cap_adj_ask, 2)
        # Front-weight contract value: earlier years matter more in fair-AAV decisions.
        front_weight = 1.0 / float(yr)
        total_nominal_value += nominal_value
        total_disc_value    += disc_value
        total_disc_ask      += disc_ask
        weighted_fair_num   += nominal_value * front_weight
        weighted_burden_num += cap_adj_ask * front_weight
        weight_den          += front_weight
        breakdown.append({
            "year": yr, "age": age,
            "projected_grade": round(grade, 1),
            "market_value":    base_value,
            "nominal_value":   round(nominal_value, 2),
            "cap_adj_ask":     round(cap_adj_ask, 2),
            "discounted_value":round(disc_value, 2),
            "year_surplus":    year_surplus,
        })
    fair_aav             = round(weighted_fair_num / max(weight_den, 1e-6), 2)
    effective_cap_burden = round(weighted_burden_num / max(weight_den, 1e-6), 2)
    return fair_aav, effective_cap_burden, round(total_nominal_value, 2), breakdown


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────
class QBAgentState(TypedDict):
    player_name:     str
    salary_ask:      float
    contract_years:  int
    player_history:      pd.DataFrame
    player_history_full: pd.DataFrame
    analysis_year:       int
    predicted_tier:    str
    confidence:        Dict[str, float]
    current_age:       int
    last_season_stats: dict
    career_stats:      List[dict]
    stats_score:       float
    composite_grade:   float
    valuation:            float
    effective_cap_burden: float
    total_nominal_value:  float
    year_breakdown:       List[dict]
    projected_stats:      List[dict]
    team_name:              str
    team_cap_available_pct: float
    positional_need:        float
    need_label:             str
    current_roster:         List[dict]
    signing_cap_pcts:       List[float]
    team_fit_summary:       str
    projected_signing_role:         str
    projected_signing_role_reason:  str
    decision:  str
    reasoning: str


def predict_performance(state: QBAgentState):
    print(f"[QB Agent] Predicting for {state['player_name']}...")
    history = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)

    resolved_age = resolve_player_age_for_evaluation(
        state.get("player_history_full"), history, analysis_year=current_year
    )
    if resolved_age is not None:
        current_age = resolved_age
    elif "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        age_at_last = int(float(last_row["age"]))
        last_yr = int(float(last_row["Year"]))
        current_age = age_at_last + (current_year - last_yr)
    else:
        current_age = 28

    last_stats   = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)
    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, inactivity_meta = inactivity_retirement_penalty(history, current_year=current_year)

    # Model grade = recent-weighted (last 3) pass grade with latest-season emphasis.
    valid_rows = _aggregate_qb_history_by_year(history).sort_values("Year")
    recent = valid_rows.tail(3).reset_index(drop=True)
    rec_w = _qb_recency_weights_n(len(recent))
    mg_num = mg_den = 0.0
    recent_dbs = 0.0
    for (_, r), rw in zip(recent.iterrows(), rec_w):
        pg = _safe_float(r.get("grades_pass") or r.get("grades_offense"), 60.0)
        db = max(1.0, _safe_float(r.get("dropbacks"), 1.0))
        mg_num += pg * db * rw
        mg_den += db * rw
        recent_dbs += db
    model_grade = (mg_num / mg_den) if mg_den > 0 else _safe_float(valid_rows.iloc[-1].get("grades_pass") or valid_rows.iloc[-1].get("grades_offense"), 60.0)

    # Team context: starter / fringe / backup → role-based dropback load (counts + INTs scale to role)
    team_nm = (state.get("team_name") or "").strip()
    roster = state.get("current_roster") or []
    need = float(state.get("positional_need") or 0.0)
    role = "starter"
    role_reason = (
        "No team selected — projecting full-time starter volume (typical FA baseline)."
    )
    if team_nm:
        role, role_reason = infer_qb_signing_role(model_grade, roster, need)
        load = qb_dropbacks_17_for_role(
            role,
            float(last_stats["proj_dbs_career_pace_17g"]),
            float(last_stats.get("proj_dbs_peak_17g") or 0.0),
        )
        _apply_qb_volume_from_rates(last_stats, load)
        last_stats["projected_signing_role"] = role
        last_stats["projected_signing_role_reason"] = role_reason
    else:
        last_stats["projected_signing_role"] = role
        last_stats["projected_signing_role_reason"] = role_reason

    # Stats grade
    sg = _stats_grade(
        last_stats["qb_rating"],
        last_stats["ypa"],
        last_stats["btt_rate"],
        last_stats["completion_pct"],
        last_stats["epa_per_db"],
    )

    raw_cg = _composite_grade(model_grade, sg)
    # Reliability adjustment: tiny recent samples should not produce elite outcomes.
    sample_reliability = max(0.45, min(1.00, recent_dbs / 900.0))
    cg_pre = raw_cg + health_adj + inactivity_adj
    cg = 60.0 + (cg_pre - 60.0) * sample_reliability
    cg = round(max(45.0, min(99.0, cg)), 2)

    # Volume reliability: very small recent samples should not project full starter volume by default.
    yrs_out = float(inactivity_meta.get("years_since_last_season", 0))
    gap_rel = max(0.15, 1.0 - 0.22 * max(0.0, yrs_out - 1.0))
    vol_rel = max(0.20, min(1.00, (recent_dbs / 820.0) * gap_rel))
    base_proj_dbs = float(last_stats.get("proj_dbs_17g") or 0.0)
    _apply_qb_volume_from_rates(last_stats, base_proj_dbs * vol_rel)
    trend_atts = float(last_stats.get("proj_atts_trend_17g") or 0.0)
    typical_atts = float(last_stats.get("proj_atts_typical_17g") or trend_atts)
    if trend_atts > 0:
        sack_rate = max(0.02, min(0.18, float(last_stats.get("sack_rate_db") or 0.065)))
        base_atts = float(last_stats.get("proj_atts_17g") or 0.0)
        # Favor recency + trend anchor over a separate "typical" mix (typical is now recency-weighted too).
        trend_component = 0.82 * trend_atts + 0.18 * typical_atts
        blended_atts = 0.40 * base_atts + 0.60 * (trend_component * vol_rel)
        blended_dbs = blended_atts / max(1e-6, (1.0 - sack_rate))
        _apply_qb_volume_from_rates(last_stats, blended_dbs)

    return {
        "predicted_tier":    _grade_to_tier(cg),
        "current_age":       current_age,
        "last_season_stats": last_stats,
        "career_stats":      career_stats,
        "stats_score":       sg,
        "composite_grade":   cg,
        "confidence": {
            "model_grade":      round(model_grade, 2),
            "stats_grade":      sg,
            "composite_grade":  cg,
            "health_factor":    health_adj,
            "inactivity_penalty": inactivity_adj,
            "years_since_last_season": inactivity_meta.get("years_since_last_season", 0),
            "recent_games_3y": inactivity_meta.get("recent_games_3y", 0.0),
            "avg_availability": avg_avail,
            "recent_dropbacks_3y": round(recent_dbs, 1),
            "sample_reliability": round(sample_reliability, 3),
            "volume_reliability": round(vol_rel, 3),
            "gap_reliability": round(gap_rel, 3),
            "projected_signing_role": role,
            "projected_signing_role_reason": role_reason,
        },
        "projected_signing_role": role,
        "projected_signing_role_reason": role_reason,
    }


def evaluate_value(state: QBAgentState):
    hist = state.get("player_history")
    ay = int(state.get("analysis_year") or 2026)
    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        state["salary_ask"],
        history=hist,
        grade_col="grades_offense",
        analysis_year=ay,
    )
    stat_proj = project_stats(
        state["last_season_stats"],
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        history=hist,
        grade_col="grades_offense",
    )
    inact_pen = float((state.get("confidence") or {}).get("inactivity_penalty", 0.0))
    stat_proj = apply_inactivity_to_projection_list(stat_proj, inact_pen)
    stat_proj = apply_projection_plausibility_caps(stat_proj, state.get("career_stats") or [])
    return {
        "valuation":            fair_aav,
        "effective_cap_burden": eff_burden,
        "total_nominal_value":  total_nom,
        "year_breakdown":       breakdown,
        "projected_stats":      stat_proj,
    }


def assess_team_fit(state: QBAgentState):
    team = state.get("team_name", "")
    if not team:
        return {}
    cap_pcts = aav_to_cap_pcts(
        state["salary_ask"],
        state["contract_years"],
        int(state.get("analysis_year") or 2025),
    )
    return {"signing_cap_pcts": cap_pcts}


def make_decision(state: QBAgentState):
    ask    = state["salary_ask"]
    val    = state["valuation"]
    burden = state["effective_cap_burden"]
    tier   = state["predicted_tier"]
    cg     = state["composite_grade"]
    mg     = state["confidence"].get("model_grade", cg)
    sg     = state["stats_score"]
    age    = state["current_age"]
    years  = state["contract_years"]
    total  = state["total_nominal_value"]

    health_adj = state["confidence"].get("health_factor", 0)
    avg_avail  = state["confidence"].get("avg_availability", 1.0)
    health_str = (
        f" Health: {'+' if health_adj >= 0 else ''}{health_adj} pts "
        f"({round(avg_avail * 100)}% availability over recent seasons)."
    )

    if age <= 26:   trajectory = "still developing"
    elif age <= 30: trajectory = "in his prime"
    elif age <= 33: trajectory = "entering post-prime"
    else:           trajectory = "in steep age-related decline"

    total_ask = round(ask * years, 2)
    cap_note  = (
        f"With ~{int(CAP_GROWTH_RATE*100)}%/yr cap growth the fixed ${ask}M AAV "
        f"costs effectively ${burden}M/yr in present-value terms."
    )

    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_for_decision = val
    rep_note = ""
    if team_nm and roster:
        _scale = cap_scale_for_year(int(state.get("analysis_year") or 2026))
        val_for_decision, rep_note = decision_fair_aav_with_replacement(
            val, lambda g: grade_to_market_value(g) * _scale, cg, roster, "QB",
        )

    surplus = round(val - burden, 2)
    surplus_pct = (val - burden) / max(val, 0.01) * 100

    if surplus_pct >= 20:    decision = "Exceptional Value";  verdict_str = f"surplus of ${surplus}M/yr — a steal."; rec = "Strongly recommend signing."
    elif surplus_pct >= 5:   decision = "Good Signing";       verdict_str = f"surplus of ${surplus}M/yr — good value."; rec = "Recommend signing."
    elif surplus_pct >= -5:  decision = "Fair Deal";          verdict_str = f"${abs(surplus)}M/yr {'surplus' if surplus >= 0 else 'overpay'} — roughly market rate."; rec = "Acceptable signing."
    elif surplus_pct >= -15: decision = "Slight Overpay";     verdict_str = f"overpay of ${abs(surplus)}M/yr — modest premium."; rec = "Proceed with caution."
    elif surplus_pct >= -30: decision = "Overpay";            verdict_str = f"overpay of ${abs(surplus)}M/yr — significant premium."; rec = "Recommend passing unless positional need is critical."
    else:                    decision = "Poor Signing";        verdict_str = f"overpay of ${abs(surplus)}M/yr — severely overpriced."; rec = "Strongly recommend passing."

    reason = (
        f"{state['player_name']} (age {age}) projects as a {tier} quarterback. "
        f"PFF pass grade: {mg:.1f} · Stats grade: {sg:.1f} → Composite: {cg:.1f}.{health_str} "
        f"He is {trajectory}. Over a {years}-yr contract the composite projects "
        f"a cap-inflation-adjusted fair value of ${val}M/yr vs. "
        f"an effective cap burden of ${burden}M/yr — {verdict_str} "
        f"{cap_note} Total nominal player value: ${total}M vs. total ask: ${total_ask}M. {rec}"
    )
    if rep_note:
        reason = reason + rep_note

    team = state.get("team_name", "")
    fit_summary = ""
    if team:
        need_score = state.get("positional_need", 50)
        need_lbl   = state.get("need_label", "Average")
        cap_pcts   = state.get("signing_cap_pcts", [])
        avail_pct  = state.get("team_cap_available_pct", 100)
        roster     = state.get("current_roster", [])
        adjusted_decision, fit_summary, team_reason = _assess_team_fit_logic(
            base_decision=decision, surplus_pct=surplus_pct,
            need_score=need_score, need_label=need_lbl,
            signing_cap_pcts=cap_pcts, available_cap_pct=avail_pct,
            roster=roster, player_name=state["player_name"],
        )
        decision = adjusted_decision
        reason = reason + " " + team_reason

    return {"decision": decision, "reasoning": reason, "team_fit_summary": fit_summary}


_workflow = StateGraph(QBAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
qb_gm_agent = _workflow.compile()
