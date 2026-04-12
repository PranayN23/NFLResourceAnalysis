"""
HB/RB GM Agent — Running Back Free Agency Evaluator

Stats grade weighted by: yco_attempt 30%, ypc 25%, elusive_rating 20%,
receptions_per_game 15%, epa_per_touch 10%.
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
    offense_target_load_17,
    inactivity_retirement_penalty,
    apply_inactivity_to_projection_list,
    apply_projection_plausibility_caps,
    shrink_model_grade_for_season_snap_volume,
    snap_value_reliability_factor,
)
import pandas as pd
import numpy as np
import os, datetime

from backend.agent.api_year_utils import resolve_player_age_for_evaluation

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HB_CSV_PATH = os.path.join(_BASE, "ML", "HB.csv")

_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [1.14, 2.27, 6.82, 9.66, 10.80, 11.93, 17.05, 19.32, 20.45, 21.59, 22.73, 23.41]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# Stats anchors: empirical league medians for HBs
_YCO_ANCHORS = [0.0, 2.0, 2.5, 3.0, 3.5, 5.0]
_YPC_ANCHORS = [0.0, 3.5, 4.0, 4.5, 5.0, 6.5]
_ELU_ANCHORS = [0.0, 30.0, 45.0, 60.0, 75.0, 100.0]
_REC_ANCHORS = [0.0, 1.0,  2.0,  3.5,  4.5,  6.5]   # receptions per game
_EPA_ANCHORS = [-0.4, -0.10, 0.00, 0.05, 0.10, 0.20]  # epa per touch
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(yco_attempt, ypc, elusive_rating, rec_per_game, epa_per_touch):
    yc = float(np.interp(yco_attempt,  _YCO_ANCHORS, _STAT_GRD_SCALE))
    yp = float(np.interp(ypc,          _YPC_ANCHORS, _STAT_GRD_SCALE))
    el = float(np.interp(elusive_rating, _ELU_ANCHORS, _STAT_GRD_SCALE))
    rc = float(np.interp(rec_per_game, _REC_ANCHORS, _STAT_GRD_SCALE))
    ep = float(np.interp(epa_per_touch, _EPA_ANCHORS, _STAT_GRD_SCALE))
    return round(0.30 * yc + 0.25 * yp + 0.20 * el + 0.15 * rc + 0.10 * ep, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.45 * model_grade + 0.55 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# HB/RB: median YoY grades_offense change on consecutive ML/HB.csv seasons (key = age at start of transition).
# Regenerate: backend/agent/compute_position_age_curves.py
_AGE_DELTAS = {
    20: -0.8, 21: -0.8, 22: +1.8, 23: -0.9, 24: -1.3, 25: -1.1, 26: -1.6,
    27: +0.2, 28: -4.6, 29: -4.3, 30: -2.7, 31: -1.1, 32: +0.2, 33: -4.0,
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


def _has_valid_stats(row):
    for col in ("yards", "attempts", "grades_offense", "total_touches"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f):
                return True
        except Exception:
            pass
    return False


def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row):
            continue
        year = int(_safe_float(row.get("Year"), 2024))
        max_g = 17.0 if year >= 2021 else 16.0
        games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        att = max(1.0, _safe_float(row.get("attempts"), 1.0))
        seasons.append({
            "season":          year,
            "games_played":    int(games),
            "max_games":       int(max_g),
            "yards":           round(_safe_float(row.get("yards"))),
            "attempts":        round(att),
            "ypc":             round(_safe_float(row.get("yards")) / att, 2),
            "yco_attempt":     round(_safe_float(row.get("yco_attempt")), 2),
            "touchdowns":      round(_safe_float(row.get("touchdowns"))),
            "receptions":      round(_safe_float(row.get("receptions"))),
            "rec_yards":       round(_safe_float(row.get("rec_yards"))),
            "elusive_rating":  round(_safe_float(row.get("elusive_rating")), 1),
            "broken_tackles":  round(_safe_float(row.get("avoided_tackles"))),
            "epa":             round(_safe_float(row.get("Net EPA")), 2),
            "run_grade":       round(_safe_float(row.get("grades_run")), 1),
            "overall_grade":   round(_safe_float(row.get("grades_offense")), 1),
        })
    return seasons


def extract_last_season_stats(history: pd.DataFrame) -> dict:
    sorted_hist = history.sort_values("Year")
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty:
        valid_rows = sorted_hist
    row = valid_rows.iloc[-1]
    year = int(_safe_float(row.get("Year"), 2024))
    max_g = 17.0 if year >= 2021 else 16.0
    games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
    avail = round(games / max_g, 3)

    # Career-weighted rates
    c_yards = c_att = c_rec = c_tgts = c_rec_yds = c_tds = c_touches = c_btts = 0.0
    c_yco = c_elu = c_epa = c_games = 0.0
    for _, r in valid_rows.iterrows():
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        c_yards   += _safe_float(r.get("yards"))
        c_att     += _safe_float(r.get("attempts"))
        c_rec     += _safe_float(r.get("receptions"))
        c_tgts    += _safe_float(r.get("targets"))
        c_rec_yds += _safe_float(r.get("rec_yards"))
        c_tds     += _safe_float(r.get("touchdowns"))
        c_touches += _safe_float(r.get("total_touches"))
        c_btts    += _safe_float(r.get("avoided_tackles"))
        c_yco     += _safe_float(r.get("yco_attempt")) * max(1, _safe_float(r.get("attempts")))
        c_elu     += _safe_float(r.get("elusive_rating"))
        c_epa     += _safe_float(r.get("Net EPA"))
        c_games   += g

    c_att   = max(c_att, 1.0)
    c_games = max(c_games, 1.0)
    c_touches = max(c_touches, 1.0)

    career_ypc       = c_yards / c_att
    career_yco       = c_yco / c_att
    career_elu       = c_elu / max(len(valid_rows), 1)
    career_rec_pg    = c_rec / c_games
    career_epa_touch = c_epa / c_touches
    c_tgts = max(c_tgts, 1.0)
    proj_att_17g     = max(round(c_att / c_games * 17), 188)
    tgt_17 = offense_target_load_17(c_tgts, c_games, floor_17=42.0, max_17=98.0)
    proj_rec_17g     = max(round(c_rec / c_games * 17), round(tgt_17 * 0.82))

    return {
        "season":         year,
        "games_played":   int(games),
        "max_games":      int(max_g),
        "availability":   avail,
        "yards":          round(_safe_float(row.get("yards"))),
        "attempts":       round(_safe_float(row.get("attempts"))),
        "touchdowns":     round(_safe_float(row.get("touchdowns"))),
        "receptions":     round(_safe_float(row.get("receptions"))),
        "rec_yards":      round(_safe_float(row.get("rec_yards"))),
        "broken_tackles": round(_safe_float(row.get("avoided_tackles"))),
        "run_grade":      round(_safe_float(row.get("grades_run")), 1),
        # Career rates
        "ypc":            round(career_ypc, 2),
        "yco_attempt":    round(career_yco, 2),
        "elusive_rating": round(career_elu, 1),
        "rec_per_game":   round(career_rec_pg, 2),
        "epa_per_touch":  round(career_epa_touch, 4),
        # 17g projections
        "yards_17g":      round(career_ypc * proj_att_17g),
        "tds_17g":        round(c_tds / c_games * 17, 1),
        "rec_17g":        proj_rec_17g,
        "rec_yards_17g":  round(c_rec_yds / c_games * 17),
        "proj_att_17g":   proj_att_17g,
        "btts_17g":       round(c_btts / c_games * 17, 1),
        "epa_17g":        round(c_epa / c_games * 17, 2),
    }


def _compute_health_factor(history: pd.DataFrame) -> tuple:
    recent = history.sort_values("Year").tail(3).reset_index(drop=True)
    weights = [0.20, 0.30, 0.50]
    n = len(recent)
    avail_list = []
    for _, row in recent.iterrows():
        yr = int(_safe_float(row.get("Year"), 2024))
        max_g = 17.0 if yr >= 2021 else 16.0
        games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        avail_list.append(games / max_g)
    w = weights[-n:]
    w = [x / sum(w) for x in w]
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
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        base_scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0
        trend_mult = projection_trend_multiplier("HB", age, yr, player_yoy)
        scale = max(0.25, min(1.8, base_scale * trend_mult))
        projections.append({
            "year":           yr,
            "age":            age,
            "projected_grade": round(grade, 1),
            "yards":          round(min(2500, last_stats["yards_17g"] * scale)),
            "attempts":       round(min(350, last_stats["proj_att_17g"] * scale)),
            "touchdowns":     round(min(25, last_stats["tds_17g"] * scale), 1),
            "receptions":     round(min(100, last_stats["rec_17g"] * scale)),
            "rec_yards":      round(min(1200, last_stats["rec_yards_17g"] * scale)),
            "ypc":            round(min(8, last_stats["ypc"] * scale), 2),
            "yco_attempt":    round(min(6, last_stats["yco_attempt"]), 2),
            "elusive_rating": round(min(100, last_stats["elusive_rating"] * scale), 1),
            "broken_tackles": round(min(60, last_stats["btts_17g"] * scale), 1),
            "epa":            round(last_stats["epa_17g"] * scale, 2),
            "run_grade":      round(min(99, last_stats["run_grade"] * scale), 1),
            "overall_grade":  round(min(99, grade), 1),
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
        front_weight  = 1.0 / float(yr)
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
            "year_surplus":    round(base_value - cap_adj_ask, 2),
        })
    fair_aav             = round(weighted_fair_num / max(weight_den, 1e-6), 2)
    effective_cap_burden = round(weighted_burden_num / max(weight_den, 1e-6), 2)
    return fair_aav, effective_cap_burden, round(total_nominal_value, 2), breakdown


class HBAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    player_history_full: pd.DataFrame; analysis_year: int
    predicted_tier: str; projected_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: HBAgentState):
    print(f"[HB Agent] Predicting for {state['player_name']}...")
    history = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    resolved_age = resolve_player_age_for_evaluation(state.get("player_history_full"), history, analysis_year=current_year)
    if resolved_age is not None:
        current_age = resolved_age
    elif "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        current_age = int(float(last_row["age"])) + (current_year - int(float(last_row["Year"])))
    else:
        current_age = 26

    last_stats   = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)
    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)

    raw_mg = _safe_float(history.sort_values("Year").iloc[-1].get("grades_offense"), 60.0)
    model_grade, snap_m = shrink_model_grade_for_season_snap_volume(
        raw_mg,
        history,
        grade_col="grades_offense",
        snap_profile=[
            ("snap_counts_offense", 515.0),
            ("total_snaps", 500.0),
            ("routes", 380.0),
        ],
    )

    sg = _stats_grade(
        last_stats["yco_attempt"], last_stats["ypc"],
        last_stats["elusive_rating"], last_stats["rec_per_game"],
        last_stats["epa_per_touch"],
    )
    raw_cg = _composite_grade(model_grade, sg)
    cg = round(max(45.0, min(99.0, raw_cg + health_adj + inactivity_adj)), 2)

    return {
        "predicted_tier": _grade_to_tier(cg), "current_age": current_age,
        "last_season_stats": last_stats, "career_stats": career_stats,
        "stats_score": sg, "composite_grade": cg,
        "confidence": {
            "model_grade": round(model_grade, 2),
            "model_grade_pre_snap_volume": round(raw_mg, 2),
            "stats_grade": sg,
            "composite_grade": cg,
            "health_factor": health_adj,
            "inactivity_penalty": inactivity_adj,
            "avg_availability": avg_avail,
            "snap_volume_stress": snap_m.get("snap_volume_stress", 1.0),
            "prior_full_snap_season": snap_m.get("prior_full_snap_season", False),
        },
    }


def evaluate_value(state: HBAgentState):
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
    return {"valuation": fair_aav, "effective_cap_burden": eff_burden,
            "total_nominal_value": total_nom, "year_breakdown": breakdown, "projected_stats": stat_proj}


def assess_team_fit(state: HBAgentState):
    if not state.get("team_name"):
        return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


def make_decision(state: HBAgentState):
    ask = state["salary_ask"]; val = state["valuation"]; burden = state["effective_cap_burden"]
    cg = state["composite_grade"]
    _yb = state.get("year_breakdown") or []
    if _yb:
        _avg_pg = sum(y.get("projected_grade", cg) for y in _yb) / len(_yb)
        tier = _grade_to_tier(_avg_pg)
    else:
        tier = state["predicted_tier"]
    mg = state["confidence"].get("model_grade", cg); sg = state["stats_score"]
    age = state["current_age"]; years = state["contract_years"]; total = state["total_nominal_value"]
    health_adj = state["confidence"].get("health_factor", 0)
    avg_avail  = state["confidence"].get("avg_availability", 1.0)
    health_str = f" Health: {'+' if health_adj >= 0 else ''}{health_adj} pts ({round(avg_avail*100)}% availability)."
    trajectory = ("still developing" if age <= 24 else "in his prime" if age <= 27 else
                  "entering decline" if age <= 29 else "in steep age-related decline")
    total_ask = round(ask * years, 2)
    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        _scale = cap_scale_for_year(int(state.get("analysis_year") or 2026))
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, lambda g: grade_to_market_value(g) * _scale, cg, roster, "HB",
        )
    surplus = round(val - burden, 2)
    surplus_pct = (val - burden) / max(val, 0.01) * 100
    if surplus_pct >= 20:    decision = "Exceptional Value"; rec = "Strongly recommend signing."
    elif surplus_pct >= 5:   decision = "Good Signing";      rec = "Recommend signing."
    elif surplus_pct >= -5:  decision = "Fair Deal";         rec = "Acceptable signing."
    elif surplus_pct >= -15: decision = "Slight Overpay";    rec = "Proceed with caution."
    elif surplus_pct >= -30: decision = "Overpay";           rec = "Recommend passing."
    else:                    decision = "Poor Signing";      rec = "Strongly recommend passing."

    reason = (
        f"{state['player_name']} (age {age}) projects as a {tier} running back. "
        f"PFF run grade: {mg:.1f} · Stats grade: {sg:.1f} → Composite: {cg:.1f}.{health_str} "
        f"He is {trajectory}. RB value: ${val}M/yr vs. ${burden}M/yr effective cap burden "
        f"(surplus: ${surplus}M/yr, {surplus_pct:.0f}%). Total ask: ${total_ask}M. {rec}"
    )
    if rep_note:
        reason = reason + rep_note
    fit_summary = ""
    if state.get("team_name"):
        adjusted_decision, fit_summary, team_reason = _assess_team_fit_logic(
            base_decision=decision, surplus_pct=surplus_pct,
            need_score=state.get("positional_need", 50), need_label=state.get("need_label", "Average"),
            signing_cap_pcts=state.get("signing_cap_pcts", []),
            available_cap_pct=state.get("team_cap_available_pct", 100),
            roster=state.get("current_roster", []), player_name=state["player_name"],
        )
        decision = adjusted_decision; reason = reason + " " + team_reason
    return {"decision": decision, "reasoning": reason, "team_fit_summary": fit_summary, "projected_tier": tier}


_workflow = StateGraph(HBAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
hb_gm_agent = _workflow.compile()
