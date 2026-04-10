"""
LB GM Agent — Linebacker Free Agency Evaluator

Stats grade weighted by: coverage_grade 30%, run_def_grade 25%,
epa_per_snap 20%, stop_rate 15%, tackle_grade 10%.
No touchdowns included (high variance). Fumble recoveries included.
"""

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from backend.agent.team_context import (
    assess_team_fit as _assess_team_fit_logic,
    aav_to_cap_pcts,
    decision_fair_aav_with_replacement,
)
from backend.agent.grade_projection import (
    grade_to_tier_universal,
    player_recent_grade_yoy,
    apply_yearly_grade_step,
)
from backend.agent.stat_projection_utils import defense_snap_load_17, inactivity_retirement_penalty, apply_inactivity_to_projection_list, apply_projection_plausibility_caps, snap_value_reliability_factor
import pandas as pd
import numpy as np
import os, datetime

from backend.agent.api_year_utils import resolve_player_age_for_evaluation

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LB_CSV_PATH = os.path.join(_BASE, "ML", "LB.csv")

_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [1.59, 3.41, 9.66, 13.92, 15.34, 16.48, 23.86, 25.68, 26.70, 28.18, 29.32, 30.11]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# Stats anchors
_COV_ANCHORS  = [0.0, 50.0, 60.0, 70.0, 80.0, 92.0]   # coverage grade
_RD_ANCHORS   = [0.0, 50.0, 60.0, 70.0, 80.0, 92.0]   # run defense grade
_EPA_ANCHORS  = [-0.15, -0.05, 0.0, 0.05, 0.10, 0.15] # epa per snap
_STOP_ANCHORS = [0.0, 0.05, 0.08, 0.12, 0.16, 0.25]   # stops per snap
_TAC_ANCHORS  = [0.0, 50.0, 60.0, 70.0, 80.0, 92.0]   # tackle grade
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(coverage_grade, run_def_grade, epa_per_snap, stop_rate, tackle_grade):
    cg = float(np.interp(coverage_grade, _COV_ANCHORS, _STAT_GRD_SCALE))
    rd = float(np.interp(run_def_grade,  _RD_ANCHORS,  _STAT_GRD_SCALE))
    ep = float(np.interp(epa_per_snap,   _EPA_ANCHORS, _STAT_GRD_SCALE))
    st = float(np.interp(stop_rate,      _STOP_ANCHORS,_STAT_GRD_SCALE))
    tg = float(np.interp(tackle_grade,   _TAC_ANCHORS, _STAT_GRD_SCALE))
    return round(0.30 * cg + 0.25 * rd + 0.20 * ep + 0.15 * st + 0.10 * tg, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.40 * model_grade + 0.60 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# LB-specific: median YoY grades_defense on consecutive ML/LB.csv seasons.
_AGE_DELTAS = {
    20: -1.8, 21: -1.8, 22: +1.4, 23: +0.8, 24: -1.1, 25: -2.3, 26: +0.1,
    27: -4.1, 28: -1.5, 29: -2.1, 30: -0.2, 31: -0.4, 32: -9.3, 33: -5.4,
    34: -4.0,
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
    for col in ("grades_defense", "tackles", "snap_counts_defense", "stops"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f): return True
        except Exception: pass
    return False


def extract_career_stats(history: pd.DataFrame) -> List[dict]:
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row): continue
        year = int(_safe_float(row.get("Year"), 2024))
        max_g = 17.0 if year >= 2021 else 16.0
        games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        snaps = max(1.0, _safe_float(row.get("snap_counts_defense"), 1.0))
        seasons.append({
            "season":           year,
            "games_played":     int(games),
            "max_games":        int(max_g),
            "tackles":          round(_safe_float(row.get("tackles"))),
            "assists":          round(_safe_float(row.get("assists"))),
            "tfl":              round(_safe_float(row.get("tackles_for_loss")), 1),
            "stops":            round(_safe_float(row.get("stops"))),
            "sacks":            round(_safe_float(row.get("sacks")), 1),
            "interceptions":    round(_safe_float(row.get("interceptions"))),
            "pass_breakups":    round(_safe_float(row.get("pass_break_ups"))),
            "fumble_recoveries":round(_safe_float(row.get("fumble_recoveries"))),
            "forced_fumbles":   round(_safe_float(row.get("forced_fumbles"))),
            "epa":              round(_safe_float(row.get("Net EPA")), 2),
            "coverage_grade":   round(_safe_float(row.get("grades_coverage_defense")), 1),
            "run_def_grade":    round(_safe_float(row.get("grades_run_defense")), 1),
            "tackle_grade":     round(_safe_float(row.get("grades_tackle")), 1),
            "overall_grade":    round(_safe_float(row.get("grades_defense")), 1),
        })
    return seasons


def extract_last_season_stats(history: pd.DataFrame) -> dict:
    sorted_hist = history.sort_values("Year")
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty: valid_rows = sorted_hist
    row = valid_rows.iloc[-1]
    year = int(_safe_float(row.get("Year"), 2024))
    max_g = 17.0 if year >= 2021 else 16.0
    games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
    avail = round(games / max_g, 3)

    c_tackles = c_stops = c_tfl = c_ints = c_pbus = c_epa = c_snaps = c_games = 0.0
    for _, r in valid_rows.iterrows():
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        s = max(1.0, _safe_float(r.get("snap_counts_defense"), 1.0))
        c_tackles += _safe_float(r.get("tackles"))
        c_stops   += _safe_float(r.get("stops"))
        c_tfl     += _safe_float(r.get("tackles_for_loss"))
        c_ints    += _safe_float(r.get("interceptions"))
        c_pbus    += _safe_float(r.get("pass_break_ups"))
        c_epa     += _safe_float(r.get("Net EPA"))
        c_snaps   += s
        c_games   += g

    c_snaps = max(c_snaps, 1.0); c_games = max(c_games, 1.0)
    proj_snaps_17g = round(defense_snap_load_17(c_snaps, c_games))
    int_per_snap = c_ints / c_snaps
    pbu_per_snap = c_pbus / c_snaps

    return {
        "season":           year,
        "games_played":     int(games),
        "max_games":        int(max_g),
        "availability":     avail,
        "tackles":          round(_safe_float(row.get("tackles"))),
        "assists":          round(_safe_float(row.get("assists"))),
        "tfl":              round(_safe_float(row.get("tackles_for_loss")), 1),
        "stops":            round(_safe_float(row.get("stops"))),
        "sacks":            round(_safe_float(row.get("sacks")), 1),
        "interceptions":    round(_safe_float(row.get("interceptions"))),
        "pass_breakups":    round(_safe_float(row.get("pass_break_ups"))),
        "fumble_recoveries":round(_safe_float(row.get("fumble_recoveries"))),
        "forced_fumbles":   round(_safe_float(row.get("forced_fumbles"))),
        "coverage_grade":   round(_safe_float(row.get("grades_coverage_defense")), 1),
        "run_def_grade":    round(_safe_float(row.get("grades_run_defense")), 1),
        "tackle_grade":     round(_safe_float(row.get("grades_tackle")), 1),
        # Career rates (used for stats grade)
        "epa_per_snap":     round(c_epa / c_snaps, 5),
        "stop_rate":        round(c_stops / c_snaps, 4),
        # 17g projections
        "tackles_17g":      round(c_tackles / max(c_snaps, 1.0) * proj_snaps_17g, 1),
        "stops_17g":        round(c_stops / max(c_snaps, 1.0) * proj_snaps_17g, 1),
        "tfl_17g":          round(c_tfl / max(c_snaps, 1.0) * proj_snaps_17g, 1),
        "ints_17g":         round(min(12.0, int_per_snap * proj_snaps_17g), 1),
        "pbus_17g":         round(min(22.0, pbu_per_snap * proj_snaps_17g), 1),
        "snaps_17g":        proj_snaps_17g,
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
    adj = max(-5.0, min(2.5, (avg_avail - 0.75) * 10.0))
    return round(adj, 2), round(avg_avail, 3)


def project_stats(
    last_stats: dict,
    composite_gr: float,
    current_age: int,
    contract_years: int,
    history: pd.DataFrame = None,
    grade_col: str = "grades_defense",
) -> List[dict]:
    projections = []
    grade = composite_gr
    player_yoy = player_recent_grade_yoy(history, grade_col)
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0
        projections.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "tackles":        round(min(200, last_stats["tackles_17g"] * scale), 1),
            "stops":          round(min(80, last_stats["stops_17g"] * scale), 1),
            "tfl":            round(min(20, last_stats["tfl_17g"] * scale), 1),
            "interceptions":  round(min(10, last_stats["ints_17g"] * scale), 1),
            "pass_breakups":  round(min(15, last_stats["pbus_17g"] * scale), 1),
            "coverage_grade": round(min(99, last_stats["coverage_grade"] * scale), 1),
            "run_def_grade":  round(min(99, last_stats["run_def_grade"] * scale), 1),
        })
    return projections


DISCOUNT_RATE = 0.08; CAP_GROWTH_RATE = 0.065


def compute_contract_value(
    composite_gr,
    current_age,
    contract_years,
    salary_ask,
    history: pd.DataFrame = None,
    grade_col: str = "grades_defense",
):
    breakdown = []; total_disc_value = total_disc_ask = total_nominal_value = 0.0
    weighted_fair_num = weighted_burden_num = weight_den = 0.0
    grade = float(composite_gr)
    player_yoy = player_recent_grade_yoy(history, grade_col)
    snap_rel, _ = snap_value_reliability_factor(history)
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        cap_factor = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))
        base_value = grade_to_market_value(grade) * snap_rel
        nominal_value = base_value * cap_factor
        cap_adj_ask = salary_ask / cap_factor
        front_weight = 1.0 / float(yr)
        total_nominal_value += nominal_value
        total_disc_value += nominal_value * time_discount
        total_disc_ask += cap_adj_ask * time_discount
        weighted_fair_num += nominal_value * front_weight
        weighted_burden_num += cap_adj_ask * front_weight
        weight_den += front_weight
        breakdown.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "market_value": base_value, "nominal_value": round(nominal_value, 2),
            "cap_adj_ask": round(cap_adj_ask, 2),
            "discounted_value": round(nominal_value * time_discount, 2),
            "year_surplus": round(base_value - cap_adj_ask, 2),
        })
    return (round(weighted_fair_num / max(weight_den, 1e-6), 2),
            round(weighted_burden_num / max(weight_den, 1e-6), 2),
            round(total_nominal_value, 2), breakdown)


class LBAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    predicted_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: LBAgentState):
    history = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    resolved_age = resolve_player_age_for_evaluation(state.get("player_history_full"), history)
    if resolved_age is not None:
        current_age = resolved_age
    elif "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        current_age = int(float(last_row["age"])) + (current_year - int(float(last_row["Year"])))
    else:
        current_age = 27
    last_stats = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)
    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)
    model_grade = _safe_float(history.sort_values("Year").iloc[-1].get("grades_defense"), 60.0)
    sg = _stats_grade(last_stats["coverage_grade"], last_stats["run_def_grade"],
                      last_stats["epa_per_snap"], last_stats["stop_rate"], last_stats["tackle_grade"])
    raw_cg = _composite_grade(model_grade, sg)
    cg = round(max(45.0, min(99.0, raw_cg + health_adj + inactivity_adj)), 2)
    return {
        "predicted_tier": _grade_to_tier(cg), "current_age": current_age,
        "last_season_stats": last_stats, "career_stats": career_stats,
        "stats_score": sg, "composite_grade": cg,
        "confidence": {"model_grade": round(model_grade, 2), "stats_grade": sg,
                       "composite_grade": cg, "health_factor": health_adj, "inactivity_penalty": inactivity_adj, "avg_availability": avg_avail},
    }


def evaluate_value(state: LBAgentState):
    hist = state.get("player_history")
    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        state["salary_ask"],
        history=hist,
        grade_col="grades_defense",
    )
    stat_proj = project_stats(
        state["last_season_stats"],
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        history=hist,
        grade_col="grades_defense",
    )
    inact_pen = float((state.get("confidence") or {}).get("inactivity_penalty", 0.0))
    stat_proj = apply_inactivity_to_projection_list(stat_proj, inact_pen)
    stat_proj = apply_projection_plausibility_caps(stat_proj, state.get("career_stats") or [])
    return {"valuation": fair_aav, "effective_cap_burden": eff_burden,
            "total_nominal_value": total_nom, "year_breakdown": breakdown, "projected_stats": stat_proj}


def assess_team_fit(state: LBAgentState):
    if not state.get("team_name"): return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


def make_decision(state: LBAgentState):
    ask = state["salary_ask"]; val = state["valuation"]; burden = state["effective_cap_burden"]
    tier = state["predicted_tier"]; cg = state["composite_grade"]
    mg = state["confidence"].get("model_grade", cg); sg = state["stats_score"]
    age = state["current_age"]; years = state["contract_years"]; total = state["total_nominal_value"]
    health_adj = state["confidence"].get("health_factor", 0)
    avg_avail = state["confidence"].get("avg_availability", 1.0)
    health_str = f" Health: {'+' if health_adj >= 0 else ''}{health_adj} pts ({round(avg_avail*100)}% availability)."
    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, grade_to_market_value, cg, roster, "LB",
        )
    surplus = round(val - burden, 2)
    surplus_pct = (val - burden) / max(val, 0.01) * 100
    if surplus_pct >= 20:    decision = "Exceptional Value"
    elif surplus_pct >= 5:   decision = "Good Signing"
    elif surplus_pct >= -5:  decision = "Fair Deal"
    elif surplus_pct >= -15: decision = "Slight Overpay"
    elif surplus_pct >= -30: decision = "Overpay"
    else:                    decision = "Poor Signing"
    reason = (
        f"{state['player_name']} (age {age}) projects as a {tier} linebacker. "
        f"PFF grade: {mg:.1f} · Stats grade (coverage 30%, run def 25%, EPA 20%, stops 15%, tackling 10%): {sg:.1f} → Composite: {cg:.1f}.{health_str} "
        f"Fair value: ${val}M/yr vs. ${burden}M/yr effective cap burden. "
        f"Surplus: ${surplus}M/yr ({surplus_pct:.0f}%). Total nominal: ${total}M."
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
    return {"decision": decision, "reasoning": reason, "team_fit_summary": fit_summary}


_workflow = StateGraph(LBAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
lb_gm_agent = _workflow.compile()
