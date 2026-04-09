"""
CB GM Agent — Cornerback Free Agency Evaluator

Stats grade weighted by: coverage_grade 45%, int+pbu_rate 30%,
qb_rating_against (inverse) 15%, tackle_grade 10%.
Coverage stats heavily weighted. No touchdowns (high variance).
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
from backend.agent.stat_projection_utils import coverage_target_load_17, inactivity_retirement_penalty, apply_inactivity_to_projection_list, apply_projection_plausibility_caps, snap_value_reliability_factor
import pandas as pd
import numpy as np
import os, datetime

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CB_CSV_PATH = os.path.join(_BASE, "ML", "CB.csv")

_GRADE_ANCHORS = [45,   55,   60,   65,   70,   75,   80,   85,   88,   92,   96,   100]
_VALUE_ANCHORS = [1.59, 3.41, 12.50, 18.18, 18.75, 19.89, 28.41, 30.68, 31.82, 32.95, 34.09, 35.23]
MARKET_CALIBRATION_FACTOR = 0.88


def grade_to_market_value(grade: float) -> float:
    grade = max(45.0, min(100.0, float(grade)))
    return round(float(np.interp(grade, _GRADE_ANCHORS, _VALUE_ANCHORS)) * MARKET_CALIBRATION_FACTOR, 2)


# Stats anchors — CBs are evaluated primarily on coverage
_COV_ANCHORS  = [0.0, 50.0, 60.0, 70.0, 80.0, 93.0]   # coverage grade
_INT_PBU_ANCHORS=[0.0, 0.04, 0.08, 0.12, 0.18, 0.28]  # (ints+pbus) / targets
# QB rating against: lower is better for CB (perfect = 0, average ~80, bad = 120+)
_QBR_INV_ANCHORS=[1.0, 0.65, 0.50, 0.35, 0.20, 0.0]  # (max_qbr - qbr) / max_qbr
_TAC_ANCHORS  = [0.0, 50.0, 60.0, 70.0, 80.0, 92.0]   # tackle grade
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]
_MAX_QBR = 118.75


def _stats_grade(coverage_grade, int_pbu_rate, qb_rating_against, tackle_grade):
    qbr_inv = max(0.0, (_MAX_QBR - qb_rating_against) / _MAX_QBR) if qb_rating_against > 0 else 0.5
    cg = float(np.interp(coverage_grade,  _COV_ANCHORS,     _STAT_GRD_SCALE))
    ip = float(np.interp(int_pbu_rate,    _INT_PBU_ANCHORS, _STAT_GRD_SCALE))
    qi = float(np.interp(qbr_inv,         _QBR_INV_ANCHORS, _STAT_GRD_SCALE))
    tg = float(np.interp(tackle_grade,    _TAC_ANCHORS,     _STAT_GRD_SCALE))
    return round(0.45 * cg + 0.30 * ip + 0.15 * qi + 0.10 * tg, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.40 * model_grade + 0.60 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# CB-specific: median YoY grades_defense on consecutive ML/CB.csv seasons.
_AGE_DELTAS = {
    20: +1.5, 21: +1.5, 22: +0.1, 23: -2.8, 24: +0.0, 25: -2.8, 26: +0.5,
    27: -3.5, 28: -2.2, 29: -1.4, 30: -3.7, 31: -5.3, 32: -1.0, 33: -2.1,
    34: -5.0,
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
    for col in ("grades_defense", "grades_coverage_defense", "snap_counts_defense", "tackles"):
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
        seasons.append({
            "season":           year,
            "games_played":     int(games),
            "max_games":        int(max_g),
            "interceptions":    round(_safe_float(row.get("interceptions"))),
            "pass_breakups":    round(_safe_float(row.get("pass_break_ups"))),
            "tackles":          round(_safe_float(row.get("tackles"))),
            "tackles_for_loss": round(_safe_float(row.get("tackles_for_loss")), 1),
            "fumble_recoveries":round(_safe_float(row.get("fumble_recoveries"))),
            "qb_rating_against":round(_safe_float(row.get("qb_rating_against")), 1),
            "coverage_grade":   round(_safe_float(row.get("grades_coverage_defense")), 1),
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

    c_ints = c_pbus = c_tgts = c_tackles = c_tfl = c_games = 0.0
    qbr_sum = qbr_count = 0.0
    for _, r in valid_rows.iterrows():
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        c_ints   += _safe_float(r.get("interceptions"))
        c_pbus   += _safe_float(r.get("pass_break_ups"))
        c_tgts   += _safe_float(r.get("targets"))
        c_tackles+= _safe_float(r.get("tackles"))
        c_tfl    += _safe_float(r.get("tackles_for_loss"))
        qbr_val   = _safe_float(r.get("qb_rating_against"), 0)
        if qbr_val > 0:
            qbr_sum += qbr_val; qbr_count += 1
        c_games  += g

    c_tgts = max(c_tgts, 1.0); c_games = max(c_games, 1.0)
    career_int_pbu_rate = (c_ints + c_pbus) / c_tgts
    avg_qbr_against = qbr_sum / max(qbr_count, 1)

    tgt_17 = coverage_target_load_17(c_tgts, c_games)
    int_per_tgt = c_ints / c_tgts
    pbu_per_tgt = c_pbus / c_tgts

    return {
        "season":           year,
        "games_played":     int(games),
        "max_games":        int(max_g),
        "availability":     avail,
        "interceptions":    round(_safe_float(row.get("interceptions"))),
        "pass_breakups":    round(_safe_float(row.get("pass_break_ups"))),
        "tackles":          round(_safe_float(row.get("tackles"))),
        "tackles_for_loss": round(_safe_float(row.get("tackles_for_loss")), 1),
        "fumble_recoveries":round(_safe_float(row.get("fumble_recoveries"))),
        "coverage_grade":   round(_safe_float(row.get("grades_coverage_defense")), 1),
        "tackle_grade":     round(_safe_float(row.get("grades_tackle")), 1),
        # Career rates
        "int_pbu_rate":     round(career_int_pbu_rate, 4),
        "qb_rating_against":round(avg_qbr_against, 1),
        # 17g projections
        "ints_17g":         round(min(12.0, int_per_tgt * tgt_17), 1),
        "pbus_17g":         round(min(22.0, pbu_per_tgt * tgt_17), 1),
        "tackles_17g":      round(max(c_tackles / c_games * 17, 52.0), 1),
        "tfl_17g":          round(c_tfl / c_games * 17, 1),
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
            "interceptions":   round(min(12, last_stats["ints_17g"] * scale), 1),
            "pass_breakups":   round(min(20, last_stats["pbus_17g"] * scale), 1),
            "tackles":         round(min(120, last_stats["tackles_17g"] * scale), 1),
            "tfl":             round(min(10, last_stats["tfl_17g"] * scale), 1),
            "coverage_grade":  round(min(99, last_stats["coverage_grade"] * scale), 1),
            "qb_rating_against":round(max(0, last_stats["qb_rating_against"] / max(scale, 0.5)), 1),
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


class CBAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    predicted_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: CBAgentState):
    history = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    if "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        current_age = int(float(last_row["age"])) + (current_year - int(float(last_row["Year"])))
    else:
        current_age = 27
    last_stats = extract_last_season_stats(history)
    career_stats = extract_career_stats(history)
    health_adj, avg_avail = _compute_health_factor(history)
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)
    model_grade = _safe_float(history.sort_values("Year").iloc[-1].get("grades_coverage_defense") or
                              history.sort_values("Year").iloc[-1].get("grades_defense"), 60.0)
    sg = _stats_grade(last_stats["coverage_grade"], last_stats["int_pbu_rate"],
                      last_stats["qb_rating_against"], last_stats["tackle_grade"])
    raw_cg = _composite_grade(model_grade, sg)
    cg = round(max(45.0, min(99.0, raw_cg + health_adj + inactivity_adj)), 2)
    return {
        "predicted_tier": _grade_to_tier(cg), "current_age": current_age,
        "last_season_stats": last_stats, "career_stats": career_stats,
        "stats_score": sg, "composite_grade": cg,
        "confidence": {"model_grade": round(model_grade, 2), "stats_grade": sg,
                       "composite_grade": cg, "health_factor": health_adj, "inactivity_penalty": inactivity_adj, "avg_availability": avg_avail},
    }


def evaluate_value(state: CBAgentState):
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


def assess_team_fit(state: CBAgentState):
    if not state.get("team_name"): return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


def make_decision(state: CBAgentState):
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
            val, grade_to_market_value, cg, roster, "CB",
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
        f"{state['player_name']} (age {age}) projects as a {tier} cornerback. "
        f"PFF coverage grade: {mg:.1f} · Stats grade (coverage 45%, INT/PBU 30%, QB rating vs 15%, tackle 10%): {sg:.1f} → Composite: {cg:.1f}.{health_str} "
        f"Fair value: ${val}M/yr vs. ${burden}M/yr cap burden (surplus: ${surplus}M/yr, {surplus_pct:.0f}%). "
        f"Total nominal value: ${total}M."
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


_workflow = StateGraph(CBAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
cb_gm_agent = _workflow.compile()
