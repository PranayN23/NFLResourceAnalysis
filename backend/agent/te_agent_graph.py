"""
TE GM Agent — Tight End Free Agency Evaluator

Stats grade weighted by: pass_block_grade 25%, yprr 25%,
yards_per_reception 20%, epa_per_target 20%, drop_rate_inverse 10%.
Blocking is a key differentiator at this position.
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
    projection_trend_multiplier,
)
from backend.agent.stat_projection_utils import (
    clamp_inactivity_year,
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
from backend.agent.market_value_curves import fair_market_aav_millions, grade_to_market_value as _gtmv_cal

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TE_CSV_PATH = os.path.join(_BASE, "ML", "TightEnds", "TE.csv")


def grade_to_market_value(grade: float) -> float:
    return _gtmv_cal(grade, "TE")


# Stats anchors for TEs
_PB_ANCHORS   = [0.0, 50.0, 60.0, 70.0, 80.0, 92.0]   # pass_block_grade
_YPRR_ANCHORS = [0.0, 0.8,  1.2,  1.6,  2.0,  3.0]    # yards per route run
_YPR_ANCHORS  = [0.0, 7.0,  9.0, 11.0, 13.0, 16.0]    # yards per reception
_EPA_T_ANCHORS= [-0.3, -0.1, 0.0,  0.1,  0.2,  0.35]  # epa per target
_DROP_ANCHORS = [0.0, 0.85, 0.90, 0.94, 0.97, 1.0]    # 1 - drop_rate
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(pass_block_grade, yprr, yards_per_rec, epa_per_target, drop_rate):
    drop_inverse = max(0.0, 1.0 - drop_rate)
    pb = float(np.interp(pass_block_grade, _PB_ANCHORS, _STAT_GRD_SCALE))
    yp = float(np.interp(yprr,          _YPRR_ANCHORS, _STAT_GRD_SCALE))
    yr = float(np.interp(yards_per_rec,  _YPR_ANCHORS, _STAT_GRD_SCALE))
    ep = float(np.interp(epa_per_target, _EPA_T_ANCHORS, _STAT_GRD_SCALE))
    dr = float(np.interp(drop_inverse,   _DROP_ANCHORS, _STAT_GRD_SCALE))
    return round(0.25 * pb + 0.25 * yp + 0.20 * yr + 0.20 * ep + 0.10 * dr, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.47 * model_grade + 0.53 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# TE-specific: median YoY grades_offense on consecutive ML/TightEnds/TE.csv seasons.
_AGE_DELTAS = {
    20: +2.0, 21: +2.0, 22: +1.4, 23: -0.8, 24: +0.1, 25: -2.3, 26: -1.7,
    27: -2.9, 28: -1.5, 29: -1.9, 30: -0.9, 31: -3.1, 32: -4.4, 33: +1.4,
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
    for col in ("yards", "receptions", "grades_offense"):
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
        recs = max(1.0, _safe_float(row.get("receptions"), 1.0))
        seasons.append({
            "season":          year,
            "games_played":    int(games),
            "max_games":       int(max_g),
            "receptions":      round(recs),
            "yards":           round(_safe_float(row.get("yards"))),
            "touchdowns":      round(_safe_float(row.get("touchdowns"))),
            "yards_per_rec":   round(_safe_float(row.get("yards_per_reception")), 2),
            "yprr":            round(_safe_float(row.get("yprr")), 3),
            "drop_rate":       round(_safe_float(row.get("drop_rate")), 4),
            "drops":           round(_safe_float(row.get("drops"))),
            "epa":             round(_safe_float(row.get("Net EPA")), 2),
            "pass_block_grade":round(_safe_float(row.get("grades_pass_block")), 1),
            "run_block_grade": round(_safe_float(row.get("snap_counts_run_block")), 0) if "snap_counts_run_block" in row.index else 0,
            "overall_grade":   round(_safe_float(row.get("grades_offense")), 1),
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

    c_yards = c_recs = c_tgts = c_drops = c_routes = c_epa = c_games = c_tds = 0.0
    for _, r in valid_rows.iterrows():
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        c_yards  += _safe_float(r.get("yards"))
        c_recs   += _safe_float(r.get("receptions"))
        c_tgts   += _safe_float(r.get("targets"))
        c_drops  += _safe_float(r.get("drops"))
        c_routes += _safe_float(r.get("routes"))
        c_epa    += _safe_float(r.get("Net EPA"))
        c_tds    += _safe_float(r.get("touchdowns"))
        c_games  += g

    c_recs  = max(c_recs, 1.0); c_tgts = max(c_tgts, 1.0)
    c_routes= max(c_routes, 1.0); c_games = max(c_games, 1.0)
    tgt_17 = offense_target_load_17(c_tgts, c_games, floor_17=58.0, max_17=118.0)
    proj_recs_17g = max(round(c_recs / c_games * 17), round(tgt_17 * 0.62))

    return {
        "season":          year,
        "games_played":    int(games),
        "max_games":       int(max_g),
        "availability":    avail,
        "receptions":      round(_safe_float(row.get("receptions"))),
        "yards":           round(_safe_float(row.get("yards"))),
        "drops":           round(_safe_float(row.get("drops"))),
        "pass_block_grade":round(_safe_float(row.get("grades_pass_block")), 1),
        # Career rates
        "yprr":            round(c_yards / c_routes, 3),
        "yards_per_rec":   round(c_yards / c_recs, 2),
        "drop_rate":       round(c_drops / c_tgts, 4),
        "epa_per_target":  round(c_epa / c_tgts, 4),
        # 17g projections
        "yards_17g":       round((c_yards / c_recs) * proj_recs_17g),
        "recs_17g":        proj_recs_17g,
        "proj_recs_17g":   proj_recs_17g,
        "tds_17g":         round(c_tds / c_games * 17, 1),
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
        trend_mult = projection_trend_multiplier("TE", age, yr, player_yoy)
        scale = max(0.25, min(1.8, base_scale * trend_mult))
        proj_recs  = round(min(120, last_stats["recs_17g"] * scale))
        proj_yards = round(min(1700, last_stats["yards_17g"] * scale))
        proj_ypr   = round(proj_yards / max(proj_recs, 1), 2)
        proj_drop  = round(max(0, last_stats["drop_rate"] * (1.0 + max(0.0, 1.0 - scale) * 0.25)), 4)
        projections.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "receptions":      proj_recs,
            "yards":           proj_yards,
            "touchdowns":      round(min(15, last_stats["tds_17g"] * scale), 1),
            "yards_per_rec":   proj_ypr,
            "yprr":            round(min(4, last_stats["yprr"] * scale), 3),
            "pass_block_grade":round(min(99, last_stats["pass_block_grade"] * scale), 1),
            "drop_rate":       proj_drop,
        })
    return projections


DISCOUNT_RATE = 0.08; CAP_GROWTH_RATE = 0.065


def compute_contract_value(
    composite_gr,
    current_age,
    contract_years,
    salary_ask,
    history: pd.DataFrame = None,
    grade_col: str = "grades_offense",
    analysis_year: int = 2026,
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
        base_value = fair_market_aav_millions(grade, "TE", analysis_year) * snap_rel
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


class TEAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    player_history_full: pd.DataFrame; analysis_year: int
    predicted_tier: str; projected_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: TEAgentState):
    history = state["player_history"]
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    resolved_age = resolve_player_age_for_evaluation(state.get("player_history_full"), history, analysis_year=current_year)
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
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=clamp_inactivity_year(history, current_year))
    raw_mg = _safe_float(history.sort_values("Year").iloc[-1].get("grades_offense"), 60.0)
    model_grade, snap_m = shrink_model_grade_for_season_snap_volume(
        raw_mg,
        history,
        grade_col="grades_offense",
        snap_profile=[
            ("snap_counts_offense", 675.0),
            ("total_snaps", 660.0),
            ("routes", 420.0),
        ],
    )
    sg = _stats_grade(last_stats["pass_block_grade"], last_stats["yprr"],
                      last_stats["yards_per_rec"], last_stats["epa_per_target"], last_stats["drop_rate"])
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


def evaluate_value(state: TEAgentState):
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


def assess_team_fit(state: TEAgentState):
    if not state.get("team_name"): return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


def make_decision(state: TEAgentState):
    ask = state["salary_ask"]; val = state["valuation"]; burden = state["effective_cap_burden"]
    cg = state["composite_grade"]
    _ps = state.get("projected_stats") or []
    if _ps:
        tier = _grade_to_tier(_ps[0].get("projected_grade", cg))
    else:
        tier = state["predicted_tier"]
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
        _yr = int(state.get("analysis_year") or 2026)
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, lambda g: fair_market_aav_millions(g, "TE", _yr), cg, roster, "TE",
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
        f"{state['player_name']} (age {age}) projects as a {tier} tight end. "
        f"PFF grade: {mg:.1f} · Stats grade (blocking 25%, receiving 75%): {sg:.1f} → Composite: {cg:.1f}.{health_str} "
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
    return {"decision": decision, "reasoning": reason, "team_fit_summary": fit_summary, "projected_tier": tier}


_workflow = StateGraph(TEAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
te_gm_agent = _workflow.compile()
