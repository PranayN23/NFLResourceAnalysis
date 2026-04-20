"""
WR GM Agent — Wide Receiver Free Agency Evaluator

Stats grade weighted by: yprr 30%, yards_per_reception 25%,
drop_rate (inverse) 20%, epa_per_target 15%, yac_per_rec 10%.
Drops counted negatively. No rushing touchdowns included (defense only).
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
WR_CSV_PATH = os.path.join(_BASE, "ML", "WR.csv")


def grade_to_market_value(grade: float) -> float:
    return _gtmv_cal(grade, "WR")


# Stats anchors (empirical, per-season / rate basis)
_YPRR_ANCHORS = [0.0, 1.0,  1.5,  2.0,  2.5,  3.5]   # yards per route run
_YPR_ANCHORS  = [0.0, 8.0, 10.0, 12.0, 14.0, 18.0]   # yards per reception
_DROP_ANCHORS = [0.0, 0.85, 0.89, 0.93, 0.96, 1.0]    # (1 - drop_rate)
_EPA_T_ANCHORS= [-0.3, -0.1, 0.0,  0.1,  0.2,  0.35]  # epa per target
_YAC_ANCHORS  = [0.0, 2.0,  3.0,  4.0,  5.5,  8.0]   # yac per reception
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _stats_grade(yprr, yards_per_rec, drop_rate, epa_per_target, yac_per_rec):
    drop_inverse = max(0.0, 1.0 - drop_rate)
    yp = float(np.interp(yprr,        _YPRR_ANCHORS, _STAT_GRD_SCALE))
    yr = float(np.interp(yards_per_rec, _YPR_ANCHORS, _STAT_GRD_SCALE))
    dr = float(np.interp(drop_inverse, _DROP_ANCHORS, _STAT_GRD_SCALE))
    ep = float(np.interp(epa_per_target, _EPA_T_ANCHORS, _STAT_GRD_SCALE))
    ya = float(np.interp(yac_per_rec,  _YAC_ANCHORS, _STAT_GRD_SCALE))
    return round(0.30 * yp + 0.25 * yr + 0.20 * dr + 0.15 * ep + 0.10 * ya, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.35 * model_grade + 0.65 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# WR-specific: median YoY grades_offense on consecutive ML/WR.csv seasons. Regenerate: compute_position_age_curves.py
_AGE_DELTAS = {
    20: +2.2, 21: +2.2, 22: +1.5, 23: -0.4, 24: +0.1, 25: -1.4, 26: -0.4,
    27: -2.1, 28: -3.7, 29: -2.4, 30: -1.3, 31: -3.9, 32: -5.4, 33: -7.4,
    34: -3.5, 35: -3.5,
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
    for col in ("yards", "receptions", "targets", "grades_offense"):
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
        tgts = max(1.0, _safe_float(row.get("targets"), 1.0))
        recs = max(0.0, _safe_float(row.get("receptions")))
        seasons.append({
            "season":          year,
            "games_played":    int(games),
            "max_games":       int(max_g),
            "receptions":      round(recs),
            "targets":         round(tgts),
            "yards":           round(_safe_float(row.get("yards"))),
            "touchdowns":      round(_safe_float(row.get("touchdowns"))),
            "yards_per_rec":   round(_safe_float(row.get("yards_per_reception")), 2),
            "yac_per_rec":     round(_safe_float(row.get("yards_after_catch_per_reception")), 2),
            "yprr":            round(_safe_float(row.get("yprr")), 3),
            "drop_rate":       round(_safe_float(row.get("drop_rate")), 4),
            "drops":           round(_safe_float(row.get("drops"))),
            "epa":             round(_safe_float(row.get("Net EPA")), 2),
            "route_grade":     round(_safe_float(row.get("grades_pass_route")), 1),
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

    # Baseline uses recent seasons with recency + opportunity weighting so
    # low-snap early years do not anchor projections for ascending young WRs.
    recent = valid_rows.sort_values("Year").tail(3).copy()
    base_w = [0.20, 0.30, 0.50][-len(recent):]
    opp_vals = []
    for _, r in recent.iterrows():
        routes = max(0.0, _safe_float(r.get("routes")))
        tgts = max(0.0, _safe_float(r.get("targets")))
        opp_vals.append(max(routes, tgts * 3.2, 1.0))
    max_opp = max(opp_vals) if opp_vals else 1.0
    weights = []
    for bw, opp in zip(base_w, opp_vals):
        # Keep opportunity signal bounded so one massive season does not fully dominate.
        opp_factor = max(0.45, min(1.0, np.sqrt(opp / max_opp)))
        weights.append(bw * opp_factor)
    w_sum = max(sum(weights), 1e-9)
    weights = [w / w_sum for w in weights]

    w_yards = w_recs = w_tgts = w_drops = w_routes = w_yac = w_epa = 0.0
    w_tgts_pg = w_recs_pg = w_tds_pg = 0.0
    for (w, (_, r)) in zip(weights, recent.iterrows()):
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        yards = _safe_float(r.get("yards"))
        recs = _safe_float(r.get("receptions"))
        tgts = _safe_float(r.get("targets"))
        drops = _safe_float(r.get("drops"))
        routes = _safe_float(r.get("routes"))
        yac = _safe_float(r.get("yards_after_catch"))
        epa = _safe_float(r.get("Net EPA"))
        tds = _safe_float(r.get("touchdowns"))

        w_yards += w * yards
        w_recs += w * recs
        w_tgts += w * tgts
        w_drops += w * drops
        w_routes += w * routes
        w_yac += w * yac
        w_epa += w * epa
        w_tgts_pg += w * (tgts / g)
        w_recs_pg += w * (recs / g)
        w_tds_pg += w * (tds / g)

    w_tgts = max(w_tgts, 1.0)
    w_recs = max(w_recs, 1.0)
    w_routes = max(w_routes, 1.0)
    proj_tgts_17g = round(offense_target_load_17(w_tgts_pg * 17.0, 17.0, floor_17=74.0))
    proj_recs_17g = max(round(w_recs_pg * 17.0), round(proj_tgts_17g * 0.58))

    career_ypr = w_yards / w_recs
    career_yprr = w_yards / w_routes
    career_drop_rate = w_drops / w_tgts
    career_epa_t = w_epa / w_tgts
    career_yac_per_rec = w_yac / w_recs

    return {
        "season":         year,
        "games_played":   int(games),
        "max_games":      int(max_g),
        "availability":   avail,
        "receptions":     round(_safe_float(row.get("receptions"))),
        "targets":        round(_safe_float(row.get("targets"))),
        "yards":          round(_safe_float(row.get("yards"))),
        "drops":          round(_safe_float(row.get("drops"))),
        "route_grade":    round(_safe_float(row.get("grades_pass_route")), 1),
        # Career rates
        "yprr":           round(career_yprr, 3),
        "yards_per_rec":  round(career_ypr, 2),
        "drop_rate":      round(career_drop_rate, 4),
        "epa_per_target": round(career_epa_t, 4),
        "yac_per_rec":    round(career_yac_per_rec, 2),
        # 17g projections
        "yards_17g":      round(career_ypr * proj_recs_17g),
        "recs_17g":       proj_recs_17g,
        "tgts_17g":       proj_tgts_17g,
        "proj_recs_17g":  proj_recs_17g,
        "tds_17g":        round(w_tds_pg * 17, 1),
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
        trend_mult = projection_trend_multiplier("WR", age, yr, player_yoy)
        scale = max(0.25, min(1.85, base_scale * trend_mult))
        proj_recs  = round(min(130, last_stats["recs_17g"] * scale))
        proj_yards = round(min(2000, last_stats["yards_17g"] * scale))
        proj_ypr   = round(proj_yards / max(proj_recs, 1), 2)
        # drop_rate stays close to career baseline — slight uptick only when performance drops sharply
        proj_drop  = round(max(0, last_stats["drop_rate"] * (1.0 + max(0.0, 1.0 - scale) * 0.25)), 4)
        projections.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "receptions":    proj_recs,
            "yards":         proj_yards,
            "touchdowns":    round(min(20, last_stats["tds_17g"] * scale), 1),
            "yards_per_rec": proj_ypr,
            "yprr":          round(min(5, last_stats["yprr"] * scale), 3),
            "drop_rate":     proj_drop,
            "route_grade":   round(min(99, last_stats["route_grade"] * scale), 1),
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
    breakdown = []
    total_disc_value = total_disc_ask = total_nominal_value = 0.0
    weighted_fair_num = weighted_burden_num = weight_den = 0.0
    grade = float(composite_gr)
    player_yoy = player_recent_grade_yoy(history, grade_col)
    # Prefer route/target workload for valuation (total_snaps alone can understate WR usage vs 700 baseline).
    # Higher floor avoids ~0.55× fair AAV when history is missing in an edge path (reads as “elite grade, $19M fair”).
    snap_rel, _ = snap_value_reliability_factor(
        history,
        floor=0.85,
        column_priority=("routes", "targets", "total_snaps", "snap_counts_offense"),
    )
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(grade, age - 1, player_yoy, _annual_grade_delta)
        cap_factor    = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))
        base_value    = fair_market_aav_millions(grade, "WR", analysis_year) * snap_rel
        nominal_value = base_value * cap_factor
        disc_value    = nominal_value * time_discount
        cap_adj_ask   = salary_ask / cap_factor
        front_weight  = 1.0 / float(yr)
        total_nominal_value += nominal_value
        total_disc_value    += disc_value
        total_disc_ask      += cap_adj_ask * time_discount
        weighted_fair_num   += nominal_value * front_weight
        weighted_burden_num += cap_adj_ask * front_weight
        weight_den          += front_weight
        breakdown.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "market_value": base_value, "nominal_value": round(nominal_value, 2),
            "cap_adj_ask": round(cap_adj_ask, 2), "discounted_value": round(disc_value, 2),
            "year_surplus": round(base_value - cap_adj_ask, 2),
        })
    return (round(weighted_fair_num / max(weight_den, 1e-6), 2),
            round(weighted_burden_num / max(weight_den, 1e-6), 2),
            round(total_nominal_value, 2), breakdown)


class WRAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    player_history_full: pd.DataFrame; analysis_year: int
    predicted_tier: str; projected_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: WRAgentState):
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
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=current_year)
    raw_mg = _safe_float(history.sort_values("Year").iloc[-1].get("grades_offense"), 60.0)
    model_grade, snap_m = shrink_model_grade_for_season_snap_volume(
        raw_mg,
        history,
        grade_col="grades_offense",
        snap_profile=[
            ("snap_counts_offense", 680.0),
            ("total_snaps", 660.0),
            ("routes", 420.0),
        ],
    )
    sg = _stats_grade(last_stats["yprr"], last_stats["yards_per_rec"],
                      last_stats["drop_rate"], last_stats["epa_per_target"], last_stats["yac_per_rec"])
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


def evaluate_value(state: WRAgentState):
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


def assess_team_fit(state: WRAgentState):
    if not state.get("team_name"): return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


def make_decision(state: WRAgentState):
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
    trajectory = ("developing" if age <= 24 else "prime" if age <= 29 else "declining")
    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        _yr = int(state.get("analysis_year") or 2026)
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, lambda g: fair_market_aav_millions(g, "WR", _yr), cg, roster, "WR",
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
        f"{state['player_name']} (age {age}) projects as a {tier} wide receiver. "
        f"PFF route grade: {mg:.1f} · Stats grade: {sg:.1f} → Composite: {cg:.1f}.{health_str} "
        f"Currently {trajectory}. Fair value: ${val}M/yr vs. ${burden}M/yr cap burden. "
        f"Surplus: ${surplus}M/yr ({surplus_pct:.0f}%). Total: ${round(burden*years,2)}M ask. "
        f"Drops penalize route grade inversely."
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


_workflow = StateGraph(WRAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
wr_gm_agent = _workflow.compile()
