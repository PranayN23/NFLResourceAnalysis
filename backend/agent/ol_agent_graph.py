"""
OL GM Agent — Offensive Lineman (T/G/C) Free Agency Evaluator

Shared agent logic for tackles, guards, and centers.
Position-specific weights:
  T: pass_block 65%, run_block 35% (premium on pass protection)
  G: pass_block 50%, run_block 50%
  C: pass_block 50%, run_block 50%

Stats: pbe, sacks_allowed (neg), hits_allowed (neg), hurries_allowed (neg),
       block_percent, pass_block_grade, run_block_grade, adjusted_value.
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
    pass_block_snap_load_17,
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
T_CSV_PATH = os.path.join(_BASE, "ML", "T.csv")
G_CSV_PATH = os.path.join(_BASE, "ML", "G.csv")
C_CSV_PATH = os.path.join(_BASE, "ML", "C.csv")


def grade_to_market_value(grade: float, position: str = "G") -> float:
    return _gtmv_cal(grade, position)


# Stats anchors for OL — PBE, pass block grade, run block grade
# Pass Block Efficiency (PBE): ideal ~99.5, average ~97-98, poor <95
_PBE_ANCHORS   = [80.0, 91.0, 94.0, 96.5, 98.0, 99.5]
_PBG_ANCHORS   = [0.0,  50.0, 60.0, 70.0, 80.0, 92.0]   # pass block grade
_RBG_ANCHORS   = [0.0,  50.0, 60.0, 70.0, 80.0, 92.0]   # run block grade
# Sacks/hurries allowed per pass block snap (lower = better)
_SA_RATE_ANCHORS  = [0.04, 0.025, 0.015, 0.008, 0.003, 0.0]
_HIT_RATE_ANCHORS = [0.08, 0.06,  0.04,  0.025, 0.01,  0.0]
_HU_RATE_ANCHORS  = [0.15, 0.12,  0.08,  0.05,  0.025, 0.0]
_STAT_GRD_SCALE = [45.0, 55.0, 65.0, 75.0, 85.0, 99.0]


def _pass_block_grade(pbe, pbg, sa_rate, hit_rate, hu_rate, position="G"):
    p  = float(np.interp(pbe,      _PBE_ANCHORS,     _STAT_GRD_SCALE))
    g  = float(np.interp(pbg,      _PBG_ANCHORS,     _STAT_GRD_SCALE))
    sa = float(np.interp(sa_rate,  _SA_RATE_ANCHORS,  _STAT_GRD_SCALE))
    h  = float(np.interp(hit_rate, _HIT_RATE_ANCHORS, _STAT_GRD_SCALE))
    hu = float(np.interp(hu_rate,  _HU_RATE_ANCHORS,  _STAT_GRD_SCALE))
    if position == "T":
        # Tackles: sacks allowed is a meaningful, well-attributed stat
        return round(0.25 * p + 0.25 * g + 0.20 * sa + 0.15 * h + 0.15 * hu, 2)
    elif position == "G":
        # Guards: sacks are rarely attributed; reduce sa_rate weight, shift to PBE/PBG
        return round(0.30 * p + 0.30 * g + 0.10 * sa + 0.15 * h + 0.15 * hu, 2)
    else:
        # Center: sacks almost never attributed — drop sa_rate entirely
        return round(0.35 * p + 0.35 * g + 0.15 * h + 0.15 * hu, 2)


def _run_block_grade_score(rbg):
    return float(np.interp(rbg, _RBG_ANCHORS, _STAT_GRD_SCALE))


def _stats_grade(pbe, pbg, rbg, sa_rate, hit_rate, hu_rate, pass_weight=0.50, position="G"):
    pb = _pass_block_grade(pbe, pbg, sa_rate, hit_rate, hu_rate, position=position)
    rb = _run_block_grade_score(rbg)
    run_weight = 1.0 - pass_weight
    return round(pass_weight * pb + run_weight * rb, 2)


def _composite_grade(model_grade, stats_gr):
    return round(0.58 * model_grade + 0.42 * stats_gr, 2)


def _grade_to_tier(grade):
    return grade_to_tier_universal(grade)


# OL: separate empirical curves (median YoY grades_offense, consecutive seasons) per ML/T.csv, G.csv, C.csv.
_AGE_DELTAS_T = {
    20: +3.6, 21: +3.6, 22: +1.7, 23: +0.3, 24: +1.3, 25: -0.2, 26: -0.8,
    27: -0.8, 28: -0.1, 29: -2.6, 30: -2.2, 31: -1.4, 32: -1.2, 33: -4.8,
    34: -3.5, 35: -3.5, 36: -3.5,
}
_AGE_DELTAS_G = {
    20: +0.3, 21: +0.3, 22: +0.3, 23: +0.6, 24: +1.6, 25: -0.2, 26: -0.4,
    27: -2.0, 28: -1.2, 29: -3.2, 30: -3.7, 31: -3.8, 32: -1.0, 33: -1.4,
    34: -3.5, 35: -3.5, 36: -3.5,
}
_AGE_DELTAS_C = {
    20: +3.8, 21: +3.8, 22: +3.8, 23: +1.1, 24: +1.9, 25: -1.7, 26: -0.7,
    27: -2.7, 28: -1.8, 29: -1.0, 30: -1.3, 31: -7.9, 32: +1.4, 33: -1.6,
    34: -5.2, 35: -3.5, 36: -3.5,
}
_AGE_DELTAS_BY_OL = {"T": _AGE_DELTAS_T, "G": _AGE_DELTAS_G, "C": _AGE_DELTAS_C}


def _annual_grade_delta(age, position: str = "G"):
    pos = position if position in _AGE_DELTAS_BY_OL else "G"
    tbl = _AGE_DELTAS_BY_OL[pos]
    k = int(age)
    lo, hi = min(tbl), max(tbl)
    return tbl[max(lo, min(hi, k))]


def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _has_valid_stats(row):
    for col in ("grades_offense", "grades_pass_block", "sacks_allowed", "snap_counts_offense"):
        val = row.get(col)
        try:
            f = float(val)
            if not np.isnan(f): return True
        except Exception: pass
    return False


def extract_career_stats(history: pd.DataFrame, position: str = "G") -> List[dict]:
    seasons = []
    for _, row in history.sort_values("Year").iterrows():
        if not _has_valid_stats(row): continue
        year = int(_safe_float(row.get("Year"), 2024))
        max_g = 17.0 if year >= 2021 else 16.0
        games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
        pb_snaps = max(1.0, _safe_float(row.get("snap_counts_pass_block"), 1.0))
        seasons.append({
            "season":           year,
            "games_played":     int(games),
            "max_games":        int(max_g),
            "sacks_allowed":    round(_safe_float(row.get("sacks_allowed"))),
            "hits_allowed":     round(_safe_float(row.get("hits_allowed"))),
            "hurries_allowed":  round(_safe_float(row.get("hurries_allowed"))),
            "pressures_allowed":round(_safe_float(row.get("pressures_allowed"))),
            "sacks_rate":       round(_safe_float(row.get("sacks_allowed")) / pb_snaps, 4),
            "pbe":              round(_safe_float(row.get("pbe")), 2),
            "block_percent":    round(_safe_float(row.get("block_percent")), 2),
            "pass_block_grade": round(_safe_float(row.get("grades_pass_block")), 1),
            "run_block_grade":  round(_safe_float(row.get("grades_run_block")), 1),
            "overall_grade":    round(_safe_float(row.get("grades_offense")), 1),
        })
    return seasons


def extract_last_season_stats(history: pd.DataFrame, position: str = "G") -> dict:
    sorted_hist = history.sort_values("Year")
    valid_rows = sorted_hist[sorted_hist.apply(_has_valid_stats, axis=1)]
    if valid_rows.empty: valid_rows = sorted_hist
    row = valid_rows.iloc[-1]
    year = int(_safe_float(row.get("Year"), 2024))
    max_g = 17.0 if year >= 2021 else 16.0
    games = max(1.0, min(max_g, _safe_float(row.get("player_game_count"), max_g)))
    avail = round(games / max_g, 3)

    c_sa = c_hits = c_hurr = c_pb_snaps = c_games = 0.0
    for _, r in valid_rows.iterrows():
        yr_g = 17.0 if int(_safe_float(r.get("Year"), 2024)) >= 2021 else 16.0
        g = max(1.0, min(yr_g, _safe_float(r.get("player_game_count"), yr_g)))
        pbs = max(1.0, _safe_float(r.get("snap_counts_pass_block"), 1.0))
        c_sa    += _safe_float(r.get("sacks_allowed"))
        c_hits  += _safe_float(r.get("hits_allowed"))
        c_hurr  += _safe_float(r.get("hurries_allowed"))
        c_pb_snaps += pbs
        c_games += g

    c_pb_snaps = max(c_pb_snaps, 1.0); c_games = max(c_games, 1.0)
    career_sa_rate  = c_sa   / c_pb_snaps
    career_hit_rate = c_hits / c_pb_snaps
    career_hu_rate  = c_hurr / c_pb_snaps
    pb_snaps_17 = pass_block_snap_load_17(c_pb_snaps, c_games)

    return {
        "season":           year,
        "games_played":     int(games),
        "max_games":        int(max_g),
        "availability":     avail,
        "sacks_allowed":    round(_safe_float(row.get("sacks_allowed"))),
        "hits_allowed":     round(_safe_float(row.get("hits_allowed"))),
        "hurries_allowed":  round(_safe_float(row.get("hurries_allowed"))),
        "pressures_allowed":round(_safe_float(row.get("pressures_allowed"))),
        "block_percent":    round(_safe_float(row.get("block_percent")), 2),
        "pass_block_grade": round(_safe_float(row.get("grades_pass_block")), 1),
        "run_block_grade":  round(_safe_float(row.get("grades_run_block")), 1),
        # Career rates
        "pbe":              round(_safe_float(row.get("pbe")), 2),
        "sa_rate":          round(career_sa_rate, 4),
        "hit_rate":         round(career_hit_rate, 4),
        "hurry_rate":       round(career_hu_rate, 4),
        # 17g projections (rate × full-time pass-block load)
        "sacks_17g":        round(career_sa_rate * pb_snaps_17, 1),
        "hits_17g":         round(career_hit_rate * pb_snaps_17, 1),
        "hurries_17g":      round(career_hu_rate * pb_snaps_17, 1),
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
    position: str = "G",
    history: pd.DataFrame = None,
    grade_col: str = "grades_offense",
) -> List[dict]:
    projections = []
    grade = composite_gr
    player_yoy = player_recent_grade_yoy(history, grade_col)
    for yr in range(1, contract_years + 1):
        age = current_age + yr - 1
        if yr > 1:
            grade = apply_yearly_grade_step(
                grade,
                age - 1,
                player_yoy,
                lambda a: _annual_grade_delta(a, position),
            )
        base_scale = max(0.25, min(1.5, grade / composite_gr)) if composite_gr > 0 else 1.0
        trend_mult = projection_trend_multiplier(position, age, yr, player_yoy)
        scale = max(0.25, min(1.8, base_scale * trend_mult))
        # For OL, lower is better for allowed stats — scale inversely
        inv = 1.0 / max(scale, 0.25)
        projections.append({
            "year": yr, "age": age, "projected_grade": round(grade, 1),
            "sacks_allowed":    round(max(0, last_stats["sacks_17g"] * inv), 1),
            "hits_allowed":     round(max(0, last_stats["hits_17g"] * inv), 1),
            "hurries_allowed":  round(max(0, last_stats["hurries_17g"] * inv), 1),
            "pass_block_grade": round(min(99, last_stats["pass_block_grade"] * scale), 1),
            "run_block_grade":  round(min(99, last_stats["run_block_grade"] * scale), 1),
        })
    return projections


DISCOUNT_RATE = 0.08; CAP_GROWTH_RATE = 0.065


def compute_contract_value(
    composite_gr,
    current_age,
    contract_years,
    salary_ask,
    position="G",
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
            grade = apply_yearly_grade_step(
                grade,
                age - 1,
                player_yoy,
                lambda a: _annual_grade_delta(a, position),
            )
        cap_factor = (1.0 + CAP_GROWTH_RATE) ** (yr - 1)
        time_discount = 1.0 / ((1.0 + DISCOUNT_RATE) ** (yr - 1))
        base_value = fair_market_aav_millions(grade, position, analysis_year) * snap_rel
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


class OLAgentState(TypedDict):
    player_name: str; salary_ask: float; contract_years: int; player_history: pd.DataFrame
    player_history_full: pd.DataFrame; analysis_year: int
    ol_position: str  # "T", "G", or "C"
    predicted_tier: str; projected_tier: str; confidence: Dict[str, float]; current_age: int
    last_season_stats: dict; career_stats: List[dict]; stats_score: float; composite_grade: float
    valuation: float; effective_cap_burden: float; total_nominal_value: float
    year_breakdown: List[dict]; projected_stats: List[dict]
    team_name: str; team_cap_available_pct: float; positional_need: float; need_label: str
    current_roster: List[dict]; signing_cap_pcts: List[float]; team_fit_summary: str
    decision: str; reasoning: str


def predict_performance(state: OLAgentState):
    history  = state["player_history"]
    position = state.get("ol_position", "G")
    pass_weight = 0.65 if position == "T" else 0.50
    current_year = int(state.get("analysis_year") or datetime.date.today().year)
    resolved_age = resolve_player_age_for_evaluation(state.get("player_history_full"), history, analysis_year=current_year)
    if resolved_age is not None:
        current_age = resolved_age
    elif "age" in history.columns and "Year" in history.columns:
        last_row = history.sort_values("Year").iloc[-1]
        current_age = int(float(last_row["age"])) + (current_year - int(float(last_row["Year"])))
    else:
        current_age = 28
    last_stats = extract_last_season_stats(history, position)
    career_stats = extract_career_stats(history, position)
    health_adj, avg_avail = _compute_health_factor(history)
    # Cap inactivity reference at data_max + 2 so future extension years aren't treated as retirement.
    _ol_data_max = current_year
    if "Year" in history.columns:
        _ys = pd.to_numeric(history["Year"], errors="coerce").dropna()
        if not _ys.empty:
            _ol_data_max = int(_ys.max())
    inactivity_adj, _ = inactivity_retirement_penalty(history, current_year=min(current_year, _ol_data_max + 2))
    raw_mg = _safe_float(history.sort_values("Year").iloc[-1].get("grades_offense"), 60.0)
    model_grade, snap_m = shrink_model_grade_for_season_snap_volume(
        raw_mg,
        history,
        grade_col="grades_offense",
        snap_profile=[
            ("snap_counts_offense", 985.0),
            ("total_snaps", 980.0),
        ],
    )
    sg = _stats_grade(
        last_stats["pbe"], last_stats["pass_block_grade"], last_stats["run_block_grade"],
        last_stats["sa_rate"], last_stats["hit_rate"], last_stats["hurry_rate"],
        pass_weight=pass_weight, position=position,
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


def evaluate_value(state: OLAgentState):
    position = state.get("ol_position", "G")
    hist = state.get("player_history")
    ay = int(state.get("analysis_year") or 2026)
    fair_aav, eff_burden, total_nom, breakdown = compute_contract_value(
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        state["salary_ask"],
        position,
        history=hist,
        grade_col="grades_offense",
        analysis_year=ay,
    )
    stat_proj = project_stats(
        state["last_season_stats"],
        state["composite_grade"],
        state["current_age"],
        state["contract_years"],
        position,
        history=hist,
        grade_col="grades_offense",
    )
    inact_pen = float((state.get("confidence") or {}).get("inactivity_penalty", 0.0))
    stat_proj = apply_inactivity_to_projection_list(stat_proj, inact_pen)
    stat_proj = apply_projection_plausibility_caps(stat_proj, state.get("career_stats") or [])
    return {"valuation": fair_aav, "effective_cap_burden": eff_burden,
            "total_nominal_value": total_nom, "year_breakdown": breakdown, "projected_stats": stat_proj}


def assess_team_fit(state: OLAgentState):
    if not state.get("team_name"): return {}
    return {
        "signing_cap_pcts": aav_to_cap_pcts(
            state["salary_ask"],
            state["contract_years"],
            int(state.get("analysis_year") or 2025),
        )
    }


_POS_LABEL = {"T": "offensive tackle", "G": "guard", "C": "center"}


def make_decision(state: OLAgentState):
    ask = state["salary_ask"]; val = state["valuation"]; burden = state["effective_cap_burden"]
    cg = state["composite_grade"]
    _ps = state.get("projected_stats") or []
    if _ps:
        tier = _grade_to_tier(_ps[0].get("projected_grade", cg))
    else:
        tier = state["predicted_tier"]
    mg = state["confidence"].get("model_grade", cg); sg = state["stats_score"]
    age = state["current_age"]; years = state["contract_years"]; total = state["total_nominal_value"]
    position = state.get("ol_position", "G")
    pos_label = _POS_LABEL.get(position, "offensive lineman")
    health_adj = state["confidence"].get("health_factor", 0)
    avg_avail = state["confidence"].get("avg_availability", 1.0)
    health_str = f" Health: {'+' if health_adj >= 0 else ''}{health_adj} pts ({round(avg_avail*100)}% availability)."
    team_nm = state.get("team_name", "")
    roster = state.get("current_roster") or []
    val_dec = val
    rep_note = ""
    if team_nm and roster:
        _yr = int(state.get("analysis_year") or 2026)
        def _gtmv(g):
            return fair_market_aav_millions(g, position, _yr)
        val_dec, rep_note = decision_fair_aav_with_replacement(
            val, _gtmv, cg, roster, position,
        )
    surplus = round(val - burden, 2)
    surplus_pct = (val - burden) / max(val, 0.01) * 100
    pass_note = " Pass protection weighted more heavily for T." if position == "T" else ""
    if surplus_pct >= 20:    decision = "Exceptional Value"
    elif surplus_pct >= 5:   decision = "Good Signing"
    elif surplus_pct >= -5:  decision = "Fair Deal"
    elif surplus_pct >= -15: decision = "Slight Overpay"
    elif surplus_pct >= -30: decision = "Overpay"
    else:                    decision = "Poor Signing"
    reason = (
        f"{state['player_name']} (age {age}) projects as a {tier} {pos_label}. "
        f"PFF grade: {mg:.1f} · Stats grade: {sg:.1f} → Composite: {cg:.1f}.{health_str}{pass_note} "
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


_workflow = StateGraph(OLAgentState)
_workflow.add_node("predict_performance", predict_performance)
_workflow.add_node("evaluate_value",      evaluate_value)
_workflow.add_node("assess_team_fit",     assess_team_fit)
_workflow.add_node("make_decision",       make_decision)
_workflow.set_entry_point("predict_performance")
_workflow.add_edge("predict_performance", "evaluate_value")
_workflow.add_edge("evaluate_value",      "assess_team_fit")
_workflow.add_edge("assess_team_fit",     "make_decision")
_workflow.add_edge("make_decision",       END)
ol_gm_agent = _workflow.compile()
