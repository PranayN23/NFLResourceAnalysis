"""
FA-style composite grades for bulk tier exports.

Mirrors each position's ``predict_performance`` (model / stats blend, health,
inactivity, and ED/DI ML ``predicted_grade``) so rankings match the Free Agency
evaluate APIs when no team context is selected.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable

import pandas as pd

from backend.agent.api_year_utils import clamp_analysis_year, history_as_of_year

# Lazy imports in _predictors() to avoid loading torch (ED/DI) until needed.

def _silent(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Batch exports should not spam the console from agent `print` debugging."""

    def _inner(st: dict[str, Any]) -> dict[str, Any]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(st)

    return _inner


def _build_predictors() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    from backend.agent import agent_graph as qb_mod
    from backend.agent import rb_agent_graph as hb_mod
    from backend.agent import wr_agent_graph as wr_mod
    from backend.agent import te_agent_graph as te_mod
    from backend.agent import ol_agent_graph as ol_mod
    from backend.agent import ed_agent_graph as ed_mod
    from backend.agent import di_agent_graph as di_mod
    from backend.agent import lb_agent_graph as lb_mod
    from backend.agent import cb_agent_graph as cb_mod
    from backend.agent import s_agent_graph as s_mod

    return {
        "QB": _silent(qb_mod.predict_performance),
        "HB": _silent(hb_mod.predict_performance),
        "WR": _silent(wr_mod.predict_performance),
        "TE": _silent(te_mod.predict_performance),
        "T": _silent(ol_mod.predict_performance),
        "G": _silent(ol_mod.predict_performance),
        "C": _silent(ol_mod.predict_performance),
        "ED": _silent(ed_mod.predict_performance),
        "DI": _silent(di_mod.predict_performance),
        "LB": _silent(lb_mod.predict_performance),
        "CB": _silent(cb_mod.predict_performance),
        "S": _silent(s_mod.predict_performance),
    }


_PREDICTORS_CACHE: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None


def _predictors() -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    global _PREDICTORS_CACHE
    if _PREDICTORS_CACHE is None:
        _PREDICTORS_CACHE = _build_predictors()
    return _PREDICTORS_CACHE


def _empty_team_state() -> dict[str, Any]:
    return {
        "team_name": "",
        "team_cap_available_pct": 0.0,
        "positional_need": 0.0,
        "need_label": "",
        "current_roster": [],
        "signing_cap_pcts": [],
        "team_fit_summary": "",
    }


def _minimal_state(
    pos_key: str,
    player_name: str,
    player_history: pd.DataFrame,
    player_history_full: pd.DataFrame,
    analysis_year: int,
) -> dict[str, Any]:
    st: dict[str, Any] = {
        "player_name": player_name,
        "salary_ask": 1.0,
        "contract_years": 1,
        "player_history": player_history,
        "player_history_full": player_history_full,
        "analysis_year": int(analysis_year),
        "predicted_tier": "",
        "confidence": {},
        "current_age": 28,
        "last_season_stats": {},
        "career_stats": [],
        "stats_score": 0.0,
        "composite_grade": 0.0,
        "valuation": 0.0,
        "effective_cap_burden": 0.0,
        "total_nominal_value": 0.0,
        "year_breakdown": [],
        "projected_stats": [],
        "decision": "",
        "reasoning": "",
        **_empty_team_state(),
    }
    if pos_key in ("T", "G", "C"):
        st["ol_position"] = pos_key
    return st


def composite_for_player_row(
    pos_key: str,
    player_name: str,
    position_full_df: pd.DataFrame,
    analysis_year: int,
) -> dict[str, Any] | None:
    """
    Return dict with composite_grade, model_grade, stats_grade, predicted_tier,
    and confidence (when present). None if no history after ``history_as_of_year``.
    """
    ay = clamp_analysis_year(analysis_year)
    preds = _predictors()
    fn = preds.get(pos_key)
    if fn is None or position_full_df.empty or "player" not in position_full_df.columns:
        return None

    nm = player_name.strip()
    if not nm:
        return None

    full = position_full_df[
        position_full_df["player"].astype(str).str.strip().str.lower() == nm.lower()
    ].copy()
    if full.empty:
        return None

    hist = history_as_of_year(full, ay)
    if hist.empty or "Year" not in hist.columns:
        return None

    st = _minimal_state(pos_key, nm, hist, full, ay)
    try:
        out = fn(st)
    except Exception:
        return None

    conf = out.get("confidence") or {}
    cg = float(out.get("composite_grade") or conf.get("composite_grade") or 0.0)
    if cg <= 0:
        return None
    return {
        "composite_grade": round(cg, 2),
        "model_grade": conf.get("model_grade"),
        "stats_grade": conf.get("stats_grade", out.get("stats_score")),
        "predicted_tier": out.get("predicted_tier", ""),
        "confidence": conf,
    }
