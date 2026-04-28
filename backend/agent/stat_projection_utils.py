"""
Helpers for 17-game / full-role projections so backup seasons don't collapse volume
or make turnover stats meaningless. Used by position agent_graph modules.
"""
from __future__ import annotations
import datetime
import math
from collections.abc import Sequence

import pandas as pd


def qb_stabilized_int_rate_per_db(
    c_ints: float,
    c_dbs: float,
    prior_per_db: float = 0.026,
    shrinkage_dbs: float = 280.0,
) -> float:
    """
    Empirical-Bayes style INT per dropback: blend observed rate with a league-ish prior
    so small-sample seasons don't yield 0.0% or wild rates.
    """
    ci = float(c_ints)
    cd = float(c_dbs)
    return (ci + shrinkage_dbs * prior_per_db) / max(cd + shrinkage_dbs, 1e-9)


def qb_stabilized_td_rate_per_db(
    c_tds: float,
    c_dbs: float,
    prior_per_db: float = 0.042,
    shrinkage_dbs: float = 220.0,
) -> float:
    """Same idea as INT rate for touchdown rate per dropback."""
    ct = float(c_tds)
    cd = float(c_dbs)
    return (ct + shrinkage_dbs * prior_per_db) / max(cd + shrinkage_dbs, 1e-9)


def qb_full_role_dropbacks_17(
    proj: float,
    min_floor: float = 560.0,
    max_cap: float = 720.0,
) -> float:
    """
    Free-agency signing is evaluated as a healthy #1 — floor near a full-season
    starter load (~32–34 DB/gm × 17).
    """
    x = float(proj)
    return min(max_cap, max(x, min_floor))


def infer_qb_signing_role(
    player_pass_grade: float,
    roster: list,
    positional_need: float,
) -> tuple[str, str]:
    """
    When team context exists, estimate whether the FA QB projects as starter,
    fringe starter (competition/split), or backup vs. the current roster.

    roster: from get_team_roster — sorted by snaps descending; entries use grade, snaps.
    """
    pg = float(player_pass_grade)
    need = float(positional_need)
    if not roster:
        return "starter", "No QB on roster snapshot — projecting a starter workload."

    top = roster[0]
    g1 = float(top.get("grade") or 0.0)
    s1 = int(top.get("snaps") or 0)
    gap = pg - g1

    if s1 < 350 or g1 < 56.0:
        if pg >= 58.0:
            return "starter", "Incumbent QB has limited snaps or a low grade — projecting starter role."
        return "fringe_starter", "Weak incumbent — projecting competition or rotation (fringe starter)."

    if gap >= 2.5:
        return "starter", "Grade edge vs. incumbent — projecting to lead the room or win the job."

    if gap >= -4.0:
        if need >= 55.0:
            return "fringe_starter", "Close to QB1 with elevated team need — projecting fringe starter / competition."
        return "fringe_starter", "Close to incumbent — projecting split snaps or QB2 with spot starts."

    return "backup", "Incumbent clearly ahead on grade — projecting backup workload."


def qb_dropbacks_17_for_role(
    role: str,
    career_pace_17: float,
    peak_17: float,
) -> float:
    """
    Map role → 17-game dropback load. Starter uses full-role floor/cap; fringe/backup
    use lower loads so counting stats match expected role (without punishing past backup seasons).
    """
    pace = float(career_pace_17)
    peak = float(peak_17) if peak_17 else pace
    starter_line = float(qb_full_role_dropbacks_17(max(pace, peak)))

    r = (role or "starter").strip().lower()
    if r == "starter":
        return starter_line

    if r == "fringe_starter":
        blended = 0.55 * starter_line + 0.45 * max(260.0, min(peak, pace * 1.08, 520.0))
        out = min(580.0, max(300.0, blended))
        return round(out, 1)

    raw = max(
        85.0,
        min(280.0, max(pace * 0.88, peak * 0.52, 110.0)),
    )
    return round(raw, 1)


def defense_snap_load_17(
    c_snaps: float,
    c_games: float,
    floor_17: float = 700.0,
    max_17: float = 1150.0,
) -> float:
    """Full-time defensive snap expectation for projection (starter role)."""
    raw = float(c_snaps) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def coverage_target_load_17(
    c_targets: float,
    c_games: float,
    floor_17: float = 58.0,
    max_17: float = 145.0,
) -> float:
    """Coverage targets scaled to 17 with a starter-role floor (CB/S/primary slot)."""
    raw = float(c_targets) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def pass_rush_snap_load_17(
    c_pr_snaps: float,
    c_games: float,
    floor_17: float = 455.0,
    max_17: float = 720.0,
) -> float:
    """Edge/interior pass-rush opportunities scaled to 17 with starter floor."""
    raw = float(c_pr_snaps) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def run_def_snap_load_17(
    c_def_snaps: float,
    c_games: float,
    floor_17: float = 640.0,
    max_17: float = 1150.0,
) -> float:
    """Defensive snaps for run-game / stop counting (DI/ED run downs)."""
    raw = float(c_def_snaps) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def offense_target_load_17(
    c_targets: float,
    c_games: float,
    floor_17: float = 72.0,
    max_17: float = 148.0,
) -> float:
    """Pass-game targets scaled to 17 with a starter-role floor (WR/TE)."""
    raw = float(c_targets) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def pass_block_snap_load_17(
    c_pb_snaps: float,
    c_games: float,
    floor_17: float = 1000.0,
    max_17: float = 1225.0,
) -> float:
    """Pass-block snaps scaled to 17 (full-time OL starter)."""
    raw = float(c_pb_snaps) / max(float(c_games), 1e-9) * 17.0
    return min(max_17, max(raw, floor_17))


def clamp_inactivity_year(history, current_year: int) -> int:
    """
    For extension evaluations the analysis year is the extension start (e.g. 2028)
    but CSV data only runs to the cutoff year (e.g. 2024). Cap the reference at
    data_max + 2 so future years don't trigger a false inactivity / retirement penalty.
    """
    if history is None or (hasattr(history, "empty") and history.empty):
        return current_year
    try:
        import pandas as _pd
        ys = _pd.to_numeric(history["Year"] if hasattr(history, "__getitem__") and "Year" in history else [], errors="coerce").dropna()
        if ys.empty:
            return current_year
        return min(int(current_year), int(ys.max()) + 2)
    except Exception:
        return current_year


def inactivity_retirement_penalty(
    history,
    current_year: int | None = None,
    meaningful_games_3y: float = 8.0,
) -> tuple[float, dict]:
    """
    Penalize likely-retired / inactive players beyond normal age curve.
    Returns (negative_or_zero_penalty, metadata).
    """
    if current_year is None:
        current_year = datetime.date.today().year
    if history is None or len(history) == 0:
        return -12.0, {"years_since_last_season": 99, "recent_games_3y": 0.0}

    years = []
    for _, r in history.iterrows():
        try:
            y = int(float(r.get("Year", 0)))
            years.append(y)
        except Exception:
            pass
    if not years:
        return -12.0, {"years_since_last_season": 99, "recent_games_3y": 0.0}

    last_year = max(years)
    ys = max(0, int(current_year) - int(last_year))
    recent = history.copy()
    recent["Year"] = recent["Year"].apply(lambda x: int(float(x)) if str(x).strip() != "" else 0)
    recent = recent[recent["Year"] >= last_year - 2]
    recent_games = 0.0
    for _, r in recent.iterrows():
        try:
            g = float(r.get("player_game_count", 0.0))
            if g > 0:
                recent_games += g
        except Exception:
            pass

    pen = 0.0
    # Strongly punish missing years.
    if ys >= 3:
        pen -= min(22.0, 6.0 + (ys - 3) * 5.0)

    # Only use game-count penalty when game counts are actually present.
    has_game_counts = any(
        str(v).strip() not in ("", "nan", "None")
        for v in recent.get("player_game_count", [])
    )
    if has_game_counts and recent_games < meaningful_games_3y:
        gap = max(0.0, meaningful_games_3y - recent_games)
        pen -= min(10.0, 0.9 * gap)

    return round(pen, 2), {
        "years_since_last_season": ys,
        "recent_games_3y": round(recent_games, 2),
    }


def inactivity_projection_scale(inactivity_penalty: float) -> float:
    """
    Convert inactivity penalty (negative points) to a projection scale.
    0.0 penalty -> 1.0x; strong negative penalties compress projections hard.
    """
    p = float(inactivity_penalty)
    return max(0.20, min(1.00, 1.0 + p / 25.0))


def snap_value_reliability_factor(
    history,
    floor: float = 0.55,
    column_priority: Sequence[str] | None = None,
) -> tuple[float, dict]:
    """
    Convert recent snap volume into a valuation reliability scalar.
    Lower-volume players receive discounted fair-value estimates so "small sample + cheap ask"
    does not look like elite value by default.

    ``column_priority``: try these workload columns first (e.g. WR should prefer ``routes`` /
    ``targets`` over ``total_snaps`` so full-time receivers are not scaled like part-time OL snaps).
    """
    if history is None or len(history) == 0:
        return floor, {"source_col": None, "recent_weighted": 0.0, "peak": 0.0}

    df = history.copy()
    base_candidates = [
        "dropbacks",
        "passing_snaps",
        "pass_block_snaps",
        "snap_counts_offense",
        "snap_counts_defense",
        "total_snaps",
        "routes",
        "targets",
        "attempts",
    ]
    if column_priority:
        snap_candidates = list(
            dict.fromkeys(list(column_priority) + [c for c in base_candidates if c not in column_priority])
        )
    else:
        snap_candidates = base_candidates
    snap_col = next((c for c in snap_candidates if c in df.columns), None)
    if snap_col is None:
        return 1.0, {"source_col": None, "recent_weighted": 0.0, "peak": 0.0}

    df[snap_col] = pd.to_numeric(df[snap_col], errors="coerce").fillna(0.0)
    if "Year" in df.columns:
        ys = pd.to_numeric(df["Year"], errors="coerce")
        df = df.assign(_year=ys).dropna(subset=["_year"])
        yearly = df.groupby(df["_year"].astype(int))[snap_col].sum().sort_index()
        snaps = yearly.tolist()
    else:
        snaps = df[snap_col].tolist()

    if not snaps:
        return floor, {"source_col": snap_col, "recent_weighted": 0.0, "peak": 0.0}

    recent = snaps[-3:]
    w = [0.2, 0.3, 0.5][-len(recent):]
    wsum = max(sum(w), 1e-9)
    recent_weighted = sum(v * wt for v, wt in zip(recent, w)) / wsum
    peak = max(snaps) if snaps else 0.0

    # Approximate full-time baselines by workload type.
    full_baseline = {
        "dropbacks": 560.0,
        "passing_snaps": 560.0,
        "pass_block_snaps": 1000.0,
        "snap_counts_offense": 700.0,
        "snap_counts_defense": 700.0,
        "total_snaps": 700.0,
        "routes": 420.0,
        "targets": 90.0,
        "attempts": 180.0,
    }.get(snap_col, 700.0)

    abs_factor = max(0.0, min(1.0, recent_weighted / max(full_baseline, 1e-9)))
    peak_denom = max(peak, full_baseline * 0.65, 1.0)
    peak_factor = max(0.0, min(1.0, recent_weighted / peak_denom))
    rel = 0.72 * abs_factor + 0.28 * peak_factor
    factor = max(float(floor), min(1.0, rel))
    return round(factor, 3), {
        "source_col": snap_col,
        "recent_weighted": round(float(recent_weighted), 2),
        "peak": round(float(peak), 2),
    }


def shrink_model_grade_for_season_snap_volume(
    raw_model_grade: float,
    history: pd.DataFrame | None,
    *,
    grade_col: str,
    snap_profile: list[tuple[str, float]],
    grade_fallback_col: str | None = None,
    anchor: float = 60.0,
    stress_floor: float = 0.40,
    established_blend: float = 0.40,
    prior_volume_ratio: float = 0.58,
    prior_grade_threshold: float = 72.0,
    prior_elite_grade: float = 78.5,
    prior_elite_volume_ratio: float = 0.46,
) -> tuple[float, dict]:
    """
    Pull the PFF / ML *model* grade toward *anchor* when the latest season has low snaps,
    on the theory that sustaining an elite mark over a full workload is harder than in a
    small role. If an *earlier* season shows a high snap year with a solid (or elite) grade,
    blend partway back toward 1.0 stress — typical injury / role blip without rewriting history.

    *snap_profile*: ordered ``(column_name, full_season_snap_baseline)`` pairs; the first
    column present in *history* is used (e.g. TE/WR exports often have ``total_snaps`` not
    ``snap_counts_offense``).

    Applied to model_grade only (stats_grade unchanged); composite is recomputed by callers.
    """
    meta: dict = {
        "snap_volume_adjust": False,
        "season_snaps": None,
        "snap_volume_stress": 1.0,
        "prior_full_snap_season": False,
        "snap_volume_column": None,
    }
    if history is None or history.empty or "Year" not in history.columns:
        return float(raw_model_grade), meta
    snap_col: str | None = None
    full_snap_reference = 700.0
    for col, ref in snap_profile:
        if col in history.columns:
            snap_col = col
            full_snap_reference = float(ref)
            break
    if snap_col is None or full_snap_reference <= 1.0:
        return float(raw_model_grade), meta
    gc, gf = grade_col, grade_fallback_col
    if gc not in history.columns:
        if gf and gf in history.columns:
            gc, gf = gf, None
        else:
            return float(raw_model_grade), meta

    h = history.copy()
    h["_y"] = pd.to_numeric(h["Year"], errors="coerce")
    h = h.dropna(subset=["_y"])
    if h.empty:
        return float(raw_model_grade), meta

    h["_sn"] = pd.to_numeric(h[snap_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    def _cell_grade(row: pd.Series) -> float:
        g1 = pd.to_numeric(row.get(gc), errors="coerce")
        if gf:
            g2 = pd.to_numeric(row.get(gf), errors="coerce")
            if pd.isna(g1) or abs(float(g1)) < 1e-6:
                return float(g2) if pd.notna(g2) else float("nan")
        return float(g1) if pd.notna(g1) else float("nan")

    yearly: list[tuple[int, float, float]] = []
    for yr, grp in h.groupby(h["_y"].astype(int)):
        ssum = float(grp["_sn"].sum())
        vals = [_cell_grade(row) for _, row in grp.iterrows()]
        vals = [v for v in vals if v == v]  # drop NaN
        gmean = float(sum(vals) / len(vals)) if vals else float("nan")
        yearly.append((int(yr), ssum, gmean))

    yearly.sort(key=lambda x: x[0])
    if not yearly:
        return float(raw_model_grade), meta

    last_y, season_snaps, _ = yearly[-1]

    prior_full_snap = False
    for yr, ssum, gmean in yearly:
        if yr >= last_y:
            continue
        if gmean != gmean:
            continue
        if ssum >= full_snap_reference * prior_volume_ratio and gmean >= prior_grade_threshold:
            prior_full_snap = True
            break
        if ssum >= full_snap_reference * prior_elite_volume_ratio and gmean >= prior_elite_grade:
            prior_full_snap = True
            break

    u = min(1.0, season_snaps / full_snap_reference)
    stress = math.sqrt(u)
    stress = max(float(stress_floor), min(1.0, stress))
    if prior_full_snap:
        stress = min(1.0, stress + (1.0 - stress) * float(established_blend))

    raw = float(raw_model_grade)
    adj = float(anchor) + (raw - float(anchor)) * stress
    adj = max(40.0, min(100.0, adj))
    meta.update(
        {
            "snap_volume_adjust": True,
            "season_snaps": round(season_snaps, 1),
            "snap_volume_stress": round(stress, 3),
            "prior_full_snap_season": prior_full_snap,
            "snap_volume_column": snap_col,
        }
    )
    return round(adj, 2), meta


def apply_inactivity_to_projection_list(
    projections: list[dict],
    inactivity_penalty: float,
) -> list[dict]:
    """
    Apply inactivity adjustment only to non-volume efficiency stats.
    Counting stats remain full 17-game volume projections.
    """
    base = inactivity_projection_scale(inactivity_penalty)
    s_eff = 0.70 + 0.30 * base
    if s_eff >= 0.999:
        return projections

    efficiency_tokens = (
        "rate", "pct", "rating", "ypa", "ypc", "ypr", "yprr", "grade", "epa",
    )
    out: list[dict] = []
    for row in projections or []:
        r = dict(row)
        for k, v in list(r.items()):
            if k in ("year", "age", "projected_grade"):
                continue
            if not isinstance(v, (int, float)):
                continue
            if any(tok in k for tok in efficiency_tokens):
                nv = float(v) * s_eff
                if isinstance(v, int):
                    r[k] = int(round(nv))
                else:
                    r[k] = round(nv, 4)
        out.append(r)
    return out


def apply_projection_plausibility_caps(
    projections: list[dict],
    career_stats: list[dict] | None = None,
) -> list[dict]:
    """
    Generic outlier control for projected counting stats across positions.
    Keeps efficiency keys untouched.
    """
    if not projections:
        return projections
    efficiency_tokens = ("rate", "pct", "rating", "ypa", "ypc", "ypr", "yprr", "grade", "epa")
    hard_caps = {
        "yards": 5800.0,
        "touchdowns": 65.0,
        "interceptions": 35.0,
        "receptions": 190.0,
        "targets": 230.0,
        "attempts": 760.0,
        "sacks": 95.0,
        "tackles": 210.0,
    }

    career_max: dict[str, float] = {}
    for row in career_stats or []:
        for k, v in row.items():
            if not isinstance(v, (int, float)):
                continue
            career_max[k] = max(career_max.get(k, 0.0), float(v))

    out = []
    for row in projections:
        r = dict(row)
        for k, v in list(r.items()):
            if k in ("year", "age", "projected_grade") or not isinstance(v, (int, float)):
                continue
            if any(tok in k for tok in efficiency_tokens):
                continue
            val = float(v)
            cap = hard_caps.get(k)
            cmax = career_max.get(k, 0.0)
            if cmax > 0:
                rel_cap = cmax * 1.22 + 8.0
                cap = rel_cap if cap is None else min(cap, rel_cap)
            if cap is not None:
                val = min(val, cap)
            r[k] = int(round(val)) if isinstance(v, int) else round(val, 4)
        out.append(r)
    return out
