"""
Helpers for 17-game / full-role projections so backup seasons don't collapse volume
or make turnover stats meaningless. Used by position agent_graph modules.
"""
from __future__ import annotations
import datetime


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
