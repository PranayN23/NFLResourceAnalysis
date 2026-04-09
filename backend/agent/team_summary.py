import os
from functools import lru_cache
from typing import Any

import pandas as pd

from backend.agent.api_year_utils import clamp_analysis_year, effective_year_for_df


_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ML = os.path.join(_BASE, "ML")

TEAM_TO_ABBR = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LAR", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}

# Optional: map (abbr, season) -> full phrase if a row must be hand-corrected.
# Default behavior uses `Wins` from nflpowerrankings.csv as total wins (RS + postseason).
RECORD_OVERRIDES: dict[tuple[str, int], str] = {}

POS_CFG = {
    "QB": {"label": "Quarterback", "path": os.path.join(_ML, "QB.csv"), "grade": "grades_pass", "snaps": "passing_snaps"},
    "HB": {"label": "Running Back", "path": os.path.join(_ML, "HB.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "WR": {"label": "Wide Receiver", "path": os.path.join(_ML, "WR.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "TE": {"label": "Tight End", "path": os.path.join(_ML, "TightEnds", "TE.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "T": {"label": "Tackle", "path": os.path.join(_ML, "T.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "G": {"label": "Guard", "path": os.path.join(_ML, "G.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "C": {"label": "Center", "path": os.path.join(_ML, "C.csv"), "grade": "grades_offense", "snaps": "snap_counts_offense"},
    "ED": {"label": "Edge", "path": os.path.join(_ML, "ED.csv"), "grade": "grades_defense", "snaps": "snap_counts_defense"},
    "DI": {"label": "Defensive Interior", "path": os.path.join(_ML, "DI.csv"), "grade": "grades_defense", "snaps": "snap_counts_defense"},
    "LB": {"label": "Linebacker", "path": os.path.join(_ML, "LB.csv"), "grade": "grades_defense", "snaps": "snap_counts_defense"},
    "CB": {"label": "Cornerback", "path": os.path.join(_ML, "CB.csv"), "grade": "grades_defense", "snaps": "snap_counts_defense"},
    "S": {"label": "Safety", "path": os.path.join(_ML, "S.csv"), "grade": "grades_defense", "snaps": "snap_counts_defense"},
}


@lru_cache(maxsize=32)
def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_power_rankings() -> pd.DataFrame:
    p = os.path.join(_ML, "data", "nflpowerrankings.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def _team_season_wins_phrase(team: str, year: int) -> str:
    """
    Human-readable season win line for team summaries.

    In `nflpowerrankings.csv`, the `Wins` column is treated as **total team wins
    for that league year — regular season plus postseason** (not regular-season
    wins only). We describe it that way instead of inventing a W-L record, which
    was wrong when wins could exceed the 16/17-game regular-season schedule.
    """
    pr = _load_power_rankings()
    if pr.empty:
        return "had no win-total data available in our rankings file"
    abbr = TEAM_TO_ABBR.get(team, "")
    if not abbr:
        return "had no win-total data available in our rankings file"
    if (abbr, year) in RECORD_OVERRIDES:
        return RECORD_OVERRIDES[(abbr, year)]
    sub = pr[(pr["Team"] == abbr) & (pd.to_numeric(pr["Season"], errors="coerce") == year)]
    if sub.empty:
        return "had no win-total data available in our rankings file"
    raw = sub.iloc[0].get("Wins", float("nan"))
    try:
        wins = int(round(float(raw)))
    except (TypeError, ValueError):
        wins = -1
    if wins < 0 or wins > 24:
        return "had no reliable win total in our rankings file"
    if wins == 0:
        return "did not record a win in our rankings data for that season"
    return f"won {wins} games, including the postseason"


def _position_strength(team: str, year: int, cfg: dict[str, Any]) -> dict[str, Any] | None:
    df = _load_csv(cfg["path"])
    if df.empty or "Team" not in df.columns:
        return None
    y = effective_year_for_df(df, year, "Year")
    if y is None:
        return None
    sub = df[(df["Team"] == team) & (pd.to_numeric(df["Year"], errors="coerce") == y)].copy()
    if sub.empty:
        return None
    grade_col = cfg["grade"] if cfg["grade"] in sub.columns else (
        "grades_offense" if "grades_offense" in sub.columns else
        "grades_defense" if "grades_defense" in sub.columns else
        "grades_pass" if "grades_pass" in sub.columns else None
    )
    snap_col = cfg["snaps"] if cfg["snaps"] in sub.columns else (
        "snap_counts_offense" if "snap_counts_offense" in sub.columns else
        "snap_counts_defense" if "snap_counts_defense" in sub.columns else
        "passing_snaps" if "passing_snaps" in sub.columns else
        "total_snaps" if "total_snaps" in sub.columns else None
    )
    if grade_col is None:
        return None
    if snap_col is None:
        sub["_snap_fallback"] = 1.0
        snap_col = "_snap_fallback"
    sub[grade_col] = pd.to_numeric(sub[grade_col], errors="coerce")
    sub[snap_col] = pd.to_numeric(sub[snap_col], errors="coerce").fillna(0)
    sub = sub.dropna(subset=[grade_col])
    if sub.empty:
        return None
    w = sub[snap_col].clip(lower=1)
    score = float((sub[grade_col] * w).sum() / w.sum())
    ranked = sub.sort_values([grade_col, snap_col], ascending=[False, False]).head(2)
    top_players = []
    for _, row in ranked.iterrows():
        top_players.append({
            "player": str(row.get("player", "Unknown")),
            "grade": round(float(row.get(grade_col, 0.0)), 1),
        })
    best = ranked.iloc[0]
    return {
        "pos": cfg["label"],
        "score": round(score, 1),
        "player": str(best.get("player", "Unknown")),
        "player_grade": round(float(best.get(grade_col, 0.0)), 1),
        "players": top_players,
        "year_used": int(y),
    }


def _position_team_rank(team: str, year: int, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Rank a team within a position group for a given year using snap-weighted grade.
    Returns rank (1=best) out of teams present in that position dataset.
    """
    df = _load_csv(cfg["path"])
    if df.empty or "Team" not in df.columns:
        return None
    y = effective_year_for_df(df, year, "Year")
    if y is None:
        return None
    sub = df[pd.to_numeric(df["Year"], errors="coerce") == y].copy()
    if sub.empty:
        return None
    grade_col = cfg["grade"] if cfg["grade"] in sub.columns else (
        "grades_offense" if "grades_offense" in sub.columns else
        "grades_defense" if "grades_defense" in sub.columns else
        "grades_pass" if "grades_pass" in sub.columns else None
    )
    snap_col = cfg["snaps"] if cfg["snaps"] in sub.columns else (
        "snap_counts_offense" if "snap_counts_offense" in sub.columns else
        "snap_counts_defense" if "snap_counts_defense" in sub.columns else
        "passing_snaps" if "passing_snaps" in sub.columns else
        "total_snaps" if "total_snaps" in sub.columns else None
    )
    if grade_col is None:
        return None
    if snap_col is None:
        sub["_snap_fallback"] = 1.0
        snap_col = "_snap_fallback"
    sub[grade_col] = pd.to_numeric(sub[grade_col], errors="coerce")
    sub[snap_col] = pd.to_numeric(sub[snap_col], errors="coerce").fillna(0)
    sub = sub.dropna(subset=[grade_col])
    if sub.empty:
        return None

    def _team_score(g: pd.DataFrame) -> float:
        w = g[snap_col].clip(lower=1)
        return float((g[grade_col] * w).sum() / max(1.0, w.sum()))

    team_scores = sub.groupby("Team").apply(_team_score)
    if team not in team_scores.index:
        return None
    ordered = team_scores.sort_values(ascending=False)
    rank = int(ordered.index.get_loc(team)) + 1
    total = int(len(ordered))
    return {
        "abbr": cfg["label"],
        "rank": rank,
        "total_teams": total,
        "score": round(float(team_scores.loc[team]), 1),
        "year_used": int(y),
    }


def build_team_year_summary(team: str, analysis_year: int) -> dict[str, Any]:
    year = clamp_analysis_year(analysis_year)
    # Free agency in year Y uses prior season (Y-1) team performance context.
    season_context_year = year - 1
    pos_rows = []
    for cfg in POS_CFG.values():
        r = _position_strength(team, season_context_year, cfg)
        if r:
            pos_rows.append(r)
    if not pos_rows:
        return {
            "team": team,
            "analysis_year": year,
            "summary": f"In {year}, team summary data was unavailable for {team}.",
            "record": "record unavailable",
            "strengths": [],
            "weaknesses": [],
        }
    ordered = sorted(pos_rows, key=lambda x: x["score"], reverse=True)
    strengths = ordered[:3]
    weaknesses = sorted(pos_rows, key=lambda x: x["score"])[:3]
    # Keep team result timeline aligned with player timeline.
    # Example: analysis_year=2025 with latest player season=2024 should use 2024 record.
    season_year_used = max(int(r.get("year_used", year)) for r in pos_rows) if pos_rows else year
    record = _team_season_wins_phrase(team, season_year_used)
    def _fmt_group(r: dict[str, Any]) -> str:
        names = [p.get("player", "Unknown") for p in (r.get("players") or [])[:2]]
        if not names:
            names = [r.get("player", "Unknown")]
        return f"{r['pos']} ({' & '.join(names)})"

    strengths_txt = ", ".join([_fmt_group(r) for r in strengths])
    weaknesses_txt = ", ".join([_fmt_group(r) for r in weaknesses])
    if season_year_used != season_context_year:
        summary = (
            f"Heading into {year} free agency, this past season ({season_context_year}) "
            f"falls back to {season_year_used} in the available data. "
            f"The {team} {record}. "
            f"Their strongest position groups were {strengths_txt}. "
            f"Their weakest spots were {weaknesses_txt}."
        )
    else:
        summary = (
            f"Heading into {year} free agency, this past season ({season_context_year}) "
            f"the {team} {record}. "
            f"Their strongest position groups were {strengths_txt}. "
            f"Their weakest spots were {weaknesses_txt}."
        )
    return {
        "team": team,
        "analysis_year": year,
        "season_year_used": season_year_used,
        "record": record,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "summary": summary,
    }


def build_team_position_rankings(team: str, analysis_year: int) -> dict[str, Any]:
    year = clamp_analysis_year(analysis_year)
    season_context_year = year - 1
    rows = []
    for key, cfg in POS_CFG.items():
        r = _position_team_rank(team, season_context_year, cfg)
        if r:
            rows.append({
                "position_key": key,
                "position_label": cfg["label"],
                **r,
            })
    rows = sorted(rows, key=lambda x: x["position_key"])
    season_year_used = max((int(r.get("year_used", year)) for r in rows), default=year)
    return {
        "team": team,
        "analysis_year": year,
        "season_year_used": season_year_used,
        "rankings": rows,
    }


def build_player_directory(analysis_year: int) -> dict[str, Any]:
    """
    Build cross-position player directory using primary position inferred by
    highest snap volume through analysis_year.
    """
    year = clamp_analysis_year(analysis_year)
    by_player: dict[str, dict[str, Any]] = {}
    for key, cfg in POS_CFG.items():
        df = _load_csv(cfg["path"])
        if df.empty or "player" not in df.columns:
            continue
        if "Year" in df.columns:
            ys = pd.to_numeric(df["Year"], errors="coerce")
            df = df[ys <= year].copy()
        if df.empty:
            continue
        snap_col = cfg["snaps"] if cfg["snaps"] in df.columns else (
            "snap_counts_offense" if "snap_counts_offense" in df.columns else
            "snap_counts_defense" if "snap_counts_defense" in df.columns else
            "passing_snaps" if "passing_snaps" in df.columns else
            "total_snaps" if "total_snaps" in df.columns else None
        )
        if snap_col is None:
            df["_snap_fallback"] = 1.0
            snap_col = "_snap_fallback"
        df[snap_col] = pd.to_numeric(df[snap_col], errors="coerce").fillna(0.0)
        snap_by_player = df.groupby("player")[snap_col].sum()
        for name, snaps in snap_by_player.items():
            nm = str(name).strip()
            if not nm:
                continue
            prev = by_player.get(nm)
            if prev is None or float(snaps) > float(prev.get("snaps", 0.0)):
                by_player[nm] = {
                    "player": nm,
                    "position_key": key,
                    "position_label": cfg["label"],
                    "snaps": float(snaps),
                }
    players = sorted(by_player.values(), key=lambda x: x["player"])
    return {"analysis_year": year, "players": players}
