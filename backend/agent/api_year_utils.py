import pandas as pd


LATEST_ANALYSIS_YEAR = 2025


def clamp_analysis_year(year: int | None) -> int:
    if year is None:
        return LATEST_ANALYSIS_YEAR
    try:
        y = int(year)
    except Exception:
        return LATEST_ANALYSIS_YEAR
    return max(1900, min(LATEST_ANALYSIS_YEAR, y))


def effective_year_for_df(df: pd.DataFrame, requested_year: int | None, year_col: str = "Year") -> int | None:
    if year_col not in df.columns:
        return None
    ys = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    if ys.empty:
        return None
    req = clamp_analysis_year(requested_year)
    eligible = ys[ys <= req]
    if not eligible.empty:
        return int(eligible.max())
    return int(ys.min())


def history_as_of_year(player_df: pd.DataFrame, analysis_year: int | None) -> pd.DataFrame:
    if player_df.empty or "Year" not in player_df.columns:
        return player_df.copy()
    eff = effective_year_for_df(player_df, analysis_year, "Year")
    if eff is None:
        return player_df.copy()
    return player_df[pd.to_numeric(player_df["Year"], errors="coerce") <= eff].copy()


def age_during_season(player_df: pd.DataFrame, season_year: int) -> int | None:
    """
    Estimate age during NFL season *season_year*.

    Uses the row with max Year in ``player_df`` as an anchor (age at that season).
    Many CSV exports repeat a single "current" age on every historical row; anchoring
    from the latest season and subtracting year deltas fixes retrospective analysis.
    """
    if player_df is None or player_df.empty:
        return None
    if "Year" not in player_df.columns or "age" not in player_df.columns:
        return None
    ycol = pd.to_numeric(player_df["Year"], errors="coerce")
    if ycol.isna().all():
        return None
    idx = int(ycol.idxmax())
    try:
        anchor_year = int(ycol.loc[idx])
        ar = player_df.loc[idx, "age"]
        if pd.isna(ar):
            return None
        anchor_age = int(float(ar))
    except (TypeError, ValueError):
        return None
    sy = int(season_year)
    out = anchor_age - (anchor_year - sy)
    return int(max(18, min(50, out)))


def resolve_player_age_for_evaluation(
    player_history_full: pd.DataFrame | None,
    player_history: pd.DataFrame,
) -> int | None:
    """
    Age during the evaluation season: max Year present in filtered ``player_history``,
    with anchor ages taken from ``player_history_full`` (all seasons) when provided.
    """
    if player_history is None or player_history.empty or "Year" not in player_history.columns:
        return None
    ys = pd.to_numeric(player_history["Year"], errors="coerce").dropna()
    if ys.empty:
        return None
    eff = int(ys.max())
    if player_history_full is None or player_history_full.empty:
        return None
    return age_during_season(player_history_full, eff)
