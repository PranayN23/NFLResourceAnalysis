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
