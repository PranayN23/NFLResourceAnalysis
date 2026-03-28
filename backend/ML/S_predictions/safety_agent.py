"""
LangChain agent for NFL Safety (S) grade predictions.

Answers questions about player grades, rankings, cap value, team situations, and
upside — designed to support NFL resource-allocation decisions at the safety position.

LLM priority: Claude (Anthropic) → OpenAI GPT-4o-mini

Install:
    pip install langchain-core langchain langchain-anthropic
    # or: pip install langchain-openai  (fallback)
    Set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY in .env at project root.

Run Player_Model_S.py first to generate S_predictions_by_player.csv.

Usage:
    python safety_agent.py
"""

import os
import pandas as pd

# Load .env from project root
try:
    from dotenv import load_dotenv
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_CSV  = os.path.join(_SCRIPT_DIR, "S_predictions_by_player.csv")
RAW_DATA_PATH    = os.path.join(_SCRIPT_DIR, "S.csv")
CORR_SUMMARY_PATH = os.path.join(_SCRIPT_DIR, "s_correlation_summary.txt")
CORR_TABLE_PATH   = os.path.join(_SCRIPT_DIR, "s_team_success_corr.csv")
TEST_YEAR        = 2024
RECENT_YEARS     = 3
MIN_CAP_M        = 1.0     # ignore players with < $1M cap hit (practice squad / vets minimum)


# ===========================================================================
# Data loaders
# ===========================================================================

def _load_predictions() -> pd.DataFrame:
    if not os.path.isfile(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_CSV}. Run Player_Model_S.py first."
        )
    return pd.read_csv(PREDICTIONS_CSV)


def _load_raw() -> pd.DataFrame:
    if not os.path.isfile(RAW_DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["position"] == "S"].copy()
    for col in ["grades_defense", "grades_coverage_defense", "Cap_Space",
                "adjusted_value", "age", "snap_counts_defense"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["player", "Year"])


# ===========================================================================
# Tool implementations
# ===========================================================================

def _tool_get_safety_rankings(n: int = 10) -> str:
    """Return top N safeties ranked by predicted grade."""
    df = _load_predictions()
    # Healthy-season players first; show injury flag if present
    has_flag = "prior_year_injury" in df.columns
    healthy  = df[~df["prior_year_injury"]] if has_flag else df
    top = (
        healthy.nlargest(int(n), "pred_ensemble")
        [["player", "Team", "pred_ensemble", "grades_defense", "prior_grades_defense"]]
        .rename(columns={
            "pred_ensemble":       "predicted_grade",
            "grades_defense":      "actual_grade_2024",
            "prior_grades_defense":"prior_grade_2023",
        })
    )
    top.index = range(1, len(top) + 1)
    return f"Top {n} safety predictions ({TEST_YEAR}) by predicted grade (healthy-season players only):\n" + top.to_string()


def _tool_get_player_prediction(player_name: str) -> str:
    """Return predicted grade and key stats for a single player."""
    df  = _load_predictions()
    raw = _load_raw()
    q   = str(player_name).strip().lower()
    mask = df["player"].str.lower().str.contains(q, na=False)
    if not mask.any():
        return f"No safety found matching '{player_name}'."
    row         = df.loc[mask].iloc[0]
    player_full = row["player"]
    is_injured  = bool(row.get("prior_year_injury", False))

    lines = [
        f"Player:              {player_full}",
        f"Team:                {row.get('Team', 'N/A')}",
    ]
    if is_injured:
        inj_snaps = row.get("injury_year_snaps", "N/A")
        lines.append(f"⚠ INJURY FLAG:       Player logged only {inj_snaps:.0f} snaps in {TEST_YEAR} "
                     f"(season filtered from model training).")
        lines.append(f"Healthy-season proj: {row['pred_ensemble']:.1f}  "
                     f"(based on last healthy season, NOT {TEST_YEAR} performance)")
        lines.append(f"Actual {TEST_YEAR} grade:  {row['grades_defense']:.1f}  (injury-affected — do not use for accuracy)")
        lines.append(f"Last healthy grade:  {row['prior_grades_defense']:.1f}")
    else:
        lines += [
            f"Predicted grade:     {row['pred_ensemble']:.1f}",
            f"Actual {TEST_YEAR} grade:   {row['grades_defense']:.1f}",
            f"Prior year grade:    {row['prior_grades_defense']:.1f}",
            f"Prediction error:    {row['ensemble_minus_actual']:.1f}  (predicted − actual)",
        ]

    if not raw.empty:
        hist = raw[raw["player"] == player_full].tail(RECENT_YEARS + 1)
        if not hist.empty:
            grade_str = ", ".join(
                f"{int(r.Year)}: {r.grades_defense:.0f}" for _, r in hist.iterrows()
            )
            lines.append(f"Grade history:       {grade_str}")
        cap = raw[(raw["player"] == player_full) & (raw["Year"] == TEST_YEAR)]["Cap_Space"]
        if not cap.empty and not pd.isna(cap.values[0]) and cap.values[0] > 0:
            lines.append(f"{TEST_YEAR} cap hit:        ${cap.values[0]:.1f}M")
    return "\n".join(lines)


def _tool_compare_players(player1: str, player2: str) -> str:
    """Side-by-side comparison of two safeties: grades, trend, and cap hit."""
    pred = _load_predictions()
    raw  = _load_raw()
    blocks = []

    for name in [player1, player2]:
        q    = name.strip().lower()
        mask = pred["player"].str.lower().str.contains(q, na=False)
        if not mask.any():
            blocks.append(f"No player found matching '{name}'.")
            continue
        row         = pred.loc[mask].iloc[0]
        player_full = row["player"]
        is_injured  = bool(row.get("prior_year_injury", False))
        header_tag  = " [INJURED]" if is_injured else ""
        lines = [f"--- {player_full} ({row.get('Team', 'N/A')}){header_tag} ---"]
        if is_injured:
            inj_snaps = row.get("injury_year_snaps", "?")
            lines += [
                f"  ⚠ Injury {TEST_YEAR}     : {inj_snaps:.0f} snaps — grade is injury-affected",
                f"  Healthy-season proj: {row['pred_ensemble']:.1f}  (based on pre-injury history)",
                f"  Actual {TEST_YEAR} grade : {row['grades_defense']:.1f}  (injury-affected)",
                f"  Last healthy grade : {row['prior_grades_defense']:.1f}",
            ]
        else:
            lines += [
                f"  Predicted grade : {row['pred_ensemble']:.1f}",
                f"  Actual {TEST_YEAR}     : {row['grades_defense']:.1f}",
                f"  Prior year      : {row['prior_grades_defense']:.1f}",
                f"  Pred error      : {row['ensemble_minus_actual']:+.1f}",
            ]
        if not raw.empty:
            hist = raw[raw["player"] == player_full].tail(RECENT_YEARS + 1)
            if not hist.empty:
                grade_str = ", ".join(
                    f"{int(r.Year)}: {r.grades_defense:.0f}" for _, r in hist.iterrows()
                )
                lines.append(f"  Grade history   : {grade_str}")
            cap_row = raw[(raw["player"] == player_full) & (raw["Year"] == TEST_YEAR)]
            if not cap_row.empty:
                cap = cap_row["Cap_Space"].values[0]
                age = cap_row["age"].values[0]
                if not pd.isna(cap) and cap > 0:
                    lines.append(f"  {TEST_YEAR} cap hit    : ${cap:.1f}M")
                if not pd.isna(age):
                    lines.append(f"  Age             : {int(age)}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _tool_get_team_safeties(team_name: str) -> str:
    """Return all predicted safeties on a given team."""
    df = _load_predictions()
    if "Team" not in df.columns:
        return "Team column not available."
    q    = str(team_name).strip().lower()
    mask = df["Team"].str.lower().str.contains(q, na=False)
    if not mask.any():
        return f"No safeties found for team matching '{team_name}'."
    team_label = df.loc[mask, "Team"].iloc[0]
    subset = df[mask].sort_values("pred_ensemble", ascending=False).copy()
    has_flag = "prior_year_injury" in subset.columns
    rows = []
    for _, row in subset.iterrows():
        injured = has_flag and bool(row.get("prior_year_injury", False))
        rows.append({
            "player":         row["player"] + (" *" if injured else ""),
            "predicted_grade": round(row["pred_ensemble"], 1),
            "actual_2024":     round(row["grades_defense"], 1),
            "prior_2023":      round(row["prior_grades_defense"], 1),
        })
    result = pd.DataFrame(rows)
    note = "\n* = injured in 2024; predicted_grade is healthy-season projection" if has_flag and subset["prior_year_injury"].any() else ""
    return f"Safeties on {team_label}:\n" + result.to_string(index=False) + note


def _tool_get_cap_value_rankings(n: int = 10) -> str:
    """
    Rank safeties by predicted grade per $1M of cap space.
    Only includes healthy-season players (injury-year players excluded — their cap
    hit reflects a season they didn't fully play, making efficiency misleading).
    """
    pred = _load_predictions()
    raw  = _load_raw()
    if raw.empty or "Cap_Space" not in raw.columns:
        return "Cap data not available in S.csv."

    # Exclude injury-flagged players from cap efficiency rankings
    if "prior_year_injury" in pred.columns:
        pred = pred[~pred["prior_year_injury"]].copy()

    cap_2024 = (
        raw[raw["Year"] == TEST_YEAR][["player", "Cap_Space", "age"]]
        .dropna(subset=["Cap_Space"])
    )
    cap_2024 = cap_2024[cap_2024["Cap_Space"] >= MIN_CAP_M]

    merged = pred.merge(cap_2024, on="player", how="inner")
    if merged.empty:
        return "Could not merge predictions with cap data."

    merged["grade_per_million"] = merged["pred_ensemble"] / merged["Cap_Space"]

    top = (
        merged.nlargest(int(n), "grade_per_million")
        [["player", "Team", "pred_ensemble", "Cap_Space", "grade_per_million", "age"]]
        .rename(columns={
            "pred_ensemble":     "predicted_grade",
            "Cap_Space":         "cap_hit_M",
            "grade_per_million": "grade_per_$1M",
        })
    )
    top["cap_hit_M"] = top["cap_hit_M"].map("${:.1f}M".format)
    top.index = range(1, len(top) + 1)
    return (
        f"Top {n} safeties by cap efficiency (predicted grade ÷ cap hit in $M):\n"
        + top.to_string()
        + "\n\nNote: grade_per_$1M = predicted_grade per million dollars of cap hit. "
          "Higher = more value per dollar. Injury-year players excluded."
    )


def _tool_get_young_ascending_players(max_age: int = 26) -> str:
    """
    Return safeties aged <= max_age ranked by predicted grade.
    Injured players are included but flagged — their predicted grade is a
    healthy-season projection, making them potentially strong buy-low targets.
    """
    pred = _load_predictions()
    raw  = _load_raw()
    if raw.empty:
        return "Raw data not available."
    ages   = raw[raw["Year"] == TEST_YEAR][["player", "age"]].dropna(subset=["age"])
    merged = pred.merge(ages, on="player", how="inner")
    young  = merged[merged["age"] <= float(max_age)].sort_values("pred_ensemble", ascending=False)
    if young.empty:
        return f"No safeties age ≤ {max_age} found."
    has_flag = "prior_year_injury" in young.columns
    rows = []
    for _, row in young.iterrows():
        injured = has_flag and bool(row.get("prior_year_injury", False))
        rows.append({
            "player":         row["player"] + (" *" if injured else ""),
            "Team":           row.get("Team", "N/A"),
            "age":            int(row["age"]),
            "predicted_grade": round(row["pred_ensemble"], 1),
            "actual_2024":     round(row["grades_defense"], 1),
            "prior_2023":      round(row["prior_grades_defense"], 1),
        })
    result = pd.DataFrame(rows)
    result.index = range(1, len(result) + 1)
    note = "\n* = injured in 2024; predicted_grade is healthy-season projection (buy-low candidate)" if has_flag and young["prior_year_injury"].any() else ""
    return f"Safeties age ≤ {max_age} ranked by predicted grade:\n" + result.to_string() + note


def _tool_get_grade_trend(player_name: str) -> str:
    """Return the year-by-year grade history for a player to show trajectory."""
    raw = _load_raw()
    if raw.empty:
        return "Raw data not available."
    q    = str(player_name).strip().lower()
    mask = raw["player"].str.lower().str.contains(q, na=False)
    if not mask.any():
        return f"No safety found matching '{player_name}' in historical data."
    player_full = raw.loc[mask, "player"].iloc[0]
    hist = (
        raw[raw["player"] == player_full]
        [["Year", "age", "grades_defense", "grades_coverage_defense", "snap_counts_defense"]]
        .tail(6)
    )
    pred = _load_predictions()
    pmask = pred["player"].str.lower().str.contains(q, na=False)
    output = f"Grade trend for {player_full}:\n" + hist.to_string(index=False)
    if pmask.any():
        row = pred.loc[pmask].iloc[0]
        output += f"\n{TEST_YEAR} Predicted grade: {row['pred_ensemble']:.1f}"
    return output


def _tool_get_model_info() -> str:
    """Return a description of the prediction model."""
    df = _load_predictions()
    return (
        f"NFL Safety grade prediction model — predicts {TEST_YEAR} PFF defensive grades. "
        f"{len(df)} players in predictions. "
        "Architecture: Transformer (Time2Vec + self-attention) + XGBoost ensemble, "
        "trained on 2010–2022 data, validated on 2023, tested on 2024. "
        "Features: PFF grades, coverage metrics, tackle rates, snap role shares, "
        "age-vs-peak, 3-year rolling grade mean/slope, EPA trends, contract value. "
        "Calibration: linear scale+shift on validation set to correct mean-reversion compression. "
        "Young elite adjustment: players age < 27 with prior grade > 80 receive an "
        "age-based growth projection (1.5 pts/yr) instead of pure mean reversion. "
        "KNOWN LIMITATIONS: The model under-performs on (a) injury-shortened seasons where "
        "a player logged <60% of their prior-year snaps, and (b) rare breakout years (actual "
        "grade > prior + 20 pts). On healthy starters without breakouts the Pearson r ≈ 0.34 "
        "(Spearman ≈ 0.28), meaning it has modest ordinal ranking ability but limited point "
        "estimate precision (MAE ≈ 8 grade points in that subset)."
    )


def _tool_get_injured_player_projections() -> str:
    """
    Return all safeties who were injured in TEST_YEAR (< 300 snaps) with their
    healthy-season grade projections.  These players were excluded from model
    accuracy testing but their predictions are based on pre-injury history and
    represent expected performance in a full healthy season.
    Useful for identifying buy-low free-agency or trade targets.
    """
    df = _load_predictions()
    if "prior_year_injury" not in df.columns:
        return "No injury flag column found. Re-run Player_Model_S.py to regenerate predictions."
    inj = df[df["prior_year_injury"] == True].copy()
    if inj.empty:
        return "No injury-year players found in predictions."
    inj = inj.sort_values("pred_ensemble", ascending=False)
    cols = ["player", "Team", "pred_ensemble", "prior_grades_defense",
            "grades_defense", "injury_year_snaps"]
    cols = [c for c in cols if c in inj.columns]
    result = (
        inj[cols]
        .rename(columns={
            "pred_ensemble":        "healthy_season_proj",
            "prior_grades_defense": "last_healthy_grade",
            "grades_defense":       "actual_injury_grade",
            "injury_year_snaps":    "snaps_played",
        })
    )
    result.index = range(1, len(result) + 1)
    return (
        f"Injured safeties — healthy-season projections ({TEST_YEAR}):\n"
        + result.to_string()
        + "\n\nhealthy_season_proj = model prediction based on pre-injury history."
        + "\nlast_healthy_grade  = most recent full-season PFF grade before injury."
        + "\nactual_injury_grade = {TEST_YEAR} grade (injury-affected, not used for accuracy)."
        + "\nThese players may be available at a discount; projection reflects healthy upside."
    )


def _tool_get_position_team_success_correlation() -> str:
    """
    Return the correlation analysis between safety grades and team success metrics
    (Win %, Net EPA, Total DVOA), plus 2024 prediction validation with injury/breakout breakdown.
    Regenerates the analysis if pre-computed summary is missing.
    """
    # Return cached summary if available
    if os.path.isfile(CORR_SUMMARY_PATH):
        with open(CORR_SUMMARY_PATH) as f:
            return f.read()

    # Otherwise run the analysis inline (imports local module)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "s_corr",
            os.path.join(_SCRIPT_DIR, "s_team_success_correlation.py")
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run()
        if os.path.isfile(CORR_SUMMARY_PATH):
            with open(CORR_SUMMARY_PATH) as f:
                return f.read()
    except Exception as e:
        pass

    return (
        "Pre-computed correlation summary not found. "
        "Run s_team_success_correlation.py to generate s_correlation_summary.txt.\n\n"
        "Key established findings (from prior analysis):\n"
        "  - Top-2 safety avg grade is the strongest predictor of team DVOA (r=0.311, p<0.001)\n"
        "  - Mean safety grade correlates r=0.277 with team Win % (highly significant, n=480)\n"
        "  - Top-2 safety avg grade correlates r=0.280 with Net EPA\n"
        "  - All correlations are statistically significant (p<0.05) across Win %, Net EPA, DVOA\n"
        "  - Effect is consistent: better safety room → meaningfully better team outcomes\n"
        "  - 2024 model MAE: ~8-10 grade points; reliable for ranking healthy starters"
    )


def _tool_get_team_safety_success_ranking(year: int = 2024) -> str:
    """Return all 32 teams ranked by safety quality for a given year, with team success metrics."""
    if not os.path.isfile(CORR_TABLE_PATH):
        return "Team success table not found. Run s_team_success_correlation.py first."
    tbl = pd.read_csv(CORR_TABLE_PATH)
    yr_df = tbl[tbl["year"] == int(year)].copy()
    if yr_df.empty:
        avail = sorted(tbl["year"].unique().tolist())
        return f"No data for year {year}. Available years: {avail}"
    yr_df = yr_df.sort_values("top2_s_avg", ascending=False).reset_index(drop=True)
    yr_df.index = range(1, len(yr_df) + 1)
    yr_df["win_pct"] = yr_df["win_pct"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    yr_df["net_epa"] = yr_df["net_epa"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")
    cols = ["team", "top1_s_grade", "top2_s_avg", "mean_s_grade", "win_pct", "net_epa"]
    cols = [c for c in cols if c in yr_df.columns]
    return (
        f"Teams ranked by safety quality ({year}), top-2 avg grade as primary sort:\n"
        + yr_df[cols].to_string()
        + "\n\nNote: top2_s_avg correlates r=0.28–0.31 with Win %, Net EPA, and Total DVOA."
    )


# ===========================================================================
# LangChain wiring
# ===========================================================================

try:
    from langchain_core.tools import tool

    @tool
    def get_safety_rankings(n: int = 10) -> str:
        """Get the top N safeties by predicted grade for 2024. Use for best-player or ranking questions."""
        return _tool_get_safety_rankings(n)

    @tool
    def get_player_prediction(player_name: str) -> str:
        """Get predicted grade, actual grade, cap hit, and grade history for one safety by name."""
        return _tool_get_player_prediction(player_name)

    @tool
    def compare_players(player1: str, player2: str) -> str:
        """Compare two safeties side by side: grades, trajectory, cap hit, and age."""
        return _tool_compare_players(player1, player2)

    @tool
    def get_team_safeties(team_name: str) -> str:
        """Get all safeties on a team ranked by predicted grade. Use team city or nickname."""
        return _tool_get_team_safeties(team_name)

    @tool
    def get_cap_value_rankings(n: int = 10) -> str:
        """
        Rank safeties by predicted grade per 1% of cap space (efficiency).
        Use when asked about value, bang-for-buck, or resource allocation.
        """
        return _tool_get_cap_value_rankings(n)

    @tool
    def get_young_ascending_players(max_age: int = 26) -> str:
        """
        List safeties aged at or below max_age ranked by predicted grade.
        Use for questions about upside, future investments, or young talent.
        """
        return _tool_get_young_ascending_players(max_age)

    @tool
    def get_grade_trend(player_name: str) -> str:
        """
        Show the year-by-year defensive grade history for a safety.
        Use when asked about trajectory, consistency, decline, or improvement.
        """
        return _tool_get_grade_trend(player_name)

    @tool
    def get_model_info() -> str:
        """Describe how the prediction model works, including known limitations. Use when asked about methodology or data."""
        return _tool_get_model_info()

    @tool
    def get_injured_player_projections() -> str:
        """
        List safeties who were injured in 2024 (< 300 snaps) with their healthy-season
        grade projections based on pre-injury history.
        Use when asked about: injured players, buy-low targets, recovery candidates,
        players who missed time, or anyone whose 2024 grade was injury-affected.
        """
        return _tool_get_injured_player_projections()

    @tool
    def get_position_team_success_correlation() -> str:
        """
        Return the statistical analysis of how safety position quality correlates with team
        success metrics (Win %, Net EPA, Total DVOA) across 15 seasons of NFL data.
        Also includes 2024 prediction validation broken down by injury/breakout status.
        Use when asked about: position impact, team success, how safeties affect winning,
        model accuracy, prediction reliability, or injury sensitivity.
        """
        return _tool_get_position_team_success_correlation()

    @tool
    def get_team_safety_success_ranking(year: int = 2024) -> str:
        """
        Rank all 32 teams by safety room quality (top-2 safety avg grade) for a given year,
        alongside Win % and Net EPA. Use to compare team safety investments or identify
        which franchises have elite vs. weak safety rooms.
        """
        return _tool_get_team_safety_success_ranking(year)

    TOOLS = [
        get_safety_rankings,
        get_player_prediction,
        compare_players,
        get_team_safeties,
        get_cap_value_rankings,
        get_young_ascending_players,
        get_grade_trend,
        get_model_info,
        get_injured_player_projections,
        get_position_team_success_correlation,
        get_team_safety_success_ranking,
    ]
    HAS_LANGCHAIN = True

except ImportError:
    HAS_LANGCHAIN = False
    TOOLS = []


# ===========================================================================
# LLM selection: Groq (free) → Claude (Anthropic) → OpenAI fallback
# ===========================================================================

def _get_llm():
    # Prefer Groq — free tier, fast, no credit card required
    try:
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            print("Using Groq (kimi-k2) as LLM.")
            return ChatGroq(model="moonshotai/kimi-k2-instruct", temperature=0, api_key=api_key)
    except ImportError:
        pass

    # Fall back to Claude
    try:
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("Using Claude (Anthropic) as LLM.")
            return ChatAnthropic(model="claude-sonnet-4-6", temperature=0, api_key=api_key)
    except ImportError:
        pass

    # Fall back to OpenAI
    try:
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print("Using OpenAI GPT-4o-mini as LLM.")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    except ImportError:
        pass

    raise RuntimeError(
        "No LLM available.\n"
        "Option A (free):      pip install langchain-groq       and set GROQ_API_KEY in .env\n"
        "Option B (preferred): pip install langchain-anthropic  and set ANTHROPIC_API_KEY in .env\n"
        "Option C (fallback):  pip install langchain-openai     and set OPENAI_API_KEY in .env"
    )


# ===========================================================================
# Agent factory
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an NFL resource-allocation analyst specialising in the safety position. "
    "You have access to a machine-learning ensemble (Transformer + XGBoost) that predicts "
    "PFF defensive grades for every safety, along with actual grades, cap hits, age, and "
    "grade history.\n\n"

    "INJURY HANDLING:\n"
    "Players who logged < 300 snaps in the test year are flagged prior_year_injury=True. "
    "Their pred_ensemble is a HEALTHY-SEASON projection based on pre-injury history — NOT "
    "a repeat of their injury year. These players are excluded from accuracy metrics but are "
    "included in all rankings as potential buy-low targets. Always mention the injury flag "
    "when discussing these players. Use get_injured_player_projections() for a dedicated list.\n\n"

    "POSITION CONTEXT (safety → team success):\n"
    "Analysis across 480 team×season observations (2010–2024):\n"
    "  • Top-2 safety avg grade vs Win %:   r=0.275, p<0.0001\n"
    "  • Top-2 safety avg grade vs Net EPA: r=0.280, p<0.0001\n"
    "  • Top-2 safety avg grade vs DVOA:    r=0.311, p=0.0004\n"
    "Elite safety rooms (top-2 avg ~80+) meaningfully associate with winning.\n\n"

    "MODEL ACCURACY (healthy starters only, R²=+0.084, MAE≈8 pts, Pearson r≈0.37):\n"
    "Reliable for ordinal ranking; point estimates carry ~8 grade-point uncertainty. "
    "Rare breakout years (grade jumps 20+ pts) remain hard to predict.\n\n"

    "TOOL GUIDANCE:\n"
    "• Rankings / investment → get_safety_rankings, get_cap_value_rankings\n"
    "• Injured players / buy-low → get_injured_player_projections\n"
    "• Individual profile → get_player_prediction, get_grade_trend\n"
    "• Head-to-head → compare_players\n"
    "• Team depth → get_team_safeties, get_team_safety_success_ranking\n"
    "• Position impact → get_position_team_success_correlation\n"
    "Always cite specific numbers from tool output in your answers."
)


def get_agent_executor(llm=None):
    if not HAS_LANGCHAIN:
        raise ImportError(
            "Install LangChain: pip install langchain-core langchain langchain-groq"
        )
    if llm is None:
        llm = _get_llm()
    return llm.bind_tools(TOOLS)


def _run_tool(name: str, args: dict) -> str:
    """Dispatch a tool call by name and return its string result."""
    tool_map = {t.name: t for t in TOOLS}
    if name not in tool_map:
        return f"Unknown tool: {name}"
    return tool_map[name].invoke(args)


# ===========================================================================
# CLI
# ===========================================================================

def run_cli():
    """Interactive REPL: type questions, the agent replies using the tools."""
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    _load_predictions()   # fail fast if CSV is missing
    llm_with_tools = get_agent_executor()

    print("\nNFL Safety predictions agent ready.")
    print("Example questions:")
    print("  - Who are the top 10 safeties by predicted grade?")
    print("  - Compare Kyle Hamilton and Xavier McKinney.")
    print("  - Which safeties give the best cap value?")
    print("  - Show me the grade trend for Jessie Bates III.")
    print("  - Which young safeties should we invest in?")
    print("  - What safeties are on the Ravens?")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user = input("You: ").strip()
            if not user or user.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break

            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

            # Agentic loop: keep going while the model wants to call tools
            for _ in range(10):
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    break  # Model gave a final answer

                # Execute each requested tool and append results
                for tc in response.tool_calls:
                    result = _run_tool(tc["name"], tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

            print(f"Agent: {response.content}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    if not HAS_LANGCHAIN:
        print("LangChain is required: pip install langchain-core langchain langchain-anthropic")
        raise SystemExit(1)
    run_cli()
