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

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_CSV = os.path.join(_SCRIPT_DIR, "S_predictions_by_player.csv")
RAW_DATA_PATH   = os.path.join(_SCRIPT_DIR, "S.csv")
TEST_YEAR       = 2024
RECENT_YEARS    = 3
MIN_CAP_M       = 1.0     # ignore players with < $1M cap hit (practice squad / vets minimum)


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
    top = (
        df.nlargest(int(n), "pred_ensemble")
        [["player", "Team", "pred_ensemble", "grades_defense", "prior_grades_defense"]]
        .rename(columns={
            "pred_ensemble":       "predicted_grade",
            "grades_defense":      "actual_grade_2024",
            "prior_grades_defense":"prior_grade_2023",
        })
    )
    top.index = range(1, len(top) + 1)
    return f"Top {n} safety predictions ({TEST_YEAR}) by predicted grade:\n" + top.to_string()


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

    lines = [
        f"Player:              {player_full}",
        f"Team:                {row.get('Team', 'N/A')}",
        f"Predicted grade:     {row['pred_ensemble']:.1f}",
        f"Actual 2024 grade:   {row['grades_defense']:.1f}",
        f"Prior year grade:    {row['prior_grades_defense']:.1f}",
        f"Prediction error:    {row['ensemble_minus_actual']:.1f}  (predicted − actual)",
    ]

    # Append last RECENT_YEARS grades and cap from raw data
    if not raw.empty:
        hist = raw[raw["player"] == player_full].tail(RECENT_YEARS + 1)
        if not hist.empty:
            grade_str = ", ".join(
                f"{int(r.Year)}: {r.grades_defense:.0f}" for _, r in hist.iterrows()
            )
            lines.append(f"Grade history:       {grade_str}")
        cap = raw[(raw["player"] == player_full) & (raw["Year"] == TEST_YEAR)]["Cap_Space"]
        if not cap.empty and not pd.isna(cap.values[0]) and cap.values[0] > 0:
            lines.append(f"2024 cap hit:        ${cap.values[0]:.1f}M")
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
        lines = [
            f"--- {player_full} ({row.get('Team', 'N/A')}) ---",
            f"  Predicted grade : {row['pred_ensemble']:.1f}",
            f"  Actual 2024     : {row['grades_defense']:.1f}",
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
                    lines.append(f"  2024 cap hit    : ${cap:.1f}M")
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
    result = (
        df[mask]
        .sort_values("pred_ensemble", ascending=False)
        [["player", "pred_ensemble", "grades_defense", "prior_grades_defense"]]
        .rename(columns={
            "pred_ensemble":       "predicted_grade",
            "grades_defense":      "actual_2024",
            "prior_grades_defense":"prior_2023",
        })
    )
    team_label = df.loc[mask, "Team"].iloc[0]
    return f"Safeties on {team_label}:\n" + result.to_string(index=False)


def _tool_get_cap_value_rankings(n: int = 10) -> str:
    """
    Rank safeties by predicted grade per percentage point of cap space.
    Higher = better value for money. Useful for resource-allocation decisions.
    """
    pred = _load_predictions()
    raw  = _load_raw()
    if raw.empty or "Cap_Space" not in raw.columns:
        return "Cap data not available in S.csv."

    cap_2024 = (
        raw[raw["Year"] == TEST_YEAR][["player", "Cap_Space", "age"]]
        .dropna(subset=["Cap_Space"])
    )
    cap_2024 = cap_2024[cap_2024["Cap_Space"] >= MIN_CAP_M]    # filter practice squad / vet-min hits

    merged = pred.merge(cap_2024, on="player", how="inner")
    if merged.empty:
        return "Could not merge predictions with cap data."

    # Grade per $1M of cap hit — higher is better value
    merged["grade_per_million"] = merged["pred_ensemble"] / merged["Cap_Space"]

    top = (
        merged.nlargest(int(n), "grade_per_million")
        [["player", "Team", "pred_ensemble", "Cap_Space", "grade_per_million", "age"]]
        .rename(columns={
            "pred_ensemble":   "predicted_grade",
            "Cap_Space":       "cap_hit_M",
            "grade_per_million": "grade_per_$1M",
        })
    )
    top["cap_hit_M"] = top["cap_hit_M"].map("${:.1f}M".format)
    top.index = range(1, len(top) + 1)
    return (
        f"Top {n} safeties by cap efficiency (predicted grade ÷ cap hit in $M):\n"
        + top.to_string()
        + "\n\nNote: grade_per_$1M = predicted_grade per million dollars of cap hit. "
          "Higher = more grade value per dollar spent."
    )


def _tool_get_young_ascending_players(max_age: int = 26) -> str:
    """
    Return safeties aged <= max_age ranked by predicted grade.
    Useful for identifying high-upside players to invest in long-term.
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
    result = (
        young[["player", "Team", "age", "pred_ensemble", "grades_defense", "prior_grades_defense"]]
        .rename(columns={
            "pred_ensemble":       "predicted_grade",
            "grades_defense":      "actual_2024",
            "prior_grades_defense":"prior_2023",
        })
    )
    result.index = range(1, len(result) + 1)
    return f"Safeties age ≤ {max_age} ranked by predicted grade:\n" + result.to_string()


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
        "age-based growth projection (1.5 pts/yr) instead of pure mean reversion."
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
        """Describe how the prediction model works. Use when asked about methodology or data."""
        return _tool_get_model_info()

    TOOLS = [
        get_safety_rankings,
        get_player_prediction,
        compare_players,
        get_team_safeties,
        get_cap_value_rankings,
        get_young_ascending_players,
        get_grade_trend,
        get_model_info,
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
    "You have access to a machine-learning model that predicts 2024 PFF defensive grades "
    "for every safety in the league, along with each player's actual grade, prior-year grade, "
    "cap hit, age, and grade history. "
    "Use the tools to answer questions accurately and concisely. "
    "When asked about value or investment, consider both predicted grade AND cap efficiency. "
    "When asked about trajectory, use the grade trend tool. "
    "Always cite the specific numbers from the tools in your answers."
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
