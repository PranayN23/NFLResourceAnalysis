"""
NFL GM Chatbot — CB & LB
Ask natural language questions about cornerbacks and linebackers.
The chatbot parses your query, runs the LangGraph agent, and responds like a GM.

Usage:
    python -m backend.agent.gm_chatbot
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agent.cb_agent_graph import cb_gm_agent
from backend.agent.lb_agent_graph import lb_gm_agent
from backend.agent.cb_model_wrapper import CBModelInference
from backend.agent.lb_model_wrapper import LBModelInference

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CB_CSV    = os.path.join(BASE_DIR, "backend/ML/CB.csv")
LB_CSV    = os.path.join(BASE_DIR, "backend/ML/LB.csv")
CB_MODEL  = os.path.join(BASE_DIR, "backend/ML/CB_Transformers/best_cb_classifier.pth")
CB_SCALER = os.path.join(BASE_DIR, "backend/ML/CB_Transformers/cb_scaler.joblib")
LB_MODEL  = os.path.join(BASE_DIR, "backend/ML/LB_Transformers/best_lb_classifier.pth")
LB_SCALER = os.path.join(BASE_DIR, "backend/ML/LB_Transformers/lb_scaler.joblib")

# ── Load Data ──────────────────────────────────────────────────────────────────
def _load_df(path, numeric_cols):
    df = pd.read_csv(path)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

CB_NUMERIC = [
    'grades_defense', 'grades_coverage_defense', 'grades_tackle',
    'qb_rating_against', 'pass_break_ups', 'interceptions',
    'targets', 'snap_counts_corner', 'snap_counts_coverage',
    'snap_counts_slot', 'snap_counts_defense', 'Cap_Space', 'age'
]
LB_NUMERIC = [
    'grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense',
    'grades_run_defense', 'grades_tackle', 'missed_tackle_rate',
    'tackles', 'sacks', 'stops', 'total_pressures', 'snap_counts_defense', 'Cap_Space', 'age'
]

print("Loading player data...")
df_cb = _load_df(CB_CSV, CB_NUMERIC)
df_lb = _load_df(LB_CSV, LB_NUMERIC)

cb_players = set(df_cb['player'].str.lower().unique())
lb_players = set(df_lb['player'].str.lower().unique())

cb_engine = CBModelInference(CB_MODEL, scaler_path=CB_SCALER)
lb_engine = LBModelInference(LB_MODEL, scaler_path=LB_SCALER)

# ── Conversation history ───────────────────────────────────────────────────────
conversation_history = []

# ── Intent Parsing ─────────────────────────────────────────────────────────────
PARSE_PROMPT = """You are a parser for an NFL GM assistant. Extract structured intent from the user's message.

Return ONLY valid JSON (no markdown fences) with these fields:
{{
  "intent": "evaluate_player" | "compare_players" | "list_best" | "general_question",
  "players": ["Player Name"],
  "salary_ask": 0.0,
  "position": "CB" | "LB" | "both" | null
}}

Rules:
- intent "evaluate_player": user asks about 1 player (should I sign, how good is, tell me about)
- intent "compare_players": user asks to compare 2+ players
- intent "list_best": user asks who the best players are (no specific player named)
- intent "general_question": general question, no specific player
- If salary not mentioned, use 0.0
- Always return proper-cased player names (e.g. "Sauce Gardner" not "sauce gardner")

User message: {msg}"""


def parse_intent(user_msg: str) -> dict:
    resp = client.models.generate_content(
        model=MODEL,
        contents=PARSE_PROMPT.format(msg=user_msg)
    )
    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Player Lookup ──────────────────────────────────────────────────────────────
def detect_position(player_name: str):
    lo = player_name.lower()
    in_cb = lo in cb_players
    in_lb = lo in lb_players
    if in_cb and not in_lb: return "CB"
    if in_lb and not in_cb: return "LB"
    if in_cb and in_lb:     return "CB"
    return None


def get_player_history(player_name: str, position: str):
    if position == "CB":
        df = df_cb[df_cb['player'].str.lower() == player_name.lower()].copy()
        return df[df['snap_counts_defense'] >= 200].copy()
    else:
        df = df_lb[df_lb['player'].str.lower() == player_name.lower()].copy()
        return df[df['snap_counts_defense'] >= 200].copy()


# ── Agent Runner ───────────────────────────────────────────────────────────────
def run_agent(player_name: str, salary_ask: float, position: str) -> dict | None:
    history = get_player_history(player_name, position)
    if history.empty:
        return None

    state = {
        "player_name":    player_name,
        "salary_ask":     salary_ask if salary_ask > 0 else 10.0,
        "player_history": history,
        "predicted_tier": "",
        "confidence":     {},
        "valuation":      0.0,
        "decision":       "",
        "reasoning":      ""
    }

    result = cb_gm_agent.invoke(state) if position == "CB" else lb_gm_agent.invoke(state)
    conf   = result["confidence"]

    return {
        "player":              player_name,
        "position":            position,
        "predicted_tier":      result["predicted_tier"],
        "predicted_grade":     conf.get("predicted_grade", 0.0),
        "age_adjustment":      conf.get("age_adjustment", 0.0),
        "volatility_index":    conf.get("volatility_index", 0.0),
        "confidence_interval": conf.get("confidence_interval", (0.0, 0.0)),
        "fair_value":          result["valuation"],
        "salary_ask":          salary_ask if salary_ask > 0 else None,
        "decision":            result["decision"],
        "reasoning":           result["reasoning"],
    }


# ── Response Generation ────────────────────────────────────────────────────────
GM_SYSTEM = """You are a sharp NFL General Manager with deep analytical knowledge.
You have access to Time2Vec Transformer model predictions for cornerbacks and linebackers.
Speak confidently, use football language, and be direct. Keep responses concise (3-6 sentences).
When given player data, interpret it meaningfully — don't just repeat numbers back.
Reference tier names (Elite, Starter, Reserve) and cap context naturally.
If no model data is available, answer from general NFL knowledge."""


def build_context(agent_data: list | None) -> str:
    if not agent_data:
        return ""
    lines = ["\n\n[MODEL DATA]"]
    for d in agent_data:
        if d is None:
            continue
        ci = d['confidence_interval']
        line = (
            f"{d['player']} ({d['position']}): "
            f"Projected grade {d['predicted_grade']:.1f} [{ci[0]:.1f}–{ci[1]:.1f}], "
            f"Tier: {d['predicted_tier']}, Fair value: ${d['fair_value']:.1f}M, "
            f"Volatility: {d['volatility_index']:.2f}, Age penalty: {d['age_adjustment']:.1f}pts"
        )
        if d['salary_ask']:
            line += f", Asking: ${d['salary_ask']:.1f}M → {d['decision']}. {d['reasoning']}"
        lines.append(line)
    return "\n".join(lines)


def generate_response(user_msg: str, agent_data: list | None) -> str:
    context = build_context(agent_data)

    # Build message list for multi-turn context
    contents = []
    for turn in conversation_history:
        contents.append(types.Content(
            role=turn["role"],
            parts=[types.Part(text=turn["content"])]
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=f"{user_msg}{context}")]
    ))

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=GM_SYSTEM)
    )
    return resp.text.strip()


# ── Main Chat Loop ─────────────────────────────────────────────────────────────
def chat():
    print("\n" + "="*60)
    print("  NFL GM CHATBOT — CB & LB Analyst")
    print("  Ask me about any cornerback or linebacker.")
    print("  Type 'quit' to exit.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGM: Good talk. Don't overpay.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("GM: Good talk. Don't overpay.")
            break

        # Parse intent
        try:
            parsed = parse_intent(user_input)
        except Exception:
            parsed = {"intent": "general_question", "players": [], "salary_ask": 0.0, "position": None}

        agent_results = []

        if parsed["intent"] in ("evaluate_player", "compare_players"):
            for name in parsed["players"]:
                pos = detect_position(name)
                if pos is None and parsed.get("position") in ("CB", "LB"):
                    pos = parsed["position"]
                if pos is None:
                    continue
                result = run_agent(name, parsed.get("salary_ask", 0.0), pos)
                if result:
                    agent_results.append(result)

        elif parsed["intent"] == "list_best":
            pos = parsed.get("position")
            if pos in ("CB", "LB"):
                df_pos = df_cb if pos == "CB" else df_lb
                top = (df_pos[df_pos['Year'] == 2024]
                       .sort_values('grades_defense', ascending=False)
                       .head(5))
                for _, row in top.iterrows():
                    r = run_agent(row['player'], 0.0, pos)
                    if r:
                        agent_results.append(r)

        response = generate_response(user_input, agent_results if agent_results else None)

        conversation_history.append({"role": "user",  "content": user_input})
        conversation_history.append({"role": "model", "content": response})
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]

        print(f"\nGM: {response}\n")


if __name__ == "__main__":
    chat()
