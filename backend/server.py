import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import re
import sys
import json
import pandas as pd
from bson import ObjectId
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── CB/LB GM Agent imports ─────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from google import genai as _genai
    from google.genai import types as _gtypes
    from backend.agent.cb_agent_graph import cb_gm_agent
    from backend.agent.lb_agent_graph import lb_gm_agent
    from backend.agent.cb_model_wrapper import CBModelInference
    from backend.agent.lb_model_wrapper import LBModelInference

    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _df_cb = pd.read_csv(os.path.join(_BASE, "backend/ML/CB.csv"))
    _df_lb = pd.read_csv(os.path.join(_BASE, "backend/ML/LB.csv"))
    for _col in ['grades_defense','grades_coverage_defense','grades_tackle','snap_counts_defense',
                 'qb_rating_against','pass_break_ups','interceptions','targets',
                 'snap_counts_corner','snap_counts_coverage','snap_counts_slot','Cap_Space','age']:
        if _col in _df_cb.columns:
            _df_cb[_col] = pd.to_numeric(_df_cb[_col], errors='coerce')
    _df_cb['adjusted_value'] = pd.to_numeric(_df_cb['adjusted_value'], errors='coerce').fillna(0)
    for _col in ['grades_defense','grades_coverage_defense','grades_pass_rush_defense',
                 'grades_run_defense','grades_tackle','missed_tackle_rate','tackles','sacks',
                 'stops','total_pressures','snap_counts_defense','Cap_Space','age']:
        if _col in _df_lb.columns:
            _df_lb[_col] = pd.to_numeric(_df_lb[_col], errors='coerce')
    _df_lb['adjusted_value'] = pd.to_numeric(_df_lb['adjusted_value'], errors='coerce').fillna(0)

    _cb_engine = CBModelInference(
        os.path.join(_BASE, "backend/ML/CB_Transformers/best_cb_classifier.pth"),
        scaler_path=os.path.join(_BASE, "backend/ML/CB_Transformers/cb_scaler.joblib")
    )
    _lb_engine = LBModelInference(
        os.path.join(_BASE, "backend/ML/LB_Transformers/best_lb_classifier.pth"),
        scaler_path=os.path.join(_BASE, "backend/ML/LB_Transformers/lb_scaler.joblib")
    )

    _gemini = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    _GEMINI_MODEL = "gemini-2.5-flash"

    # Per-session chat histories keyed by session_id
    _chat_histories = {}

    GM_SYSTEM = """You are a sharp NFL General Manager with deep analytical knowledge.
You have access to Time2Vec Transformer model predictions for cornerbacks and linebackers.
Speak confidently, use football language, and be direct. Keep responses concise (3-6 sentences).
When given player data, interpret it meaningfully — don't just repeat numbers back.
Reference tier names (Elite, Starter, Reserve) and cap context naturally.
If no model data is available, answer from general NFL knowledge."""

    PARSE_PROMPT = """You are a parser for an NFL GM assistant. Extract structured intent from the user's message.
Return ONLY valid JSON (no markdown fences) with these fields:
{{"intent": "evaluate_player" | "compare_players" | "list_best" | "general_question", "players": ["Player Name"], "salary_ask": 0.0, "position": "CB" | "LB" | "both" | null}}
Rules:
- intent "evaluate_player": user asks about 1 player
- intent "compare_players": 2+ players
- intent "list_best": no specific player, asks for best
- intent "general_question": general question
- salary_ask: float in millions if mentioned, else 0.0
- Proper-cased player names
User message: {msg}"""

    ML_AGENTS_LOADED = True
    print("✅ CB/LB ML agents loaded")
except Exception as _e:
    ML_AGENTS_LOADED = False
    print(f"⚠️  CB/LB ML agents not loaded: {_e}")

# Map ML model feature columns to the corresponding Mongo fields
ml_to_mongo_map = {
    "Value_cap_space": "Cap_Space",
    "Previous_twp_rate": "twp_rate",
    "Previous_AV": "adjusted_value",
    "Previous_PFF": "grades_offense",  # example: PFF-like grade
    "Previous_ypa": "ypa",
    "Previous_qb_rating": "qb_rating",
    "Previous_grades_pass": "grades_pass",
    "Previous_accuracy_percent": "accuracy_percent",
    "Previous_btt_rate": "btt_rate"
}


model = tf.keras.models.load_model("backend/ML/combined_notebook/best_model.keras")
model.compile(optimizer='adam', loss='mse')  # Use your original settings

# Load your template data for scaling / feature order
df_template = pd.read_csv("backend/ML/data/Combined_QB.csv")  # adjust path if needed

# Define feature columns your model expects
feature_cols = [
    'Value_cap_space', 'Previous_twp_rate', 'Previous_AV', 'Previous_PFF',
    'Previous_ypa', 'Previous_qb_rating', 'Previous_grades_pass', 
    'Previous_accuracy_percent', 'Previous_btt_rate'
]



from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Connect to MongoDB
mongo_uri = (    "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

client = MongoClient(mongo_uri)

# ✅ Define position names
positions = ['qb', 'hb', 'wr', 'te', 't', 'g', 'c', 'di', 'ed', 'lb', 'cb', 's']
position_fields = {
  "S": [
    "grades_defense", "grades_coverage_defense", "position", "grades_tackle", "Team", "player", "age", "snap_counts_defense", "assists", 
    "forced_fumbles", "fumble_recoveries", "interceptions", "interception_touchdowns", "missed_tackle_rate", 
    "pass_break_ups", "tackles", "receptions", "touchdowns", "yards", "stops", "targets", "tackles_for_loss"
],


  "CB": [
    "grades_defense", "position", "Team", "player", "age", "snap_counts_corner", "grades_coverage_defense", 
    "interceptions", "pass_break_ups", "qb_rating_against", "receptions", "stops", "targets", "touchdowns", "yards"
],

  "DI": ["grades_defense", "position", "Team", "player", "age", "snap_counts_dl", "assists", 
         "batted_passes", "forced_fumbles", "grades_pass_rush_defense", "grades_run_defense", 
         "hits", "hurries", "missed_tackle_rate", "sacks", "stops", "tackles", "tackles_for_loss", "total_pressures"],

  "ED": ["grades_defense", "position", "Team", "player", "age", "snap_counts_defense", "assists", 
         "batted_passes", "forced_fumbles", "grades_pass_rush_defense", "grades_run_defense", "hits", "hurries", 
         "missed_tackle_rate", "sacks", "stops", "tackles", "tackles_for_loss", "total_pressures"],

  "LB": [
    "grades_defense", "position", "Team", "player", "age", "snap_counts_defense", "assists", 
    "forced_fumbles", "grades_coverage_defense", "grades_pass_rush_defense", "grades_run_defense", 
    "hits", "hurries", "missed_tackle_rate", "sacks", "stops", "tackles", "tackles_for_loss", "total_pressures",
    "grades_defense_penalty", "grades_tackle", "interceptions", "pass_break_ups", "penalties", "snap_counts_box", 
    "snap_counts_offball", "snap_counts_pass_rush", "snap_counts_run_defense", "targets"
],

  "QB": ["grades_offense", "position", "Team", "player", "age", "completion_percent", "avg_time_to_throw", 
         "qb_rating", "interceptions", "sack_percent", "passing_snaps", "touchdowns", "yards", "ypa"],

  "T": ["grades_offense", "position", "Team", "player", "age", "hits_allowed", "hurries_allowed", "penalties", 
        "grades_pass_block", "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

  "G": ["grades_offense", "position", "Team", "player", "age", "hits_allowed", "hurries_allowed", "penalties", 
        "grades_pass_block", "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

  "C": ["grades_offense", "position", "Team", "player", "age", "hits_allowed", "hurries_allowed", "penalties", 
        "grades_pass_block", "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

  "TE": ["grades_offense", "position", "Team", "player", "age", "caught_percent", "contested_catch_rate", 
         "fumbles", "grades_pass_block", "penalties", "receptions", "targets", "touchdowns", "yards_after_catch", "yards_per_reception", 
         "total_snaps"],

  "WR": ["grades_offense", "position", "Team", "player", "age", "caught_percent", "contested_catch_rate", 
         "drop_rate", "receptions", "targeted_qb_rating", "targets", "touchdowns", "yards", "yards_after_catch_per_reception", 
         "yprr", "total_snaps"],

  "HB": [
    "grades_offense", "position", "Team", "player", "age", "attempts", "avoided_tackles", 
    "breakaway_percent", "breakaway_yards", "elusive_rating", "explosive", "first_downs", 
    "fumbles", "grades_run", "longest", "rec_yards", "receptions", "total_touches", 
    "touchdowns", "yards", "yards_after_contact", "yco_attempt", "ypa", "yprr"
]

}



# Mapping each position to the corresponding weight field(s) and grade field.
# For each position, the first element represents the weight factor (what to average by)
# and the second element represents the grade to average.
position_fields_summary = {
    "C":   ("snap_counts_offense",  "grades_offense"),
    "CB":  ("snap_counts_corner",     "grades_defense"),
    "DI":  ("snap_counts_dl",         "grades_defense"),
    "ED":  ("snap_counts_dl",         "grades_defense"),
    "G":   ("snap_counts_offense",    "grades_offense"),
    "HB":  ("total_touches",          "grades_offense"),
    "LB":  ("snap_counts_offball",    "grades_defense"),
    "QB":  ("passing_snaps",          "grades_offense"),
    "S":   (["snap_counts_box", "snap_counts_fs", "snap_counts_coverage", "snap_counts_slot"], "grades_defense"),
    "T":   ("snap_counts_offense",    "grades_offense"),
    "TE":  ("total_snaps",            "grades_offense"),
    "WR":  ("total_snaps",            "grades_offense")
}

def get_last_3_years_input(player_name, position="QB", feature_cols=None):
    pos = position.upper()
    pos_db = client[pos]

    years_data = []
    for team in pos_db.list_collection_names():
        collection = pos_db[team]
        cursor = collection.find({"player": player_name}).sort("Year", -1)
        for doc in cursor:
            year_dict = {}
            for ml_col in feature_cols:
                mongo_col = ml_to_mongo_map.get(ml_col)
                val = 0.0
                if mongo_col in doc:
                    try:
                        val = float(doc[mongo_col])
                    except (ValueError, TypeError):
                        val = 0.0
                year_dict[ml_col] = val

            # Randomize Previous_AV
            if 'Previous_AV' in feature_cols:
                year_dict['Previous_AV'] = float(np.random.uniform(14, 22))

            years_data.append(year_dict)

    # Take last 3 years, pad if needed
    last_3 = years_data[:3]
    print(last_3)
    while len(last_3) < 3:
        last_3.append({col: 0.0 for col in feature_cols})
        if 'Previous_AV' in feature_cols:
            last_3[-1]['Previous_AV'] = float(np.random.uniform(14, 22))

    last_3 = last_3[::-1]  # oldest → newest
    df_input = pd.DataFrame(last_3, columns=feature_cols)
    X_input = df_input.to_numpy()[np.newaxis, :, :]
    return X_input


@app.route("/predict_player_group", methods=["POST"])
def predict_player_group():
    data = request.get_json()
    print(data)
    if not data:
        return jsonify({"error": "JSON payload required"}), 400

    player_name = data.get("player_name")
    projected_cap = data.get("projected_cap")

    if player_name is None or projected_cap is None:
        return jsonify({"error": "player_name and projected_cap are required"}), 400

    try:
        projected_cap = float(projected_cap)
    except ValueError:
        return jsonify({"error": "projected_cap must be a number"}), 400

    X_input = get_last_3_years_input(player_name, position="QB", feature_cols=feature_cols)
    print("Input\n")
    print(X_input)

    # Latest year is the last row
    X_input[0, -1, feature_cols.index("Value_cap_space")] = projected_cap
    print(X_input)
    # Make prediction
    print("About to predict")
    print(model)
    y_pred = model(X_input, training=False).numpy()
    print(y_pred)
    predicted_pff = float(y_pred[0][0])
    print("PFF\n")
    print(predicted_pff)
    # Map predicted PFF to a group tier
    if predicted_pff >= 80:
        group = "Elite"
    elif predicted_pff >= 65:
        group = "High"
    elif predicted_pff >= 50:
        group = "Medium"
    else:
        group = "Low"

    return jsonify({
        "player_name": player_name,
        "projected_cap": projected_cap,
        "predicted_pff": predicted_pff,
        "group": group
    }), 200


@app.route("/player_ranking", methods=["GET"])
def player_ranking():
    """
    Endpoint to return a ranking of players for a specified position and season
    year, aggregated across all teams (collections) in the position's namespace.

    Query Parameters
    ----------------
    position        : REQUIRED – Player position (e.g. QB, WR …).
    year            : REQUIRED – Season year (e.g. 2023).
    snap_counts     : OPTIONAL – Minimum snaps / touches (float, default 0).
    min_grade       : OPTIONAL – Minimum PFF grade to keep (float, default None).

    Example
    -------
    /player_ranking?position=QB&year=2023&min_grade=75
    """

    # ----‑‑ 1. Read & validate query params ----------------------------------
    position = request.args.get("position", "").upper()
    year_raw = request.args.get("year", "")
    snaps_raw = request.args.get("snap_counts", "0")
    grade_raw = request.args.get("min_grade", "0")      # may be None

    if not position or not year_raw:
        return jsonify({"error": "`position` and `year` are required"}), 400

    # Year --------------------------------------------------------------------
    try:
        year = int(year_raw)
    except ValueError:
        return jsonify({"error": "`year` must be an integer"}), 400

    # Snap‑count threshold (defaults to 0) ------------------------------------
    try:
        snap_counts_threshold = float(snaps_raw)
    except ValueError:
        snap_counts_threshold = 0.0

    # Grade threshold (optional) ----------------------------------------------
    grade_threshold = None
    if grade_raw is not None:
        try:
            grade_threshold = float(grade_raw)
        except ValueError:
            return jsonify({"error": "`min_grade` must be numeric"}), 400

    # ----‑‑ 2. Look‑up position metadata -------------------------------------
    fields = position_fields_summary.get(position)
    if not fields:
        return jsonify({"error": f"Unsupported position '{position}'"}), 400

    snap_counts_field, ranking_field = fields

    # ----‑‑ 3. Build the Mongo query -----------------------------------------
    query = {"Year": year}

    # (a) snap‑count filter (single field or list of fields)
    if isinstance(snap_counts_field, str):
        query[snap_counts_field] = {"$gte": snap_counts_threshold}
    else:  # list
        query["$expr"] = {
            "$gte": [{"$add": [f"${f}" for f in snap_counts_field]},
                     snap_counts_threshold]
        }

    # (b) **NEW** min‑grade filter
    if grade_threshold is not None:
        query[ranking_field] = {"$gte": grade_threshold}

    # ----‑‑ 4. Gather data across all team collections -----------------------
    pos_db = client[position]
    team_collections = pos_db.list_collection_names()
    if not team_collections:
        return jsonify({"error": f"No team data for position '{position}'"}), 404

    players = []
    for team in team_collections:
        for doc in pos_db[team].find(query):
            grade_value = doc.get(ranking_field)
            if grade_value is None:
                continue

            record = {
                "player": doc.get("player"),
                "team":   team,
                "Year":   doc.get("Year"),
                ranking_field: grade_value
            }

            # add snap totals for transparency
            if isinstance(snap_counts_field, str):
                record["snap_counts"] = doc.get(snap_counts_field)
            else:
                record["snap_counts"] = sum(
                    float(doc.get(f, 0) or 0) for f in snap_counts_field
                )

            if "_id" in doc:
                record["_id"] = str(doc["_id"])

            players.append(record)

    if not players:
        return jsonify({"error": "No players passed the filters"}), 404

    # ----‑‑ 5. Sort by grade & return ----------------------------------------
    players = [
        p for p in players
        if p.get(ranking_field) is not None and not math.isnan(float(p[ranking_field]))
    ]
    players.sort(key=lambda p: float(p[ranking_field]), reverse=True)
    return jsonify(players), 200

@app.route("/get_player_year_team", methods=["GET"])
def get_player_teams_by_year():
    # Extract the player's name from the query parameters
    player_name = request.args.get("player_name")
    if not player_name:
        return jsonify({"error": "player_name is required"}), 400

    # Initialize a dictionary to store the player's team history by year.
    # For each year, we'll use a set to avoid duplicates.
    history = {}

    # Iterate over every position in the global list 'positions'
    for pos in positions:
        # Access the database for the current position
        pos_db = client[pos]
        # Get all team collection names within that database
        team_names = pos_db.list_collection_names()
        for team in team_names:
            # Access the team collection
            team_players = pos_db[team]
            # Find all documents for the given player in this team/position
            cursor = team_players.find({"player": player_name})
            for doc in cursor:
                year = doc.get("Year")
                if year is None:
                    continue  # Skip if there's no year
                year_key = str(year)  # Use the year as a string key
                if year_key not in history:
                    history[year_key] = set()
                history[year_key].add(team)

    # Convert the sets to lists for JSON serialization
    for year_key in history:
        history[year_key] = list(history[year_key])

    result = {
        "name": player_name,
        "years": history
    }
    return jsonify(result), 200


@app.route("/get_player_year_pos_team", methods=["GET"])
def get_player_history():
    # Get the player name from the query string
    player_name = request.args.get("player_name")
    if not player_name:
        return jsonify({"error": "player_name is required"}), 400

    # Initialize a dictionary to aggregate history by year.
    # For each year, we'll use sets to avoid duplicates.
    history = {}

    # Iterate over every position in the global list "positions"
    for pos in positions:
        # Access the database for this position
        pos_db = client[pos]
        # List all team collections within the position database
        team_names = pos_db.list_collection_names()
        for team in team_names:
            # Access the team collection
            team_players = pos_db[team]
            # Find all documents for this player
            cursor = team_players.find({"player": player_name})
            for doc in cursor:
                # Get the Year from the document; skip if missing
                year = doc.get("Year")
                if year is None:
                    continue
                # Use the year (as a string) as the key
                year_key = str(year)
                # Initialize the entry for this year if needed
                if year_key not in history:
                    history[year_key] = {"teams": set(), "positions": set()}
                # Add the team and position to the corresponding sets
                history[year_key]["teams"].add(team)
                history[year_key]["positions"].add(pos)

    # Convert the sets to lists for JSON serialization
    for year_key in history:
        history[year_key]["teams"] = list(history[year_key]["teams"])
        history[year_key]["positions"] = list(history[year_key]["positions"])

    result = {
        "name": player_name,
        "years": history
    }
    return jsonify(result), 200



@app.route("/get_draft_capital", methods=["GET"])
def get_draft_capital():
    player_name = request.args.get("player_name")
    position = request.args.get("position")
    team = request.args.get("team")
    year = request.args.get("year")

    if not player_name or not position or not team or not year:
        return jsonify({"error": "player_name, team, position, and year are required"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "year must be an integer"}), 400

    position = position.upper()
    pos_db = client[position]
    team_players = pos_db[team]


    query = {"player": player_name, "Year": year}
    cursor = team_players.find(query)

    result_list = []
    for doc in cursor:
        result_list.append({
            "player": doc.get("player"),
            "year": doc.get("Year"),
            "position": doc.get("position"),
            "team": doc.get("Team"),
            "draft_rating": doc.get("adjusted_value"),
        })

    if not result_list:
        return jsonify({"error": "Player not found or Draft Capital unavailable"}), 404

    return jsonify(result_list), 200

@app.route("/get_player_data", methods=["GET"])
def get_player_grades():
    # Get the parameters from the request
    position = request.args.get('position')
    team = request.args.get('team')
    player_name = request.args.get('player_name').strip()
    year = request.args.get('year')
    pos_db = client[position]
    print(pos_db)
    team_players = pos_db[team]
    print(team_players)
    query = {"Year": year}
    
    cursor = team_players.find({"Year": 2024, "player": player_name})
    fields_to_return = position_fields.get(position, [])
    filtered_players = []
    for player_doc in cursor:
        print(player_doc)
        filtered = {field: player_doc.get(field) for field in fields_to_return}
        if player_doc.get("_id"):
            filtered["_id"] = str(player_doc.get("_id"))
        filtered_players.append(filtered)
    return jsonify(filtered_players), 200
    

@app.route("/team_pff", methods=["GET"])
def team_pff():
    """
    Return every player on <team> for <year> that meets <snap_counts>,
    sorted by his PFF grade (offense or defense).

    Query parameters
      • team  (required) – 3‑letter team code used for your collections
      • year  (required) – season (int)
      • snap_counts (optional, default 0) – minimum snaps/touches
    """
    team = request.args.get("team")
    year = request.args.get("year")
    snap_counts_threshold = request.args.get("snap_counts", "0")
    print(team)
    # -------- basic validation ----------
    if not team or not year:
        return jsonify({"error": "Parameters 'team' and 'year' are required"}), 400
    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Parameter 'year' must be an integer"}), 400
    try:
        snap_counts_threshold = float(snap_counts_threshold)
    except ValueError:
        snap_counts_threshold = 0.0

    players = []
    all_positions = ["QB", "HB", "WR", "TE", "T", "G", "C", "ED", "DI", "LB", "CB", "S"]
    # -------- iterate over every POSITION DB -------------
    for pos in all_positions:
        print(pos)
        snap_field, grade_field = position_fields_summary[pos]

        pos_db = client[pos]
        print(pos_db.list_collection_names())
        if team not in pos_db.list_collection_names():
            continue                                     # team didn’t use this position that year
        col = pos_db[team]
        print(col)
        # ---- build year + snap filter --------------
        query = {"Year": year}
        if isinstance(snap_field, str):
            query[snap_field] = {"$gte": snap_counts_threshold}
        else:  # list of snap fields (e.g., Safety)
            add_expr = {"$add": [f"${f}" for f in snap_field]}
            query["$expr"] = {"$gte": [add_expr, snap_counts_threshold]}

        for doc in col.find(query):
            print(doc)
            grade = doc.get(grade_field)
            if grade is None or (isinstance(grade, float) and math.isnan(grade)):
                continue

            # -------- build uniform record ------------
            record = {
                "player":  doc.get("player"),
                "position": pos,
                "pff_score": grade,                     # <- renamed
                "snap_counts": (
                    doc.get(snap_field) if isinstance(snap_field, str)
                    else sum(float(doc.get(f, 0) or 0) for f in snap_field)
                ),
                "Year": year,
                "_id": str(doc.get("_id"))
            }
            players.append(record)
    print(players)
    if not players:
        return jsonify({"error": "No players found for that query."}), 404

    # sort highest grade first
    players.sort(key=lambda x: float(x["pff_score"]), reverse=True)
    return jsonify(players), 200

@app.route('/weighted-average-grade', methods=['GET'])
def weighted_average_grade():
    """
    Expects a JSON payload with:
      - team: e.g., "49ers"
      - position: e.g., "QB"
      - year: e.g., 2021
      
    The endpoint will query the MongoDB collection for the provided position (e.g., "QB")
    and compute the weighted average grade based on the provided mapping:
      - The weight field(s) is used to compute the total weight.
      - The grade field is multiplied by the respective weight for each player.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    team = data.get("team")
    position = data.get("position")
    year = data.get("year")

    if not team or not position or not year:
        return jsonify({"error": "Missing required parameters: team, position, and year"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Year must be an integer"}), 400

    # Ensure the position is standardized to uppercase to match the keys in our mapping.
    pos = position.upper()
    if pos not in position_fields_summary:
        return jsonify({"error": f"Unknown position: {position}"}), 400

    # Unpack the mapping for the position.
    weight_field, grade_field = position_fields_summary[pos]

    # Access the relevant MongoDB collection based on the position.
    collection = client[pos]

    # Query documents for the given team and year.
    # Adjust field names ("Team" and "Year") if needed based on your MongoDB documents.
    query = {"Team": team, "Year": year}
    cursor = collection.find(query)

    total_weighted_grade = 0.0
    total_weight = 0.0

    # Iterate through the documents.
    for doc in cursor:
        # Retrieve and convert the grade value.
        try:
            grade = float(doc.get(grade_field, 0))
        except (ValueError, TypeError):
            grade = 0.0

        # Retrieve the weighting value(s).
        weight_value = 0.0
        if isinstance(weight_field, list):
            # For a list of fields (example: for "S" position), sum their values.
            for wf in weight_field:
                try:
                    weight_value += float(doc.get(wf, 0))
                except (ValueError, TypeError):
                    weight_value += 0.0
        else:
            try:
                weight_value = float(doc.get(weight_field, 0))
            except (ValueError, TypeError):
                weight_value = 0.0

        # Only use documents with a positive weight.
        if weight_value > 0:
            total_weighted_grade += grade * weight_value
            total_weight += weight_value

    # Calculate the weighted average grade.
    weighted_average = total_weighted_grade / total_weight if total_weight > 0 else 0.0

    return jsonify({
        "team": team,
        "position": position,
        "year": year,
        "weighted_average_grade": weighted_average
    }), 200

@app.route("/get_pos_team_name", methods=["GET"])
def get_latest_player_history():
    pos = request.args.get("pos")
    team = request.args.get("team")
    print(pos)
    print(team)
    if not pos or not team:
        return jsonify({"error": "Missing 'pos' or 'team' parameter"}), 400

    pos_db = client[pos]
    team_collection = pos_db[team]
    cursor = team_collection.find({"Year": 2024}, {"player": 1, "_id": 0})
    players = [doc["player"] for doc in cursor]
    print(players)
    return jsonify(players), 200


@app.route("/get_pos_team", methods=["GET"])
def get_players_for_pos_team():
    pos = request.args.get("pos")
    team = request.args.get("team")
    print(pos)
    print(team)
    if not pos or not team:
        return jsonify({"error": "Missing 'pos' or 'team' parameter"}), 400

    pos_db = client[pos]
    print(pos_db)
    team_collection = pos_db[team]
    print(team_collection)
    cursor = team_collection.find({"Year": 2024})
    # Build a list of filtered documents
    filtered_players = []
    fields_to_return = position_fields.get(pos, [])

    for player_doc in cursor:
        filtered = {}
        contains_nan = False

        for field in fields_to_return:
            value = player_doc.get(field)
            if isinstance(value, float) and math.isnan(value):
                contains_nan = True
                break
            elif value is None:
                contains_nan = True
                break
            filtered[field] = value

        if contains_nan:
            continue

        if player_doc.get("_id"):
            filtered["_id"] = str(player_doc.get("_id"))

        filtered_players.append(filtered)
    return jsonify(filtered_players), 200

@app.route("/get_pos", methods=["GET"])
def get_pos_data():
    pos = request.args.get("pos")
    print(pos)
    if not pos:
        return jsonify({"error": "Missing 'pos' parameter"}), 400

    pos_db = client[pos]
    print(pos_db)
    team_collections = pos_db.list_collection_names()
    filtered_players = []
    fields_to_return = position_fields.get(pos, [])

    for team in team_collections:
        collection = pos_db[team]
        cursor = collection.find({"Year": 2024})

        for player_doc in cursor:
            filtered = {}
            contains_nan = False

            for field in fields_to_return:
                value = player_doc.get(field)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    contains_nan = True
                    break
                filtered[field] = value

            if contains_nan:
                continue

            if player_doc.get("_id"):
                filtered["_id"] = str(player_doc.get("_id"))

            filtered_players.append(filtered)

    return jsonify(filtered_players), 200



@app.route("/get_team", methods=["GET"])
def get_team_data():
    team = request.args.get("team")
    print(team)
    if not team:
        return jsonify({"error": "Missing 'team' parameter"}), 400

    filtered_players = []

    for pos in positions:
        pos = pos.upper()
        try:
            pos_db = client[pos]              # Database = position
            collection = pos_db[team]         # Collection = team
            fields_to_return = position_fields.get(pos, [])  # Use UPPER key for field access
            
            cursor = collection.find({"Year": 2024})  # ← no need to filter by "position"
            print(collection)
            for player_doc in cursor:
                print(player_doc.get("player"))
                filtered = {}
                contains_nan = False

                for field in fields_to_return:
                    value = player_doc.get(field)
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        contains_nan = True
                        break
                    filtered[field] = value

                if contains_nan:
                    continue

                if player_doc.get("_id"):
                    filtered["_id"] = str(player_doc.get("_id"))

                filtered_players.append(filtered)

        except Exception as e:
            print(f"Error accessing {pos} for team {team}: {e}")
            continue  # Skip if team collection doesn't exist for a position

    return jsonify(filtered_players), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"message": "Server is running!", "status": "healthy"}), 200


@app.route("/upload_data", methods=["POST"])
def upload_data():
    # Load data for each position CSV
    for position in positions:
        file_path = f"{position.upper()}.csv"  # File name convention: QB.csv, RB.csv, etc.
        # Check if the file exists
        if os.path.exists(file_path):
            try:
                # Read the CSV data into a DataFrame
                df = pd.read_csv(file_path)

                # Convert the DataFrame to a list of dictionaries
                data = df.to_dict(orient="records")
                

                # Connect to the database specific to the position
                position_db = client[position.upper()]  # e.g., 'qb', 'rb', etc.

                # Now, insert the data into collections based on teams
                for record in data:
                    team_name = record.get('Team')  # Get the 'Team' column value

                    # Ensure a collection exists for the team
                    team_collection = position_db[team_name]  # Collection is named after the team

                    # Insert the record into the team collection
                    team_collection.insert_one(record)
                print(f"✅ Data uploaded successfully for {position.upper()} position!")
            except Exception as e:
                print(f"❌ Error while uploading data for {position.upper()}: {str(e)}")
        else:
            print(f"❌ File for {position.upper()} does not exist at {file_path}")

    return jsonify({"message": "Data uploaded for all available positions!"}), 200


@app.route("/login", methods=["POST"])
def login_handler():
    try:
        data = request.get_json()
        email = data.get("username")
        password = data.get("password")

        print("\n🔹 LOGIN REQUEST RECEIVED")
        print(f"🔸 Email: {email}, Password: {'*' * len(password) if password else ''}")

        if not email or not password:
            return jsonify({"message": "Email and password are required."}), 400
        db = client["users"]
        collection = db["users"]

        user = collection.find_one({"email": email})
        if not user:
            return jsonify({"message": "Incorrect username. User does not exist"}), 401

        if user["password"] != password:
            return jsonify({"message": "Incorrect password."}), 401

        user["_id"] = str(user["_id"])
        return jsonify(user), 200

    except Exception as e:
        print(f"❌ LOGIN ERROR: {e}")
        return jsonify({"message": "Internal server error."}), 500


# ✅ Signup Route (Requires Resume Upload)
@app.route("/signup", methods=["POST"])
def signup_handler():
    try:
        email = request.form.get("username")
        password = request.form.get("password")

        print("\n🔹 SIGNUP REQUEST RECEIVED")

        if not email or not password:
            return jsonify({"message": "Username and password are required."}), 400
        db = client["users"]
        collection = db["users"]

        existing_user = collection.find_one({"email": email})
        if existing_user:
            return jsonify({"message": "User already exists."}), 409


        new_user = {
            "email": email,
            "password": password,
        }
        insert_result = collection.insert_one(new_user)
        if not insert_result.acknowledged:
            return jsonify({"message": "User creation failed."}), 500

        return jsonify(
            {
                "id": str(insert_result.inserted_id),
                "email": email,
            }
        ), 201

    except Exception as e:
        print(f"❌ SIGNUP ERROR: {e}")
        return jsonify({"message": f"Internal Server Error: {e}"}), 500
    

@app.route("/get_player_cap_space", methods=["GET"])
def get_player_cap_space():
    player_name = request.args.get("player_name")
    position = request.args.get("position")
    team = request.args.get("team")
    year = request.args.get("year")

    if not player_name or not position or not team or not year:
        return jsonify({"error": "player_name, team, position, and year are required"}), 400

    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "year must be an integer"}), 400

    position = position.upper()
    pos_db = client[position]
    team_players = pos_db[team]

    query = {"player": player_name, "Year": year}
    cursor = team_players.find(query)

    result_list = []
    for doc in cursor:
        result_list.append({
            "player": doc.get("player"),
            "year": doc.get("Year"),
            'team': doc.get("Team"),
            "position": doc.get("position"),
            "Cap_Space": doc.get("Cap_Space")
        })

    if not result_list:
        return jsonify({"error": "Player not found or Cap_Space unavailable"}), 404

    return jsonify(result_list), 200


# ── CB/LB GM Agent Routes ──────────────────────────────────────────────────────

def _run_agent(player_name, salary_ask, position):
    if position == "CB":
        history = _df_cb[_df_cb['player'].str.lower() == player_name.lower()]
        history = history[history['snap_counts_defense'] >= 200].copy()
    else:
        history = _df_lb[_df_lb['player'].str.lower() == player_name.lower()]
        history = history[history['snap_counts_defense'] >= 200].copy()
    if history.empty:
        return None
    state = {
        "player_name": player_name,
        "salary_ask": salary_ask if salary_ask > 0 else 10.0,
        "player_history": history,
        "predicted_tier": "", "confidence": {}, "valuation": 0.0,
        "decision": "", "reasoning": ""
    }
    result = cb_gm_agent.invoke(state) if position == "CB" else lb_gm_agent.invoke(state)
    conf = result["confidence"]
    ci = conf.get("confidence_interval", (0.0, 0.0))
    return {
        "player": player_name, "position": position,
        "predicted_tier": result["predicted_tier"],
        "predicted_grade": round(conf.get("predicted_grade", 0.0), 1),
        "age_adjustment": round(conf.get("age_adjustment", 0.0), 1),
        "volatility_index": round(conf.get("volatility_index", 0.0), 2),
        "conf_low": round(ci[0], 1), "conf_high": round(ci[1], 1),
        "fair_value": result["valuation"],
        "decision": result["decision"],
        "reasoning": result["reasoning"],
    }


def _detect_position(name):
    lo = name.lower()
    in_cb = lo in set(_df_cb['player'].str.lower().unique())
    in_lb = lo in set(_df_lb['player'].str.lower().unique())
    if in_cb and not in_lb: return "CB"
    if in_lb and not in_cb: return "LB"
    if in_cb: return "CB"
    return None


@app.route("/evaluate-player", methods=["POST"])
def evaluate_player():
    if not ML_AGENTS_LOADED:
        return jsonify({"error": "ML agents not available"}), 503
    data = request.json or {}
    name = data.get("player_name", "").strip()
    salary = float(data.get("salary_ask", 0.0))
    position = data.get("position") or _detect_position(name)
    if not name:
        return jsonify({"error": "player_name required"}), 400
    if not position:
        return jsonify({"error": f"Player '{name}' not found in CB or LB database"}), 404
    result = _run_agent(name, salary, position)
    if result is None:
        return jsonify({"error": f"No qualifying seasons found for '{name}'"}), 404
    return jsonify(result), 200


@app.route("/gm-chat", methods=["POST"])
def gm_chat():
    if not ML_AGENTS_LOADED:
        return jsonify({"error": "ML agents not available"}), 503
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    position_filter = data.get("position")  # "CB", "LB", or None
    if not user_msg:
        return jsonify({"error": "message required"}), 400

    # Parse intent
    try:
        parse_resp = _gemini.models.generate_content(
            model=_GEMINI_MODEL,
            contents=PARSE_PROMPT.format(msg=user_msg)
        )
        raw = parse_resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        parsed = json.loads(raw.strip())
    except Exception:
        parsed = {"intent": "general_question", "players": [], "salary_ask": 0.0, "position": None}

    # Constrain position to page context if provided
    if position_filter and not parsed.get("position"):
        parsed["position"] = position_filter

    agent_results = []
    if parsed["intent"] in ("evaluate_player", "compare_players"):
        for pname in parsed.get("players", []):
            pos = _detect_position(pname) or parsed.get("position")
            if pos:
                r = _run_agent(pname, parsed.get("salary_ask", 0.0), pos)
                if r: agent_results.append(r)
    elif parsed["intent"] == "list_best":
        pos = parsed.get("position") or position_filter
        if pos in ("CB", "LB"):
            df_pos = _df_cb if pos == "CB" else _df_lb
            top = df_pos[df_pos['Year'] == 2024].sort_values('grades_defense', ascending=False).head(5)
            for _, row in top.iterrows():
                r = _run_agent(row['player'], 0.0, pos)
                if r: agent_results.append(r)

    # Build context string
    context = ""
    if agent_results:
        context = "\n\n[MODEL DATA]\n"
        for d in agent_results:
            context += (f"{d['player']} ({d['position']}): grade {d['predicted_grade']} "
                       f"[{d['conf_low']}–{d['conf_high']}], tier {d['predicted_tier']}, "
                       f"fair value ${d['fair_value']}M, vol {d['volatility_index']}, "
                       f"age penalty {d['age_adjustment']}pts")
            context += f", asking ${parsed.get('salary_ask',0)}M → {d['decision']}. {d['reasoning']}\n" \
                if parsed.get('salary_ask', 0) > 0 else "\n"

    # Build multi-turn history
    history = _chat_histories.get(session_id, [])
    contents = []
    for turn in history:
        contents.append(_gtypes.Content(role=turn["role"], parts=[_gtypes.Part(text=turn["content"])]))
    contents.append(_gtypes.Content(role="user", parts=[_gtypes.Part(text=user_msg + context)]))

    resp = _gemini.models.generate_content(
        model=_GEMINI_MODEL, contents=contents,
        config=_gtypes.GenerateContentConfig(system_instruction=GM_SYSTEM)
    )
    reply = resp.text.strip()

    # Update history
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "model", "content": reply})
    _chat_histories[session_id] = history[-20:]

    return jsonify({
        "reply": reply,
        "agent_data": agent_results,
        "intent": parsed.get("intent")
    }), 200


@app.route("/gm-chat/reset", methods=["POST"])
def reset_chat():
    session_id = (request.json or {}).get("session_id", "default")
    _chat_histories.pop(session_id, None)
    return jsonify({"status": "reset"}), 200


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)

    app.run(debug=True)