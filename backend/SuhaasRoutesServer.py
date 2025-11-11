from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import re
import pandas as pd
from bson import ObjectId

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Connect to MongoDB
mongo_uri = (
    "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

client = MongoClient(mongo_uri)

position_fields = {
    "S": ["age", "snap_counts", "assists", "grades_defense", "grades_tackle", "forced_fumbles",
          "fumble_recoveries", "interceptions", "interception_touchdown", "missed_tackle_rate",
          "pass_break_ups", "tackles", "receptions", "touchdowns", "yards", "stops"],

    "CB": ["age", "snap_counts_defense", "grades_tackle", "grades_coverage_defense", "grades_defense",
           "interceptions", "pass_break_ups", "qb_rating_against", "receptions", "stops",
           "targets", "touchdowns", "yards"],

    "DI": ["age", "snap_counts_defense", "assists", "batted_passes", "forced_fumbles",
           "grades_defense", "grades_coverage_defense", "grades_pass_rush_defense",
           "grades_run_defense", "hits", "hurries", "missed_tackle_rate", "sacks",
           "stops", "tackles", "tackles_for_loss", "total_pressures"],

    "ED": ["age", "snap_counts_defense", "assists", "batted_passes", "forced_fumbles",
           "grades_defense", "grades_pass_rush_defense", "grades_run_defense",
           "hits", "hurries", "missed_tackle_rate", "sacks", "stops", "tackles",
           "tackles_for_loss", "total_pressures"],

    "LB": ["age", "snap_counts_defense", "assists", "batted_passes", "forced_fumbles",
           "grades_defense", "grades_coverage_defense", "grades_pass_rush_defense",
           "grades_run_defense", "grades_tackle", "hits", "interception",
           "missed_tackle_rate", "passed_break_ups", "penalties", "sacks", "stops",
           "tackles", "tackles_for_loss", "total_pressures"],

    "QB": ["age", "completion_percent", "avg_time_to_throw", "qb_rating", "interceptions",
           "sack_percent", "passing_snaps", "touchdowns", "yards", "ypa"],

    "T": ["age", "hits_allowed", "hurries_allowed", "penalties", "grades_pass_block",
          "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

    "G": ["age", "hits_allowed", "hurries_allowed", "penalties", "grades_pass_block",
          "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

    "C": ["age", "hits_allowed", "hurries_allowed", "penalties", "grades_pass_block",
          "grades_run_block", "pressures_allowed", "sacks_allowed", "snap_counts_offense"],

    "TE": ["age", "caught_percent", "contested_catch_rate", "fumbles", "grades_pass_block",
           "penalties", "receptions", "targets", "touchdowns", "yards_after_catch",
           "yards_per_reception", "total_snaps"],

    "WR": ["age", "caught_percent", "contested_catch_rate", "drop_rate", "receptions",
           "targeted_qb_rating", "targets", "touchdowns", "yards",
           "yards_after_catch_per_reception", "yprr", "total_snaps"],

    "HB": ["age", "attempts", "avoided_tackles", "breakaway_percent", "breakaway_yards",
           "elusive_rating", "explosive", "first_down", "fumbles", "grades_offense",
           "grades_run", "grades_pass", "grades_pass_block", "longest", "rec_yards",
           "receptions", "total_touches", "touchdowns", "yards", "yards_after_contact",
           "yco_attempt", "ypa", "yprr"]
}



@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"message": "Server is running!", "status": "healthy"}), 200


@app.route("/get_player_data", methods=["GET"])
def get_player_data():
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
    fields_to_return = position_fields.get(position, [])

    pos_db = client[position]
    team_players = pos_db[team]

    query = {"player": player_name, "Year": year}
    cursor = team_players.find(query)

    filtered_players = []
    for player_doc in cursor:
        filtered = {field: player_doc.get(field) for field in fields_to_return}
        # 🧹 Don't include _id
        filtered_players.append(filtered)

    if not filtered_players:
        return jsonify({"error": "Player not found"}), 404

    return jsonify(filtered_players), 200


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


@app.route("/get_player_progression_data", methods=["GET"])
def get_player_progression_data():
    player_name = request.args.get("player_name")
    position = request.args.get("position")

    if not player_name or not position:
        return jsonify({"error": "player_name and position are required"}), 400

    position = position.upper()
    pos_db = client[position]
    all_teams = pos_db.list_collection_names()

    # Get all seasons for this player
    player_seasons = []
    for team in all_teams:
        for doc in pos_db[team].find({"player": player_name}):
            player_seasons.append(doc)

    if not player_seasons:
        return jsonify({"error": "Player not found"}), 404

    # Detect which grade type is available for this player
    example_doc = next((doc for doc in player_seasons if "grades_offense" in doc or "grades_defense" in doc), None)
    if not example_doc:
        return jsonify({"error": "No grade field found for player"}), 400

    grade_field = "grades_offense" if "grades_offense" in example_doc else "grades_defense"

    player_seasons.sort(key=lambda x: x["Year"])
    rookie_year = player_seasons[0]["Year"]

    result = []

    for doc in player_seasons:
        year = doc["Year"]
        experience_year = year - rookie_year
        player_grade = doc.get(grade_field)

        if player_grade is None:
            continue

        # Gather position average for players with same experience
        grades = []
        for team in all_teams:
            for other_doc in pos_db[team].find({"Year": year}):
                other_name = other_doc.get("player")
                if not other_name:
                    continue

                other_docs = list(pos_db[team].find({"player": other_name}))
                if not other_docs:
                    continue

                other_rookie = min(d["Year"] for d in other_docs)
                other_exp = year - other_rookie

                if other_exp != experience_year:
                    continue

                grade = other_doc.get(grade_field)
                if grade is not None:
                    grades.append(grade)

        avg_grade = sum(grades) / len(grades) if grades else None

        result.append({
            "year": year,
            "experience_year": experience_year,
            "team": doc.get("Team"),
            "player_grade": player_grade,
            "position_avg": avg_grade
        })

    return jsonify(result), 200




if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)

    app.run(debug=True)
