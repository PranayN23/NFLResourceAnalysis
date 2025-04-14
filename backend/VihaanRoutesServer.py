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

positions = ['QB', 'HB', 'WR', 'TE', 'TE', 'G', 'C', 'DL', 'ED', 'LB', 'CB', 'S']


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"message": "Server is running!", "status": "healthy"}), 200
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



if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)

    app.run(debug=True)
