from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import math
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Connect to MongoDB using your connection string.
mongo_uri = (
    "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
client = MongoClient(mongo_uri)

# Mapping of positions to a tuple: (snap_counts_field, ranking_field)
# For positions where snap counts is stored in multiple fields (like "S"),
# the first element is a list.
position_fields = {
    "C":   ("snap_counts_offense",    "grades_offense"),
    "CB":  ("snap_counts_corner",       "grades_defense"),
    "DI":  ("snap_counts_dl",           "grades_defense"),
    "ED":  ("snap_counts_dl",           "grades_defense"),
    "G":   ("snap_counts_offense",      "grades_offense"),
    "HB":  ("total_touches",            "grades_offense"),
    "LB":  ("snap_counts_offball",      "grades_defense"),
    "QB":  ("passing_snaps",            "grades_offense"),
    "S":   (["snap_counts_box", "snap_counts_fs", "snap_counts_coverage", "snap_counts_slot"], "grades_defense"),
    "T":   ("snap_counts_offense",      "grades_offense"),
    "TE":  ("total_snaps",              "grades_offense"),
    "WR":  ("total_snaps",              "grades_offense")
}

@app.route("/player_ranking", methods=["GET"])
def player_ranking():
    """
    Endpoint to return a ranking of players for a specified position and season year,
    aggregated across all teams (collections) in the position's namespace.
    
    Query Parameters:
      - position: The player's position (e.g., "QB", "WR", etc.)
      - year: The season year (e.g., 2021)
      - snap_counts (optional): Minimum snap counts (default is 0)
    """
    # Retrieve query parameters.
    position = request.args.get("position")
    year = request.args.get("year")
    snap_counts_threshold = request.args.get("snap_counts", "0")  # default value "0"

    # Validate presence of required parameters.
    if not position or not year:
        return jsonify({"error": "Parameters 'position' and 'year' are required"}), 400

    # Validate and convert 'year' to an integer.
    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Parameter 'year' must be an integer"}), 400

    # Validate and convert 'snap_counts' to a float.
    try:
        snap_counts_threshold = float(snap_counts_threshold)
    except ValueError:
        snap_counts_threshold = 0.0

    # Normalize the position (assumed database names are uppercase).
    position = position.upper()

    # Get the fields for the specified position.
    fields = position_fields.get(position)
    if not fields:
        return jsonify({"error": f"Unsupported position '{position}'."}), 400

    # Unpack the snap counts field(s) and ranking field.
    snap_counts_field, ranking_field = fields

    # Access the MongoDB namespace (database) corresponding to the position.
    pos_db = client[position]

    # Get all team collections within this position.
    team_collections = pos_db.list_collection_names()
    if not team_collections:
        return jsonify({"error": f"No team collections found in the '{position}' namespace."}), 404

    players = []
 
    # Build the base query with the year condition.
    query = {"Year": year}
    if isinstance(snap_counts_field, str):
        # For a single snap count field, add a simple filter.
        query[snap_counts_field] = {"$gte": snap_counts_threshold}
    elif isinstance(snap_counts_field, list):
        # For multiple snap count fields, use $expr with $add to sum them.
        add_expression = {"$add": [f"${field}" for field in snap_counts_field]}
        query["$expr"] = {"$gte": [add_expression, snap_counts_threshold]}

    # For each team collection, query for players using the constructed query.
    for team in team_collections:
        collection = pos_db[team]
        cursor = collection.find(query)
        for doc in cursor:
            # Only include players that have a valid ranking field.
            rank_value = doc.get(ranking_field)
            if rank_value is None:
                continue

            # Construct the player record.
            player_record = {
                "player": doc.get("player"),
                "team": team,
                "Year": doc.get("Year"),
                ranking_field: rank_value
            }
            # Also, include the snap counts.
            if isinstance(snap_counts_field, str):
                player_record["snap_counts"] = doc.get(snap_counts_field)
            elif isinstance(snap_counts_field, list):
                # For multiple snap count fields, compute the total snap count.
                snap_total = 0
                for field in snap_counts_field:
                    value = doc.get(field, 0) or 0
                    try:
                        snap_total += float(value)
                    except ValueError:
                        continue
                player_record["snap_counts"] = snap_total

            if doc.get("_id"):
                player_record["_id"] = str(doc.get("_id"))
            players.append(player_record)

    if not players:
        return jsonify({"error": "No players found for the given position, year, and snap counts criteria"}), 404

    # Filter out players with NaN for the ranking field.
    filtered_players = [
        p for p in players
        if p.get(ranking_field) is not None and not math.isnan(float(p[ranking_field]))
    ]

    # Sort the filtered players in descending order based on the ranking field.
    ranked_players = sorted(filtered_players, key=lambda x: float(x[ranking_field]), reverse=True)
    return jsonify(ranked_players), 200

if __name__ == '__main__':
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)
    app.run(debug=True)
