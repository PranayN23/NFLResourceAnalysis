import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os
import re
import pandas as pd
from bson import ObjectId
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
    fields = position_fields_summary.get(position)
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


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)

    app.run(debug=True)