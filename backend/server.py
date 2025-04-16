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
mongo_uri = (    "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

client = MongoClient(mongo_uri)

# ✅ Define position names
positions = ['qb', 'hb', 'wr', 'te', 't', 'g', 'c', 'di', 'ed', 'lb', 'cb', 's']

position_fields = {
    "S": ["age", "snap_counts", "assists", "grades_defense", "grades_tackles", "forced_fumbles",
          "fumble_recoveries", "interceptions", "interception_touchdown", "missed_tackle_rate",
          "pass_break_ups", "tackles", "receptions", "touchdowns", "yards", "stops"],

    "CB": ["age", "snap_counts", "grades_tackles", "grades_coverage_defense", "grades_defense",
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

# Mapping each position to the corresponding weight field(s) and grade field.
# For each position, the first element represents the weight factor (what to average by)
# and the second element represents the grade to average.
position_fields = {
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
    if pos not in position_fields:
        return jsonify({"error": f"Unknown position: {position}"}), 400

    # Unpack the mapping for the position.
    weight_field, grade_field = position_fields[pos]

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
    print(pos_db)
    team_collection = pos_db[team]
    print(team_collection)
    cursor = team_collection.find({"Year": 2024}, {"player": 1, "_id": 0})
    players = [doc["player"] for doc in cursor]
    print(players)
    return jsonify(players), 200



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