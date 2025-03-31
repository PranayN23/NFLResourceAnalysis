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


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ Connected to MongoDB successfully!")
    except Exception as e:
        print("❌ MongoDB connection error:", e)

    app.run(debug=True)
