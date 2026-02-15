
import pandas as pd
import numpy as np
import os
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from backend.ML.WideReceivers.WRTime2Vec import load_and_engineer_features

def analyze_player(player_name, target_year=2024):
    data, full_df = load_and_engineer_features("/Users/suhaasnachannagari/Projects/NFLResourceAnalysis/backend/ML/WR.csv")
    
    player_data = full_df[full_df["player"] == player_name].sort_values("Year")
    
    print(f"\n==== HISTORY FOR {player_name} ====")
    cols = ["Year", "grades_offense", "total_snaps", "weighted_grade", "yprr", "receptions"]
    print(player_data[cols].to_string(index=False))
    
    # Show what the model sees as "predictors" for the NEXT year
    print(f"\n==== PREDICTION INPUTS (Looking at {target_year-1} for {target_year} prediction) ====")
    row = data[(data["player"] == player_name) & (data["Year"] == target_year)]
    if not row.empty:
        # Just show a few key ones
        key_preds = ["weighted_grade_prev", "weighted_grade_trend", "grades_offense_prev", "total_snaps_prev", "yprr_prev"]
        existing = [c for c in key_preds if c in row.columns]
        print(row[existing].to_string(index=False))
    else:
        print("No prediction row found for this year.")

if __name__ == "__main__":
    analyze_player("CeeDee Lamb")
    analyze_player("Ja'Marr Chase")
    analyze_player("Justin Jefferson")
