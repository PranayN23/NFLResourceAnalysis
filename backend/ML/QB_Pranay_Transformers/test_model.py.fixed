import os
import argparse
import pandas as pd
import torch
import sys
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.model_wrapper import PlayerModelInference

# --- CONFIGURATION (Change these values to run different tests) ---
START_YEAR = 2022
END_YEAR   = 2024
TOP_K      = 10
# -------------------------------------------------------------

def get_tier(grade):
    """Matches the logic in Player_Model_QB.py"""
    try:
        grade = float(grade)
        if grade >= 80.0: return "Elite/High Quality"
        elif grade >= 60.0: return "Starter/Average"
        else: return "Reserve/Poor"
    except (ValueError, TypeError):
        return None

def run_test():
    base_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(base_dir, "best_classifier.pth")
    SCALER_PATH = os.path.join(base_dir, "player_scaler.joblib")
    CSV_PATH = os.path.join(base_dir, "../../ML/QB.csv")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Initialize inference engine
    engine = PlayerModelInference(MODEL_PATH, scaler_path=SCALER_PATH)
    
    # Load Data
    df = pd.read_csv(CSV_PATH)
    # Handle Data Issues
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce')
    
    # Pre-calculate Team Performance Proxy (Required for model accuracy)
    # This matches the logic in Player_Model_QB.py
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    all_y_true = []
    all_y_pred = []

    for year in range(START_YEAR, END_YEAR + 1):
        # Get top K QBs for the current season in the loop
        # Only consider players with significant dropbacks to match model focus
        df_year = df[(df['Year'] == year) & (df['dropbacks'] >= 100)].sort_values('dropbacks', ascending=False).head(TOP_K)
        
        if len(df_year) == 0:
            continue

        print(f"\n" + "="*85)
        print(f" PREDICTING {year} PERFORMANCE (Based on data up to {year-1})")
        print("="*85)
        print(f"{'Player':<25} | {'Predicted Tier':<20} | {'Conf %':<10} | {'Actual Grade'}")
        print("-"*85)

        for _, row in df_year.iterrows():
            name = row['player']
            
            # Predict Year X using data from before Year X
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            
            actual_grade = row.get('grades_offense', 'N/A')
            actual_tier = get_tier(actual_grade)
            
            if len(history) == 0:
                pred = "Rookie/No Hist"
                conf_val = 0.0
            else:
                pred, confs = engine.predict(history)
                conf_val = confs.get(pred, 0.0)
                if np.isnan(conf_val):
                    conf_val = 0.0
                conf_val *= 100
                
                if actual_tier and pred != "No Data":
                    all_y_true.append(actual_tier)
                    all_y_pred.append(pred)

            print(f"{name:<25} | {pred:<20} | {conf_val:>6.1f}% | {actual_grade}")

        print("="*85)

    if all_y_true:
        print("\n" + "!"*30 + " AGGREGATE PERFORMANCE METRICS " + "!"*30)
        print(f"Total Evaluations: {len(all_y_true)}")
        print(f"Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, zero_division=0))
        print("!"*91)

if __name__ == "__main__":
    run_test()
