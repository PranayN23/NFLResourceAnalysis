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
START_YEAR = 2014
END_YEAR   = 2024
TOP_K      = 32
# -------------------------------------------------------------

def get_tier(grade):
    """Matches the logic in Player_Model_QB.py"""
    try:
        grade = float(grade)
        if grade >= 80.0: return "Elite"
        elif grade >= 60.0: return "Starter"
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
    
    # Pre-calculate Team Performance Proxy
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    all_y_true = []
    all_y_pred = []

    for year in range(START_YEAR, END_YEAR + 1):
        df_year = df[(df['Year'] == year) & (df['dropbacks'] >= 100)].sort_values('dropbacks', ascending=False).head(TOP_K)
        
        if len(df_year) == 0:
            continue

        print(f"\n" + "="*85)
        print(f" PREDICTING {year} PERFORMANCE (Based on data up to {year-1})")
        print("="*85)
        print(f"{'Player':<25} | {'Predicted Tier':<20} | {'Pred Grade':<10} | {'Actual Grade'}")
        print("-"*85)

        for _, row in df_year.iterrows():
            name = row['player']
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            
            actual_grade = row.get('grades_offense', 'N/A')
            actual_tier = get_tier(actual_grade)
            
            if len(history) == 0:
                pred_tier = "Rookie/No Hist"
                pred_grade = 0.0
            else:
                pred_tier, details = engine.predict(history)
                pred_grade = details.get("predicted_grade", 0.0)
                
                if actual_tier and pred_tier != "No Data":
                    all_y_true.append(actual_tier)
                    all_y_pred.append(pred_tier)

            print(f"{name:<25} | {pred_tier:<20} | {pred_grade:>10.1f} | {actual_grade}")

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
