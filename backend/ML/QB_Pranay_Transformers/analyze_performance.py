import os
import pandas as pd
import numpy as np
import torch
import sys
import joblib
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.model_wrapper import PlayerModelInference

def analyze():
    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "best_classifier.pth")
    XGB_PATH = os.path.join(base_dir, "best_xgb.joblib")
    SCALER_PATH = os.path.join(base_dir, "player_scaler.joblib")
    CSV_PATH = os.path.join(base_dir, "../QB.csv")

    engine = PlayerModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH, xgb_path=XGB_PATH)
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(['player', 'Year'])
    
    years = range(2014, 2025)
    yearly_stats = []
    all_results = []

    print("Running Year-by-Year Multi-Factor Analysis...")
    for year in years:
        df_year = df[(df['Year'] == year) & (df['dropbacks'] >= 100)]
        y_true_tiers = []
        y_pred_tiers = []
        errors = []
        
        for _, row in df_year.iterrows():
            name = row['player']
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            
            actual_grade = row.get('grades_offense', 0.0)
            actual_tier = engine.get_tier(actual_grade)
            
            pred_tier, details = engine.get_prediction(history, mode="ensemble")
            pred_grade = details.get("predicted_grade", 0.0)
            
            if pred_tier != "No Data":
                y_true_tiers.append(actual_tier)
                y_pred_tiers.append(pred_tier)
                errors.append(actual_grade - pred_grade)
                
                # Check Transition (Elite Calibration)
                prev_year_grade = history.iloc[-1]['grades_offense'] if not history.empty else 0.0
                prev_tier = engine.get_tier(prev_year_grade)
                
                all_results.append({
                    "Year": year,
                    "Player": name,
                    "Prev_Tier": prev_tier,
                    "Actual_Tier": actual_tier,
                    "Pred_Tier": pred_tier,
                    "Error": actual_grade - pred_grade,
                    "Is_Elite_Transition": (actual_tier == "Elite" and prev_tier != "Elite"),
                    "Pred_Elite_Transition": (pred_tier == "Elite" and prev_tier != "Elite")
                })
        
        if y_true_tiers:
            acc = accuracy_score(y_true_tiers, y_pred_tiers)
            mae = np.mean(np.abs(errors))
            yearly_stats.append({
                "Year": year,
                "Accuracy": acc,
                "MAE": mae,
                "Sample_Size": len(y_true_tiers)
            })

    stats_df = pd.DataFrame(yearly_stats)
    results_df = pd.DataFrame(all_results)

    print("\n==== [1/3] YEAR-BY-YEAR ACCURACY BREAKDOWN ====")
    print(stats_df.to_string(index=False))

    print("\n==== [2/3] RESIDUAL DISTRIBUTION (2022-2024) ====")
    for y in [2022, 2023, 2024]:
        y_errs = results_df[results_df["Year"] == y]["Error"]
        print(f"{y} -> Mean Error (Bias): {y_errs.mean():.2f} | Std Dev (Volatility): {y_errs.std():.2f}")

    print("\n==== [3/3] ELITE TIER CALIBRATION (TRANSITIONS) ====")
    transitions = results_df[results_df["Is_Elite_Transition"] == True]
    if not transitions.empty:
        hits = (transitions["Actual_Tier"] == transitions["Pred_Tier"]).sum()
        print(f"Total New Elite Transitions: {len(transitions)}")
        print(f"Correctly Predicted Transitions: {hits}")
        print(f"Recall on 'Breakout' Elite Seasons: {hits/len(transitions):.2%}")
    else:
        print("No Elite Transitions found in sample.")

    # Maintenance: Out-of-Elite Transitions
    exits = results_df[results_df["Prev_Tier"] == "Elite"]
    if not exits.empty:
        print(f"\nTotal Returning Elites: {len(exits)}")
        stayed_elite = (exits["Actual_Tier"] == "Elite").sum()
        pred_stayed = (exits["Pred_Tier"] == "Elite").sum()
        print(f"Actual Elite Retention: {stayed_elite/len(exits):.2%}")
        print(f"Predicted Elite Retention: {pred_stayed/len(exits):.2%}")
        print(f"Over-Precision Bias (Elite): {pred_stayed - stayed_elite} players")

if __name__ == "__main__":
    analyze()
