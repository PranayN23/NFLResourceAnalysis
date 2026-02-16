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

def run_test():
    parser = argparse.ArgumentParser(description="Test QB Prediction Models")
    parser.add_argument("--mode", type=str, default="ensemble", choices=["transformer", "xgb", "ensemble"], help="Model component to test")
    parser.add_argument("--year_start", type=int, default=2014)
    parser.add_argument("--year_end", type=int, default=2024)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--export", action="store_true", help="Export results to CSV")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "best_classifier.pth")
    XGB_PATH = os.path.join(base_dir, "best_xgb.joblib")
    SCALER_PATH = os.path.join(base_dir, "player_scaler.joblib")
    CSV_PATH = os.path.join(base_dir, "../QB.csv")

    if not os.path.exists(TRANSFORMER_PATH) and args.mode != "xgb":
        print(f"Error: Transformer model not found at {TRANSFORMER_PATH}")
        return

    # Initialize inference engine
    engine = PlayerModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH, xgb_path=XGB_PATH)
    
    # Load Data
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(['player', 'Year'])
    
    all_y_true = []
    all_y_pred = []
    export_rows = []

    print(f"\n[TESTING MODE: {args.mode.upper()}]")

    for year in range(args.year_start, args.year_end + 1):
        df_year = df[(df['Year'] == year) & (df['dropbacks'] >= 100)].sort_values('dropbacks', ascending=False).head(args.top_k)
        
        if len(df_year) == 0:
            continue

        print(f"\n" + "="*85)
        print(f" PREDICTING {year} PERFORMANCE (Mode: {args.mode})")
        print("="*85)
        print(f"{'Player':<25} | {'Predicted Tier':<15} | {'Pred Grade':<10} | {'Actual Grade'}")
        print("-"*85)

        for _, row in df_year.iterrows():
            name = row['player']
            # Prediction requires history UP TO the target year
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            
            actual_grade = row.get('grades_offense', 0.0)
            actual_tier = engine.get_tier(actual_grade)
            
            pred_tier, details = engine.get_prediction(history, mode=args.mode)
            pred_grade = details.get("predicted_grade", 0.0)
            
            if pred_tier != "No Data":
                all_y_true.append(actual_tier)
                all_y_pred.append(pred_tier)
            
            if args.export:
                export_rows.append({
                    "player": name,
                    "Year": year,
                    "Actual_Grade": actual_grade,
                    "Actual_Tier": actual_tier,
                    "Pred_Grade": pred_grade,
                    "Pred_Tier": pred_tier,
                    "Transformer_Grade": details.get("transformer_grade"),
                    "XGB_Grade": details.get("xgb_grade"),
                    "Mode": args.mode
                })

            print(f"{name:<25} | {pred_tier:<15} | {pred_grade:>10.1f} | {actual_grade:>12.1f}")

        print("="*85)

    if args.export and export_rows:
        export_df = pd.DataFrame(export_rows)
        export_path = os.path.join(base_dir, f"QB_Test_Results_{args.mode}_{args.year_start}_{args.year_end}.csv")
        export_df.to_csv(export_path, index=False)
        print(f"\nResults exported to {export_path}")

    if all_y_true:
        print("\n" + "!"*30 + " AGGREGATE PERFORMANCE METRICS " + "!"*30)
        print(f"Total Evaluations: {len(all_y_true)}")
        print(f"Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, zero_division=0))
        print("!"*91)

if __name__ == "__main__":
    run_test()
