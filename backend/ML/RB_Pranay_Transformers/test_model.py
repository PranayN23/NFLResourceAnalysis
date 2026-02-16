import os
import argparse
import pandas as pd
import torch
import sys
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.rb_model_wrapper import RBModelInference

def run_test():
    parser = argparse.ArgumentParser(description="Test RB Prediction Models")
    parser.add_argument("--mode", type=str, default="ensemble", choices=["transformer", "xgb", "ensemble"], help="Model component to test")
    parser.add_argument("--year_start", type=int, default=2014)
    parser.add_argument("--year_end", type=int, default=2024)
    parser.add_argument("--top_k", type=int, default=32, help="Number of top RBs by touches to evaluate per year")
    parser.add_argument("--export", action="store_true", help="Export results to CSV")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "rb_best_classifier.pth")
    XGB_PATH = os.path.join(base_dir, "rb_best_xgb.joblib")
    SCALER_PATH = os.path.join(base_dir, "rb_player_scaler.joblib")
    CSV_PATH = os.path.join(base_dir, "../HB.csv")

    if not os.path.exists(TRANSFORMER_PATH) and args.mode != "xgb":
        print(f"Error: Transformer model not found at {TRANSFORMER_PATH}")
        return

    # Initialize inference engine
    engine = RBModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH, xgb_path=XGB_PATH)
    
    # Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Convert numeric columns that might be strings
    numeric_cols = ['age', 'total_touches', 'yards', 'touchdowns', 'grades_offense', 'receptions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(['player', 'Year'])
    
    all_y_true = []
    all_y_pred = []
    export_rows = []

    print(f"\n[TESTING RB MODE: {args.mode.upper()}]")

    for year in range(args.year_start, args.year_end + 1):
        # Filter by minimum touches (50+) and get top K by touches
        df_year = df[(df['Year'] == year) & (df['total_touches'] >= 50)].sort_values('total_touches', ascending=False).head(args.top_k)
        
        if len(df_year) == 0:
            continue

        print(f"\n" + "="*100)
        print(f" PREDICTING {year} RB PERFORMANCE (Mode: {args.mode})")
        print("="*100)
        print(f"{'Player':<25} | {'Predicted Tier':<15} | {'Pred Grade':<10} | {'Actual Grade':<12} | {'Age':<4} | {'Touches'}")
        print("-"*100)

        for _, row in df_year.iterrows():
            name = row['player']
            # Prediction requires history UP TO the target year
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            
            actual_grade = row.get('grades_offense', 0.0)
            actual_tier = engine.get_tier(actual_grade)
            age = row.get('age', 0)
            touches = row.get('total_touches', 0)
            
            pred_tier, details = engine.get_prediction(history, mode=args.mode)
            pred_grade = details.get("predicted_grade", 0.0)
            
            if pred_tier != "No Data":
                all_y_true.append(actual_tier)
                all_y_pred.append(pred_tier)
            
            if args.export:
                export_rows.append({
                    "player": name,
                    "Year": year,
                    "Age": age,
                    "Touches": touches,
                    "Actual_Grade": actual_grade,
                    "Actual_Tier": actual_tier,
                    "Pred_Grade": pred_grade,
                    "Pred_Tier": pred_tier,
                    "Transformer_Grade": details.get("transformer_grade"),
                    "XGB_Grade": details.get("xgb_grade"),
                    "Age_Adjustment": details.get("age_adjustment"),
                    "Volatility_Index": details.get("volatility_index"),
                    "Mode": args.mode
                })

            print(f"{name:<25} | {pred_tier:<15} | {pred_grade:>10.1f} | {actual_grade:>12.1f} | {age:>4.0f} | {touches:>7.0f}")

        print("="*100)

    if args.export and export_rows:
        export_df = pd.DataFrame(export_rows)
        export_path = os.path.join(base_dir, f"RB_Test_Results_{args.mode}_{args.year_start}_{args.year_end}.csv")
        export_df.to_csv(export_path, index=False)
        print(f"\nResults exported to {export_path}")

    if all_y_true:
        print("\n" + "!"*35 + " AGGREGATE PERFORMANCE METRICS " + "!"*35)
        print(f"Total Evaluations: {len(all_y_true)}")
        print(f"Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, zero_division=0))
        print("!"*101)
        
        # RB-specific metrics
        print("\n" + "="*50)
        print("RB-SPECIFIC INSIGHTS")
        print("="*50)
        
        if export_rows:
            results_df = pd.DataFrame(export_rows)
            
            # Age-based accuracy
            print("\nAccuracy by Age Group:")
            for age_group, label in [((0, 25), "Young (≤25)"), ((26, 27), "Prime (26-27)"), ((28, 100), "Veteran (28+)")]:
                age_mask = results_df['Age'].between(age_group[0], age_group[1])
                if age_mask.sum() > 0:
                    group_true = results_df[age_mask]['Actual_Tier'].values
                    group_pred = results_df[age_mask]['Pred_Tier'].values
                    group_acc = accuracy_score(group_true, group_pred)
                    print(f"  {label}: {group_acc:.4f} (n={age_mask.sum()})")
            
            # Error analysis
            print("\nMean Absolute Error by Tier:")
            for tier in ['Elite', 'Starter', 'Rotation', 'Reserve/Poor']:
                tier_mask = results_df['Actual_Tier'] == tier
                if tier_mask.sum() > 0:
                    tier_mae = (results_df[tier_mask]['Actual_Grade'] - results_df[tier_mask]['Pred_Grade']).abs().mean()
                    print(f"  {tier}: {tier_mae:.2f} points (n={tier_mask.sum()})")
            
            # Aging effect
            print(f"\nAverage Age Adjustment Applied: {results_df['Age_Adjustment'].mean():.2f} points")
            print(f"Max Age Adjustment: {results_df['Age_Adjustment'].max():.2f} points")

if __name__ == "__main__":
    run_test()