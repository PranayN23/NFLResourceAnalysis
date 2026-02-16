import os
import argparse
import pandas as pd
import torch
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.ol_model_wrapper import OLModelInference


def run_test():
    parser = argparse.ArgumentParser(description="Test OL Prediction Models")
    parser.add_argument("--mode", type=str, default="ensemble",
                        choices=["transformer", "xgb", "ensemble"],
                        help="Model component to test")
    parser.add_argument("--year_start", type=int, default=2014)
    parser.add_argument("--year_end", type=int, default=2024)
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(base_dir, "ol_best_classifier.pth")
    XGB_PATH = os.path.join(base_dir, "ol_best_xgb.joblib")
    SCALER_PATH = os.path.join(base_dir, "ol_player_scaler.joblib")

    # ===============================
    # LOAD OL DATA (G / C / T)
    # ===============================
    parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
    df_guard = pd.read_csv(os.path.join(parent_dir, "G.csv"))
    df_center = pd.read_csv(os.path.join(parent_dir, "C.csv"))
    df_tackle = pd.read_csv(os.path.join(parent_dir, "T.csv"))
    df = pd.concat([df_guard, df_center, df_tackle],
                   axis=0, ignore_index=True)

    # One-hot encode position
    df = pd.get_dummies(df, columns=['position'], prefix='pos')

    # ===============================
    # NUMERIC CONVERSION
    # ===============================
    numeric_cols = [
        'grades_offense', 'grades_run_block', 'grades_pass_block',
        'adjusted_value', 'Cap_Space', 'age',
        'snap_counts_offense', 'snap_counts_run_block',
        'snap_counts_pass_block', 'snap_counts_block',
        'sacks_allowed', 'hits_allowed', 'hurries_allowed',
        'pressures_allowed', 'penalties', 'pbe', 'Net EPA'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Minimum snap filter
    df = df[df['snap_counts_offense'] >= 100].copy()
    df.sort_values(['player', 'Year'], inplace=True)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    df["years_in_league"] = df.groupby("player").cumcount()

    df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
    df["delta_run_block"] = df.groupby("player")["grades_run_block"].diff().fillna(0)
    df["delta_pass_block"] = df.groupby("player")["grades_pass_block"].diff().fillna(0)

    df['team_performance_proxy'] = df.groupby(
        ['Team', 'Year'])['Net EPA'].transform('mean')

    # Rate metrics
    df['sacks_allowed_rate'] = df['sacks_allowed'] / df['snap_counts_pass_block']
    df['hits_allowed_rate'] = df['hits_allowed'] / df['snap_counts_pass_block']
    df['hurries_allowed_rate'] = df['hurries_allowed'] / df['snap_counts_pass_block']
    df['pressures_allowed_rate'] = df['pressures_allowed'] / df['snap_counts_pass_block']
    df['penalties_rate'] = df['penalties'] / df['snap_counts_offense']
    df['pass_block_efficiency'] = df['pbe']

    # Snap shares
    df['snap_counts_block_share'] = df['snap_counts_block'] / df['snap_counts_offense']
    df['snap_counts_run_block_share'] = df['snap_counts_run_block'] / df['snap_counts_offense']
    df['snap_counts_pass_block_share'] = df['snap_counts_pass_block'] / df['snap_counts_offense']

    # Lag features
    groups = df.groupby('player')
    df['lag_grades_offense'] = groups['grades_offense'].shift(1)
    df['lag_grades_run_block'] = groups['grades_run_block'].shift(1)
    df['lag_grades_pass_block'] = groups['grades_pass_block'].shift(1)
    df['delta_grade_lag'] = groups['grades_offense'].diff().shift(1)
    df['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)

    # ===============================
    # INITIALIZE MODEL
    # ===============================
    engine = OLModelInference(
        MODEL_PATH,
        scaler_path=SCALER_PATH,
        xgb_path=XGB_PATH
    )

    all_results = []
    all_y_true = []
    all_y_pred = []

    print(f"\n[TESTING OL MODE: {args.mode.upper()}]")

    # ===============================
    # YEAR LOOP
    # ===============================
    for year in range(args.year_start, args.year_end + 1):

        df_year = df[df['Year'] == year]
        if df_year.empty:
            continue

        print("\n" + "=" * 100)
        print(f"PREDICTING OL PERFORMANCE FOR {year}")
        print("=" * 100)
        print(f"{'Player':<25} | {'Pred Tier':<12} | {'Pred Grade':<10} | {'Actual Grade':<12} | {'Age'}")
        print("-" * 100)

        for _, row in df_year.iterrows():
            name = row['player']
            history = df[(df['player'] == name) &
                         (df['Year'] < year)].copy()

            if history.empty:
                continue

            actual_grade = row['grades_offense']
            actual_tier = engine.get_tier(actual_grade)
            age = row.get('age', 0)

            pred_tier, details = engine.get_prediction(
                history, mode=args.mode)

            pred_grade = details.get("predicted_grade", 0.0)

            if pred_tier != "No Data":
                all_y_true.append(actual_tier)
                all_y_pred.append(pred_tier)

            print(f"{name:<25} | {pred_tier:<12} | "
                  f"{pred_grade:>10.1f} | {actual_grade:>12.1f} | {age}")

            all_results.append({
                "player": name,
                "Year": year,
                "Team": row["Team"],
                "Actual_Grade": actual_grade,
                "Actual_Tier": actual_tier,
                "Pred_Grade": pred_grade,
                "Pred_Tier": pred_tier,
                "Age": age,
                "Position": 'G' if row.get('pos_G', 0) == 1
                else ('C' if row.get('pos_C', 0) == 1 else 'T')
            })

    results_df = pd.DataFrame(all_results)

    # ===============================
    # REGRESSION METRICS
    # ===============================
    if not results_df.empty:
        overall_mae = mean_absolute_error(
            results_df['Actual_Grade'],
            results_df['Pred_Grade']
        )
        print(f"\nOverall MAE: {overall_mae:.2f} points")

        print("\nPer-Tier MAE:")
        for tier in ['Elite', 'Starter', 'Rotation', 'Reserve/Poor']:
            tier_mask = results_df['Actual_Tier'] == tier
            if tier_mask.sum() > 0:
                tier_mae = (
                    results_df[tier_mask]['Actual_Grade'] -
                    results_df[tier_mask]['Pred_Grade']
                ).abs().mean()
                print(f"  {tier:<12}: {tier_mae:.2f} points (n={tier_mask.sum()})")

        print("\nPer-Position MAE:")
        for pos in ['G', 'C', 'T']:
            pos_mask = results_df['Position'] == pos
            if pos_mask.sum() > 0:
                pos_mae = (
                    results_df[pos_mask]['Actual_Grade'] -
                    results_df[pos_mask]['Pred_Grade']
                ).abs().mean()
                print(f"  {pos:<2}: {pos_mae:.2f} points (n={pos_mask.sum()})")

    # ===============================
    # CLASSIFICATION METRICS
    # ===============================
    if all_y_true:
        print("\n" + "!" * 35 + " AGGREGATE TIER PERFORMANCE " + "!" * 35)
        print(f"Total Evaluations: {len(all_y_true)}")
        print(f"Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")

        print("\nClassification Report:")
        print(classification_report(
            all_y_true,
            all_y_pred,
            zero_division=0
        ))
        print("!" * 101)

    # ===============================
    # EXPORT
    # ===============================
    if args.export and not results_df.empty:
        export_path = os.path.join(
            base_dir,
            f"OL_Test_Results_{args.mode}_{args.year_start}_{args.year_end}.csv"
        )
        results_df.to_csv(export_path, index=False)
        print(f"\nResults exported to {export_path}")


if __name__ == "__main__":
    run_test()
