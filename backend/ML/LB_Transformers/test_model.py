import os
import argparse
import pandas as pd
import sys
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.lb_model_wrapper import LBModelInference


def run_test():
    parser = argparse.ArgumentParser(description="Test LB Prediction Model")
    parser.add_argument("--year_start", type=int, default=2014)
    parser.add_argument("--year_end",   type=int, default=2024)
    parser.add_argument("--top_k",      type=int, default=40,  help="Top N LBs by snap count per year")
    parser.add_argument("--export",     action="store_true",    help="Export results to CSV")
    args = parser.parse_args()

    base_dir         = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "best_lb_classifier.pth")
    SCALER_PATH      = os.path.join(base_dir, "lb_scaler.joblib")
    CSV_PATH         = os.path.join(base_dir, "../LB.csv")

    if not os.path.exists(TRANSFORMER_PATH):
        print(f"Error: Model not found at {TRANSFORMER_PATH}")
        print("Run Player_Model_LB.py first to train the model.")
        return

    engine = LBModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH)

    df = pd.read_csv(CSV_PATH)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    numeric_cols = [
        'grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense',
        'grades_run_defense', 'grades_tackle', 'missed_tackle_rate',
        'tackles', 'sacks', 'stops', 'total_pressures', 'snap_counts_defense', 'Cap_Space',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['snap_counts_defense'] >= 200].copy()
    df = df.sort_values(['player', 'Year'])

    all_y_true  = []
    all_y_pred  = []
    export_rows = []

    print(f"\n[TESTING LB TIME2VEC TRANSFORMER — Years {args.year_start}–{args.year_end}]")

    for year in range(args.year_start, args.year_end + 1):
        df_year = (
            df[df['Year'] == year]
            .sort_values('snap_counts_defense', ascending=False)
            .head(args.top_k)
        )

        if len(df_year) == 0:
            continue

        print(f"\n{'='*85}")
        print(f" PREDICTING {year} PERFORMANCE")
        print(f"{'='*85}")
        print(f"{'Player':<25} | {'Predicted Tier':<15} | {'Pred Grade':>10} | {'Actual Grade':>12}")
        print(f"{'-'*85}")

        for _, row in df_year.iterrows():
            name    = row['player']
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()

            actual_grade = row.get('grades_defense', 0.0)
            actual_tier  = engine.get_tier(actual_grade)

            pred_tier, details = engine.get_prediction(history)
            pred_grade = details.get("predicted_grade", 0.0)

            if pred_tier != "No Data":
                all_y_true.append(actual_tier)
                all_y_pred.append(pred_tier)

            if args.export:
                export_rows.append({
                    "player":            name,
                    "Year":              year,
                    "Actual_Grade":      actual_grade,
                    "Actual_Tier":       actual_tier,
                    "Pred_Grade":        pred_grade,
                    "Pred_Tier":         pred_tier,
                    "Transformer_Grade": details.get("transformer_grade"),
                    "Age_Adjustment":    details.get("age_adjustment"),
                    "Vol_Index":         details.get("volatility_index"),
                })

            print(f"{name:<25} | {pred_tier:<15} | {pred_grade:>10.1f} | {actual_grade:>12.1f}")

        print(f"{'='*85}")

    if args.export and export_rows:
        export_df   = pd.DataFrame(export_rows)
        export_path = os.path.join(base_dir, f"LB_Test_Results_{args.year_start}_{args.year_end}.csv")
        export_df.to_csv(export_path, index=False)
        print(f"\nResults exported to {export_path}")

    if all_y_true:
        print("\n" + "!" * 30 + " AGGREGATE PERFORMANCE METRICS " + "!" * 30)
        print(f"Total Evaluations : {len(all_y_true)}")
        print(f"Accuracy          : {accuracy_score(all_y_true, all_y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, zero_division=0))
        print("!" * 91)


if __name__ == "__main__":
    run_test()
