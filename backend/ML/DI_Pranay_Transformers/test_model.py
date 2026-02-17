import os
import argparse
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.di_model_wrapper import DIModelInference


def run_test():
    parser = argparse.ArgumentParser(description="Test DI Prediction Models")
    parser.add_argument(
        "--mode", type=str, default="transformer",
        choices=["transformer", "xgb", "ensemble"],
        help="Model component to evaluate (default: transformer — XGB hurts ensemble, see validation analysis)"
    )
    parser.add_argument("--year_start", type=int, default=2014)
    parser.add_argument("--year_end",   type=int, default=2024)
    parser.add_argument(
        "--min_snaps", type=int, default=200,
        help="Minimum snap count to include a player-season (default: 200)"
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Number of top DI players by snap count to evaluate per year (default: 40)"
    )
    parser.add_argument("--export", action="store_true", help="Export per-player results to CSV")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "di_best_classifier.pth")
    XGB_PATH         = os.path.join(base_dir, "di_best_xgb.joblib")
    SCALER_PATH      = os.path.join(base_dir, "di_player_scaler.joblib")
    CSV_PATH         = os.path.join(base_dir, "../DI.csv")

    # Validate required files exist before loading anything
    if not os.path.exists(TRANSFORMER_PATH) and args.mode != "xgb":
        print(f"Error: Transformer model not found at {TRANSFORMER_PATH}")
        print("Run Player_Model_DI.py first to generate the checkpoint.")
        return
    if not os.path.exists(CSV_PATH):
        print(f"Error: Data file not found at {CSV_PATH}")
        return

    # Initialize inference engine
    engine = DIModelInference(
        TRANSFORMER_PATH,
        scaler_path=SCALER_PATH,
        xgb_path=XGB_PATH,
    )

    # --------------------------------------------------
    # Load & prepare data
    # --------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    df = df[df["position"] == "DI"].copy()

    numeric_cols = [
        "age", "snap_counts_defense", "grades_defense",
        "grades_pass_rush_defense", "grades_run_defense",
        "total_pressures", "sacks",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["player", "Year"])

    all_y_true  = []
    all_y_pred  = []
    export_rows = []

    print(f"\n[TESTING DI MODE: {args.mode.upper()}]")
    print(f"  Years:     {args.year_start} – {args.year_end}")
    print(f"  Min snaps: {args.min_snaps}")
    print(f"  Top-K:     {args.top_k} per year")

    # --------------------------------------------------
    # Year loop
    # --------------------------------------------------
    for year in range(args.year_start, args.year_end + 1):

        df_year = (
            df[(df["Year"] == year) & (df["snap_counts_defense"] >= args.min_snaps)]
            .sort_values("snap_counts_defense", ascending=False)
            .head(args.top_k)
        )

        if len(df_year) == 0:
            continue

        print(f"\n{'='*110}")
        print(f"  PREDICTING {year} DI PERFORMANCE  (mode: {args.mode})")
        print(f"{'='*110}")
        print(
            f"{'Player':<25} | {'Pred Tier':<15} | {'Pred Grade':>10} | "
            f"{'Actual Grade':>12} | {'Age':>4} | {'Snaps':>6} | {'Err':>7}"
        )
        print("-" * 110)

        for _, row in df_year.iterrows():
            name = row["player"]

            # History = all seasons STRICTLY before the target year (no leakage)
            history = df[(df["player"] == name) & (df["Year"] < year)].copy()

            actual_grade = float(row.get("grades_defense", 0.0))
            actual_tier  = engine._tier(actual_grade)
            age          = row.get("age", 0)
            snaps        = row.get("snap_counts_defense", 0)

            pred_tier, details = engine.get_prediction(history, mode=args.mode)
            pred_grade = details.get("predicted_grade", 0.0)
            error      = actual_grade - pred_grade

            if pred_tier != "No Data":
                all_y_true.append(actual_tier)
                all_y_pred.append(pred_tier)

            if args.export:
                export_rows.append({
                    "player":             name,
                    "Year":               year,
                    "Age":                age,
                    "Snaps":              snaps,
                    "Actual_Grade":       actual_grade,
                    "Actual_Tier":        actual_tier,
                    "Pred_Grade":         pred_grade,
                    "Pred_Tier":          pred_tier,
                    "Transformer_Grade":  details.get("transformer_grade"),
                    "XGB_Grade":          details.get("xgb_grade"),
                    "Age_Adjustment":     details.get("age_adjustment"),
                    "Error":              error,
                    "Abs_Error":          abs(error),
                    "Mode":               args.mode,
                })

            print(
                f"{name:<25} | {pred_tier:<15} | {pred_grade:>10.1f} | "
                f"{actual_grade:>12.1f} | {age:>4.0f} | {snaps:>6.0f} | {error:>+7.1f}"
            )

        print("=" * 110)

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    if args.export and export_rows:
        export_df  = pd.DataFrame(export_rows)
        export_path = os.path.join(
            base_dir,
            f"DI_Test_Results_{args.mode}_{args.year_start}_{args.year_end}.csv"
        )
        export_df.to_csv(export_path, index=False)
        print(f"\nResults exported to {export_path}")

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    if not all_y_true:
        print("\nNo predictions made — check year range and min_snaps threshold.")
        return

    print("\n" + "!" * 40 + " AGGREGATE METRICS " + "!" * 40)
    print(f"  Total evaluations: {len(all_y_true)}")
    print(f"  Tier accuracy:     {accuracy_score(all_y_true, all_y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    print("!" * 99)

    # --------------------------------------------------
    # DI-specific insights (requires --export to have row data)
    # --------------------------------------------------
    if not export_rows:
        print("\n(Run with --export for DI-specific breakdown)")
        return

    results_df = pd.DataFrame(export_rows)
    results_df["Age"]       = pd.to_numeric(results_df["Age"],       errors="coerce")
    results_df["Abs_Error"] = pd.to_numeric(results_df["Abs_Error"], errors="coerce")

    print("\n" + "=" * 55)
    print("DI-SPECIFIC INSIGHTS")
    print("=" * 55)

    # Overall MAE
    overall_mae  = results_df["Abs_Error"].mean()
    overall_bias = results_df["Error"].mean()
    print(f"\n  Overall MAE:  {overall_mae:.2f}")
    print(f"  Overall Bias: {overall_bias:+.2f}  "
          f"({'overestimating' if overall_bias < 0 else 'underestimating'})")

    # MAE by grade bucket
    print("\n  MAE by Actual Grade Bucket:")
    bins   = [0, 50, 60, 70, 80, 100]
    labels = ["<50", "50-60", "60-70", "70-80", "80+"]
    results_df["bucket"] = pd.cut(results_df["Actual_Grade"], bins=bins, labels=labels)
    for b in labels:
        sub = results_df[results_df["bucket"] == b]
        if len(sub):
            print(f"    {b:>6}:  MAE={sub['Abs_Error'].mean():.2f}  "
                  f"Bias={sub['Error'].mean():+.2f}  n={len(sub)}")

    # Accuracy by age group (DI-specific brackets — linemen peak later)
    print("\n  Tier Accuracy by Age Group:")
    age_groups = [((0,  26), "Young  (≤26)"),
                  ((27, 29), "Prime  (27-29)"),
                  ((30, 32), "Senior (30-32)"),
                  ((33, 99), "Veteran (33+)")]
    for (lo, hi), label in age_groups:
        mask = results_df["Age"].between(lo, hi)
        if mask.sum() > 0:
            acc = accuracy_score(
                results_df[mask]["Actual_Tier"],
                results_df[mask]["Pred_Tier"]
            )
            mae = results_df[mask]["Abs_Error"].mean()
            print(f"    {label}: acc={acc:.3f}  MAE={mae:.2f}  n={mask.sum()}")

    # MAE by tier
    print("\n  MAE by Actual Tier:")
    for tier in ["Elite", "Starter", "Rotation", "Reserve/Poor"]:
        mask = results_df["Actual_Tier"] == tier
        if mask.sum() > 0:
            print(f"    {tier:<15}: MAE={results_df[mask]['Abs_Error'].mean():.2f}  "
                  f"n={mask.sum()}")

    # Age adjustment stats
    adj_col = results_df["Age_Adjustment"].dropna()
    if len(adj_col):
        print(f"\n  Age Adjustment Applied:")
        print(f"    Mean: {adj_col.mean():.2f} pts")
        print(f"    Max:  {adj_col.max():.2f} pts")

    # Biggest misses
    print("\n  Worst Misses (|Error| > 20):")
    big_miss = (results_df[results_df["Abs_Error"] > 20]
                .sort_values("Abs_Error", ascending=False))
    if len(big_miss):
        for _, r in big_miss.iterrows():
            direction = "under" if r["Error"] > 0 else "over"
            print(f"    {r['player']:<25} {r['Year']:.0f}  "
                  f"Actual={r['Actual_Grade']:.0f}  Pred={r['Pred_Grade']:.0f}  "
                  f"Err={r['Error']:+.1f} ({direction}est)")
    else:
        print("    None")


if __name__ == "__main__":
    run_test()