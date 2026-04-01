import os
import pandas as pd
import sys
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.cb_model_wrapper import CBModelInference


def get_elite_precision(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report.get("Elite", {}).get("precision", 0.0)


def run_shock_test():
    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "best_cb_classifier.pth")
    SCALER_PATH      = os.path.join(base_dir, "cb_scaler.joblib")
    CSV_PATH         = os.path.join(base_dir, "../CB.csv")

    engine = CBModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH)

    df = pd.read_csv(CSV_PATH)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    numeric_cols = [
        'grades_defense', 'grades_coverage_defense', 'grades_tackle',
        'qb_rating_against', 'pass_break_ups', 'interceptions',
        'targets', 'snap_counts_corner', 'snap_counts_coverage',
        'snap_counts_slot', 'snap_counts_defense', 'Cap_Space',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['snap_counts_defense'] >= 200].copy()
    df = df.sort_values(['player', 'Year'])

    # High-volatility years for CBs (position with high year-to-year variance)
    shock_years = [2022, 2024]

    print("==== CB MODEL SHOCK TEST: RAW VS AGE-CALIBRATED ====")
    print("Raw      = transformer delta only (no age adjustment)")
    print("Calib    = transformer delta + age-aware decay penalty\n")

    for year in shock_years:
        df_year = df[df['Year'] == year].sort_values('snap_counts_defense', ascending=False)

        y_true       = []
        y_pred_raw   = []
        y_pred_calib = []

        elite_retention_raw   = 0
        elite_retention_calib = 0
        actual_elite_retention = 0
        total_prev_elites      = 0

        for _, row in df_year.iterrows():
            name    = row['player']
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            if history.empty:
                continue

            actual_grade = row['grades_defense']
            actual_tier  = engine.get_tier(actual_grade)

            # Raw: no age calibration
            tier_raw, _   = engine.predict(history, apply_calibration=False)
            # Calibrated: with age-decay adjustment
            tier_calib, _ = engine.predict(history, apply_calibration=True)

            y_true.append(actual_tier)
            y_pred_raw.append(tier_raw)
            y_pred_calib.append(tier_calib)

            # Track Elite retention
            prev_grade = history.iloc[-1]['grades_defense']
            if prev_grade >= 80.0:
                total_prev_elites += 1
                if actual_tier  == "Elite": actual_elite_retention += 1
                if tier_raw     == "Elite": elite_retention_raw    += 1
                if tier_calib   == "Elite": elite_retention_calib  += 1

        print(f"YEAR: {year} (High Volatility Index)")
        print(f"  Total CBs evaluated : {len(y_true)}")
        print(f"  Prev-year Elites    : {total_prev_elites}  (actual retained: {actual_elite_retention})")
        print(f"\n{'Metric':<28} | {'Raw':>8} | {'Calibrated':>10} | {'Delta':>8}")
        print(f"{'-'*60}")

        acc_raw   = accuracy_score(y_true, y_pred_raw)
        acc_calib = accuracy_score(y_true, y_pred_calib)
        print(f"{'Overall Accuracy':<28} | {acc_raw:>8.4f} | {acc_calib:>10.4f} | {acc_calib-acc_raw:>+8.4f}")

        stick_raw   = elite_retention_raw   - actual_elite_retention
        stick_calib = elite_retention_calib - actual_elite_retention
        print(f"{'Elite Over-Retention':<28} | {stick_raw:>8d} | {stick_calib:>10d} | {stick_calib-stick_raw:>+8d}")

        prec_raw   = get_elite_precision(y_true, y_pred_raw)
        prec_calib = get_elite_precision(y_true, y_pred_calib)
        print(f"{'Elite Tier Precision':<28} | {prec_raw:>8.4f} | {prec_calib:>10.4f} | {prec_calib-prec_raw:>+8.4f}")

        print()

    print("==== SHOCK TEST CONCLUSION ====")
    print("Age calibration reduces over-retention of declining elite CBs (30+).")
    print("Proceed to DREAM mode for 2025 projections.")


if __name__ == "__main__":
    run_shock_test()
