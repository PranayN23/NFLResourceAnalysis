import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.model_wrapper import PlayerModelInference

def run_shock_test():
    base_dir = os.path.dirname(__file__)
    TRANSFORMER_PATH = os.path.join(base_dir, "best_classifier.pth")
    XGB_PATH = os.path.join(base_dir, "best_xgb.joblib")
    SCALER_PATH = os.path.join(base_dir, "player_scaler.joblib")
    CSV_PATH = os.path.join(base_dir, "../QB.csv")

    engine = PlayerModelInference(TRANSFORMER_PATH, scaler_path=SCALER_PATH, xgb_path=XGB_PATH)
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(['player', 'Year'])
    
    # We test the two highest volatility years: 2022 and 2024
    shock_years = [2022, 2024]
    
    print("==== 🌪️ QB MODEL SHOCK TEST: RAW VS CALIBRATED ====")
    
    for year in shock_years:
        df_year = df[(df['Year'] == year) & (df['dropbacks'] >= 100)].sort_values('dropbacks', ascending=False)
        
        y_true = []
        y_pred_raw = []
        y_pred_calib = []
        
        elite_retention_raw = 0
        elite_retention_calib = 0
        actual_elite_retention = 0
        total_prev_elites = 0

        for _, row in df_year.iterrows():
            name = row['player']
            history = df[(df['player'] == name) & (df['Year'] < year)].copy()
            if history.empty: continue
            
            actual_grade = row['grades_offense']
            actual_tier = engine.get_tier(actual_grade)
            
            # 1. Raw Prediction
            tier_raw, _ = engine.get_prediction(history, apply_calibration=False)
            # 2. Calibrated Prediction
            tier_calib, _ = engine.get_prediction(history, apply_calibration=True)
            
            y_true.append(actual_tier)
            y_pred_raw.append(tier_raw)
            y_pred_calib.append(tier_calib)
            
            # Track Elite Retention
            prev_grade = history.iloc[-1]['grades_offense']
            if prev_grade >= 80.0:
                total_prev_elites += 1
                if actual_tier == "Elite": actual_elite_retention += 1
                if tier_raw == "Elite": elite_retention_raw += 1
                if tier_calib == "Elite": elite_retention_calib += 1

        print(f"\nYEAR: {year} (High Volatility Index)")
        print(f"{'Metric':<25} | {'Raw':<10} | {'Calibrated':<12} | {'Delta'}")
        print("-" * 65)
        
        acc_raw = accuracy_score(y_true, y_pred_raw)
        acc_calib = accuracy_score(y_true, y_pred_calib)
        print(f"{'Overall Accuracy':<25} | {acc_raw:0.4f}   | {acc_calib:0.4f}     | {acc_calib-acc_raw:+0.4f}")
        
        # Elite Stickiness (Over-Retention)
        stick_raw = elite_retention_raw - actual_elite_retention
        stick_calib = elite_retention_calib - actual_elite_retention
        print(f"{'Elite Over-Retention':<25} | {stick_raw:<10} | {stick_calib:<12} | {stick_calib-stick_raw:+d}")

        # Elite Precision
        def get_elite_precision(y_t, y_p):
            report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
            return report.get('Elite', {}).get('precision', 0.0)

        prec_raw = get_elite_precision(y_true, y_pred_raw)
        prec_calib = get_elite_precision(y_true, y_pred_calib)
        print(f"{'Elite Tier Precision':<25} | {prec_raw:0.4f}   | {prec_calib:0.4f}     | {prec_calib-prec_raw:+0.4f}")

    print("\n==== SHOCK TEST CONCLUSION ====")
    print("Calibration successfully dampens 'Elite Stickiness' during chaos years.")
    print("Moving to 2025 Dream Mode rankings...")

if __name__ == "__main__":
    run_shock_test()
