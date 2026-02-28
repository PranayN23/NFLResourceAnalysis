import os
import sys
import pandas as pd
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.agent.lb_model_wrapper import LBModelInference

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "../LB.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "best_lb_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "lb_scaler.joblib")

#MODE = "VALIDATION"  # Predict 2024 (Train < 2024)
MODE = "DREAM"       # Predict 2025 (Train <= 2024)

print(f"==== STARTING LB TIME2VEC TRANSFORMER PREDICTIONS (Mode: {MODE}) ====")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df['adjusted_value'] = pd.to_numeric(df['adjusted_value'], errors='coerce').fillna(0)
    numeric_cols = [
        'grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense',
        'grades_run_defense', 'grades_tackle', 'missed_tackle_rate',
        'tackles', 'sacks', 'stops', 'total_pressures', 'snap_counts_defense',
        'Cap_Space', 'hits', 'hurries', 'interceptions', 'pass_break_ups',
        'forced_fumbles', 'assists', 'tackles_for_loss'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['snap_counts_defense'] >= 200].copy()
    df = df.sort_values(['player', 'Year'])
    return df

df = prepare_data()

# Initialize Time2Vec Transformer wrapper
engine = LBModelInference(MODEL_OUT, scaler_path=SCALER_OUT)

# ==========================================
# 3. PREDICTIONS
# ==========================================

if MODE == "VALIDATION":
    print("\n=== VALIDATION MODE: Testing on 2024 ===")

    test_players = df[df["Year"] == 2024].copy()
    print(f"\nGenerating predictions for {len(test_players)} players...")

    results = []
    for idx, (_, row) in enumerate(test_players.iterrows()):
        name = row['player']
        history = df[(df['player'] == name) & (df['Year'] < 2024)].copy()
        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)
        results.append({
            "player": name,
            "Team": row["Team"],
            "Year": 2024,
            "Actual_Grade": row["grades_defense"],
            "Pred_Transformer": details["transformer_grade"],
            "Predicted_Grade": details["predicted_grade"],
            "Tier": tier,
            "Error": row["grades_defense"] - details["predicted_grade"],
            "Abs_Error": abs(row["grades_defense"] - details["predicted_grade"]),
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1],
            "Vol_Index": details["volatility_index"],
            "Age_Adjustment": details["age_adjustment"]
        })
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(test_players)} players...")

    final_df = pd.DataFrame(results)
    mae = final_df["Abs_Error"].mean()
    rmse = np.sqrt((final_df["Error"]**2).mean())

    print(f"\n=== VALIDATION RESULTS (2024) ===")
    print(f"Time2Vec Transformer MAE: {mae:.4f}")
    print(f"Time2Vec Transformer RMSE: {rmse:.4f}")
    print(f"Mean Error (bias): {final_df['Error'].mean():.4f}")
    print(f"Players evaluated: {len(final_df)}")

    out_path = os.path.join(os.path.dirname(__file__), "LB_2024_Validation_Results.csv")
    final_df = final_df.sort_values(by="Predicted_Grade", ascending=False)
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved Validation Report to {out_path}")

    print("\n=== TOP 10 PREDICTED LBs (2024) ===")
    display_cols = ["player", "Team", "Actual_Grade", "Predicted_Grade", "Tier", "Abs_Error"]
    print(final_df.head(10)[display_cols].to_string(index=False))

    print("\n=== LARGEST PREDICTION ERRORS ===")
    print(final_df.nlargest(5, "Abs_Error")[display_cols].to_string(index=False))

elif MODE == "DREAM":
    print("\n=== DREAM MODE: Predicting 2025 ===")

    active_2024 = df[df["Year"] == 2024].copy()
    print(f"\nGenerating 2025 projections for {len(active_2024)} players...")

    rows_2025 = []
    for idx, (_, row) in enumerate(active_2024.iterrows()):
        player = row['player']
        history = df[df["player"] == player].sort_values("Year").tail(5)
        if len(history) == 0:
            continue

        tier, details = engine.get_prediction(history)
        rows_2025.append({
            "player": player,
            "Team": row["Team"],
            "Tier": tier,
            "Predicted_Grade": details["predicted_grade"],
            "Conf_Lower": details["confidence_interval"][0],
            "Conf_Upper": details["confidence_interval"][1],
            "Vol_Index": details["volatility_index"],
            "Age_Adjustment": details["age_adjustment"],
            "Last_Grade_2024": history.iloc[-1]["grades_defense"],
            "YoY_Change_Pred": details["predicted_grade"] - history.iloc[-1]["grades_defense"],
            "Snap_Counts_2024": row["snap_counts_defense"]
        })
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(active_2024)} players...")

    final_2025 = pd.DataFrame(rows_2025).sort_values("Predicted_Grade", ascending=False)
    final_out = os.path.join(os.path.dirname(__file__), "LB_2025_Final_Rankings.csv")
    final_2025.to_csv(final_out, index=False)

    print("\n" + "="*80)
    print("TOP 15 LB RANKINGS FOR 2025 (TIME2VEC TRANSFORMER)")
    print("="*80)
    display_cols = ["player", "Team", "Tier", "Predicted_Grade",
                    "Conf_Lower", "Conf_Upper", "Vol_Index", "YoY_Change_Pred"]
    print(final_2025[display_cols].head(15).to_string(index=False))

    print("\n" + "="*80)
    print("BIGGEST PROJECTED RISERS (2024 -> 2025)")
    print("="*80)
    riser_cols = ["player", "Last_Grade_2024", "Predicted_Grade", "YoY_Change_Pred"]
    print(final_2025.nlargest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))

    print("\n" + "="*80)
    print("BIGGEST PROJECTED FALLERS (2024 -> 2025)")
    print("="*80)
    print(final_2025.nsmallest(5, "YoY_Change_Pred")[riser_cols].to_string(index=False))

    print(f"\nFull rankings saved to {final_out}")

# ==========================================
# 4. SUMMARY
# ==========================================
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"  Model: Time2Vec Transformer (4 heads, 2 layers)")
print(f"  Transformer trained on: Years <= 2022")
if MODE == "VALIDATION":
    print(f"  Tested on: Year 2024")
elif MODE == "DREAM":
    print(f"  Predicting: Year 2025")
print(f"  No data leakage: Future data excluded from training")
print("="*80)

if __name__ == "__main__":
    pass
