"""
Compare predictions for Wirfs, Corey Levin, and elite tackles
"""
import pandas as pd
import numpy as np
from backend.agent.ol_model_wrapper import OLModelInference

# Initialize OL model
ol_model = OLModelInference(
    "backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth",
    "backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"
)

# Get Wirfs
print("="*100)
print("WIRFS PREDICTION")
print("="*100)
df_t = pd.read_csv("backend/ML/T.csv")
wirfs = df_t[df_t['player'].str.contains('Wirfs', case=False, na=False)].copy()

# Get recent years
wirfs_recent = wirfs[wirfs['Year'] >= 2022].sort_values('Year')
print(f"\nWirfs data from 2022 onward ({len(wirfs_recent)} rows):")
print(wirfs_recent[['player', 'Year', 'Team', 'grades_offense']])

try:
    tier, details = ol_model.predict(wirfs_recent, mode="ensemble")
    print(f"\n✓ Predicted tier: {tier}")
    print(f"  Predicted grade: {details['predicted_grade']}")
    print(f"  Transformer grade: {details['transformer_grade']}")
    print(f"  XGB grade: {details['xgb_grade']}")
    print(f"  Age adjustment: {details['age_adjustment']}")
except Exception as e:
    print(f"\n✗ Error: {e}")

# Now try Corey Levin (only has partial data)
print("\n" + "="*100)
print("COREY LEVIN PREDICTION (Replacement Level)")
print("="*100)
df_g = pd.read_csv("backend/ML/G.csv")
corey_levin = df_g[df_g['player'] == 'Corey Levin'].copy()

print(f"\nCorey Levin data ({len(corey_levin)} rows):")
print(corey_levin[['player', 'Year', 'Team', 'grades_offense']])

# Only has data with valid grades
corey_valid = corey_levin[corey_levin['grades_offense'].notna()].sort_values('Year')
if len(corey_valid) > 0:
    try:
        tier, details = ol_model.predict(corey_valid, mode="ensemble")
        print(f"\n✓ Predicted tier: {tier}")
        print(f"  Predicted grade: {details['predicted_grade']}")
        print(f"  Transformer grade: {details['transformer_grade']}")
        print(f"  XGB grade: {details['xgb_grade']}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
else:
    print("✗ No valid grades for Corey Levin")

# Compare ordering with other high/low tackles
print("\n" + "="*100)
print("COMPARISONS: ELITE vs AVERAGE TACKLES (2023-2024)")
print("="*100)

tackle_comparisons = [
    ("Jordan Mailata", 2024, 95.2),
    ("Penei Sewell", 2024, 89.6),
    ("Trent Williams", 2024, 85.6),
    ("Tristan Wirfs", 2024, 82.5),
]

results = []
for player_name, year, csv_grade in tackle_comparisons:
    p_data = df_t[(df_t['player'] == player_name) & (df_t['Year'] >= year-1)].sort_values('Year')
    if len(p_data) > 0:
        try:
            tier, details = ol_model.predict(p_data, mode="ensemble")
            results.append({
                'Player': player_name,
                'CSV_Grade': csv_grade,
                'Predicted': details['predicted_grade'],
                'Transformer': details['transformer_grade']
            })
            print(f"  {player_name:20s}: CSV Grade {csv_grade:5.1f} → Model {details['predicted_grade']:6.2f} (Trans: {details['transformer_grade']:6.2f})")
        except Exception as e:
            print(f"  {player_name:20s}: ERROR - {str(e)[:50]}")
    else:
        print(f"  {player_name:20s}: No data for year {year}")

# Check ordering
print("\n" + "="*100)
print("ORDERING CHECK")
print("="*100)
if len(results) > 0:
    df_results = pd.DataFrame(results).sort_values('CSV_Grade', ascending=False)
    print("\nOrdered by CSV grades (desc):")
    print(df_results[['Player', 'CSV_Grade', 'Predicted']])
    
    # Check if model preserves this ordering
    model_ordering = df_results.sort_values('Predicted', ascending=False)
    csv_ordering = df_results.sort_values('CSV_Grade', ascending=False)
    
    print("\nOrdered by Model predictions (desc):")
    print(model_ordering[['Player', 'CSV_Grade', 'Predicted']])
    
    # Check rank correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(df_results['CSV_Grade'], df_results['Predicted'])
    print(f"\nSpearman rank correlation (CSV vs Model): {corr:.3f} (p={pval:.3f})")
    if corr > 0.8:
        print("✓ Excellent rank correlation - ordering is CORRECT even if absolute values compressed")
    elif corr > 0.5:
        print("⚠️ Moderate rank correlation - ordering mostly correct but degraded")
    else:
        print("✗ Poor rank correlation - ordering not preserved")

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)
print("""
The model's feature vector for Wirfs looks CLEAN after the PBE fix:
- All z-scores within ±2 range
- No extreme scale mismatches detected
- Penalties_rate z-score of 1.86 is high but not extreme

If Wirfs grades as 40.92 while an elite tackle (Mailata) grades similarly low,
this suggests the issue is MODEL TRAINING (grade compression) not FEATURE SCALING.

This would be a training data issue:
- Training data may have compressed grade distribution
- Model may have learned to predict grades in a narrower range
- OR scaler was trained on a different distribution than inference data
""")
