"""
DIAGNOSTIC: Wirfs Feature Vector Analysis
Prints all 20 transformer features with:
- Raw values from CSV
- Scaler mean/std
- Calculated z-scores  
- Compares to Corey Levin (bad OL player)
"""
import os
import pandas as pd
import numpy as np
import joblib

print("="*100)
print("WIRFS FEATURE VECTOR DIAGNOSTIC")
print("="*100)

# Load OL scaler to see means/stds
scaler_path = "backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"
scaler = joblib.load(scaler_path)

print(f"\nScaler loaded from: {scaler_path}")
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler scale (std) shape: {scaler.scale_.shape}")

# OL features list (from ol_model_wrapper.py)
transformer_features = [
    'adjusted_value', 'Cap_Space', 'age', 'years_in_league',
    'delta_grade', 'delta_run_block', 'delta_pass_block', 'team_performance_proxy',
    'sacks_allowed_rate', 'hits_allowed_rate', 'hurries_allowed_rate',
    'pressures_allowed_rate', 'penalties_rate', 'pass_block_efficiency',
    'snap_counts_block_share', 'snap_counts_run_block_share', 'snap_counts_pass_block_share',
    'pos_T', 'pos_G', 'pos_C'
]

print(f"\nTotal transformer features: {len(transformer_features)}")
print("Features:", transformer_features)

# Verify scaler has data for all features
print(f"\nScaler has {len(scaler.mean_)} means and {len(scaler.scale_)} stds")

# Load Wirfs data
df_ol = pd.read_csv("backend/ML/T.csv")
wirfs = df_ol[df_ol['player'].str.contains('Wirfs', case=False, na=False)].copy()

print(f"\n{'='*100}")
print(f"WIRFS DATA")
print(f"{'='*100}")
print(f"Found {len(wirfs)} rows for Wirfs")
print(f"Years: {sorted(wirfs['Year'].unique())}")

if len(wirfs) == 0:
    print("ERROR: Wirfs not found!")
else:
    # Prepare features for Wirfs (same function as wrapper uses)
    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    
    df = wirfs.copy()
    
    # Coerce numeric columns
    raw_cols = [
        'grades_offense', 'grades_run_block', 'grades_pass_block', 'Net EPA',
        'sacks_allowed', 'hits_allowed', 'hurries_allowed', 'pressures_allowed', 
        'penalties', 'pbe', 'age', 'adjusted_value', 'Cap_Space',
        'snap_counts_offense', 'snap_counts_block', 'snap_counts_run_block', 'snap_counts_pass_block'
    ]
    for col in raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.sort_values('Year')
    
    # Engineering: Basic
    df["years_in_league"] = range(len(df))
    df["delta_grade"] = df["grades_offense"].diff().fillna(0)
    df["delta_run_block"] = df["grades_run_block"].diff().fillna(0)
    df["delta_pass_block"] = df["grades_pass_block"].diff().fillna(0)
    df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
    
    # Engineering: Rates with Safe Division
    pass_snaps = df['snap_counts_pass_block'].values
    total_snaps = df['snap_counts_offense'].values
    
    df['sacks_allowed_rate'] = safe_div(df['sacks_allowed'].values, pass_snaps)
    df['hits_allowed_rate'] = safe_div(df['hits_allowed'].values, pass_snaps)
    df['hurries_allowed_rate'] = safe_div(df['hurries_allowed'].values, pass_snaps)
    df['pressures_allowed_rate'] = safe_div(df['pressures_allowed'].values, pass_snaps)
    df['penalties_rate'] = safe_div(df['penalties'].values, pass_snaps)
    df['pass_block_efficiency'] = df['pbe']  # Raw percentage
    
    df['snap_counts_block_share'] = safe_div(df['snap_counts_block'].values, total_snaps)
    df['snap_counts_run_block_share'] = safe_div(df['snap_counts_run_block'].values, total_snaps)
    df['snap_counts_pass_block_share'] = safe_div(df['snap_counts_pass_block'].values, total_snaps)
    
    # Positional one-hot
    pos = df['position'].iloc[-1].upper() if 'position' in df.columns else 'T'
    df['pos_T'] = 1.0 if 'T' in pos else 0.0
    df['pos_G'] = 1.0 if 'G' in pos else 0.0
    df['pos_C'] = 1.0 if 'C' in pos else 0.0
    
    # Use most recent row (2024 or latest)
    wirfs_row = df.iloc[-1]
    
    print(f"\nUsing row: Year {wirfs_row['Year']}, Team {wirfs_row['Team']}, Position {wirfs_row['position']}")
    print(f"\nRAW VALUES FROM CSV:")
    print(f"  sacks_allowed: {wirfs_row['sacks_allowed']}, snap_counts_pass_block: {wirfs_row['snap_counts_pass_block']}")
    print(f"  penalties: {wirfs_row['penalties']}")
    print(f"  pbe: {wirfs_row['pbe']}")
    print(f"  grades_offense: {wirfs_row['grades_offense']}")
    
    # Extract feature values for Wirfs
    wirfs_values = wirfs_row[transformer_features].values.astype(float)
    
    print(f"\n{'='*100}")
    print(f"{'Feature':<35} {'Raw Value':>15} {'Scaler Mean':>15} {'Scaler Std':>15} {'Z-Score':>15} {'Flag':>8}")
    print(f"{'='*100}")
    
    max_zscore = 0
    for i, feature in enumerate(transformer_features):
        raw_val = wirfs_values[i]
        mean = scaler.mean_[i]
        std = scaler.scale_[i]
        zscore = (raw_val - mean) / std if std != 0 else 0
        
        flag = ""
        if abs(zscore) > 5:
            flag = "⚠️ EXTREME"
            max_zscore = max(max_zscore, abs(zscore))
        elif abs(zscore) > 3:
            flag = "⚡ HIGH"
        
        print(f"{feature:<35} {raw_val:>15.4f} {mean:>15.4f} {std:>15.4f} {zscore:>15.4f} {flag:>8}")
    
    # Check specific columns
    print(f"\n{'='*100}")
    print(f"SPECIFIC CHECKS")
    print(f"{'='*100}")
    
    print(f"\nPENALTIES_RATE ANALYSIS:")
    print(f"  Raw penalties: {wirfs_row['penalties']}")
    print(f"  Denominator (snap_counts_pass_block): {wirfs_row['snap_counts_pass_block']}")
    print(f"  Computed penalties_rate: {wirfs_row['penalties_rate']:.6f}")
    print(f"  Scaler mean for penalties_rate: {scaler.mean_[transformer_features.index('penalties_rate')]:.6f}")
    print(f"  Scaler std for penalties_rate: {scaler.scale_[transformer_features.index('penalties_rate')]:.6f}")
    penalties_rate_idx = transformer_features.index('penalties_rate')
    penalties_zscore = (wirfs_row['penalties_rate'] - scaler.mean_[penalties_rate_idx]) / scaler.scale_[penalties_rate_idx]
    print(f"  Z-score: {penalties_zscore:.4f}")
    
    print(f"\nPASS_BLOCK_EFFICIENCY (PBE) ANALYSIS:")
    print(f"  Raw pbe: {wirfs_row['pbe']}")
    print(f"  Scaler mean for pass_block_efficiency: {scaler.mean_[transformer_features.index('pass_block_efficiency')]:.4f}")
    print(f"  Scaler std for pass_block_efficiency: {scaler.scale_[transformer_features.index('pass_block_efficiency')]:.4f}")
    pbe_idx = transformer_features.index('pass_block_efficiency')
    pbe_zscore = (wirfs_row['pbe'] - scaler.mean_[pbe_idx]) / scaler.scale_[pbe_idx]
    print(f"  Z-score: {pbe_zscore:.4f}")
    
    print(f"\nADJUSTED_VALUE SANITY CHECK:")
    adj_val_idx = transformer_features.index('adjusted_value')
    adj_val_zscore = (wirfs_row['adjusted_value'] - scaler.mean_[adj_val_idx]) / scaler.scale_[adj_val_idx]
    print(f"  Raw adjusted_value: {wirfs_row['adjusted_value']:.4f}")
    print(f"  Scaler mean: {scaler.mean_[adj_val_idx]:.4f}")
    print(f"  Scaler std: {scaler.scale_[adj_val_idx]:.4f}")
    print(f"  Z-score: {adj_val_zscore:.4f}")
    print(f"  ✓ In reasonable range" if abs(adj_val_zscore) <= 3 else f"  ⚠️  Outside ±3 range")

# Now compare to Corey Levin (bad OL player)
print(f"\n{'='*100}")
print(f"COMPARISON: COREY LEVIN (Replacement-Level Guard)")
print(f"{'='*100}")

df_g = pd.read_csv("backend/ML/G.csv")
corey_levin = df_g[df_g['player'] == 'Corey Levin'].copy()

if len(corey_levin) == 0:
    print("Searching for Corey Levin...")
    # Try searching by last name
    corey_candidates = df_g[df_g['player'].str.contains('Corey', case=False, na=False)]
    print(f"Found {len(corey_candidates)} Coreys in guard data:")
    if len(corey_candidates) > 0:
        print(corey_candidates[['player', 'Year', 'Team', 'grades_offense']].drop_duplicates('player'))
else:
    print(f"Found {len(corey_levin)} rows for Corey Levin")
    print(corey_levin[['player', 'Year', 'Team', 'grades_offense']].sort_values('Year'))

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
if 'wirfs_values' in locals():
    print(f"Wirfs feature vector has {sum(1 for i, f in enumerate(transformer_features) if (wirfs_values[i] - scaler.mean_[i]) / scaler.scale_[i] > 5)} features with z-score > +5")
    print(f"Wirfs feature vector has {sum(1 for i, f in enumerate(transformer_features) if (wirfs_values[i] - scaler.mean_[i]) / scaler.scale_[i] < -5)} features with z-score < -5")
    print(f"Max observed z-score magnitude: {max_zscore:.2f}")
else:
    print("Wirfs values not computed due to missing data.")
