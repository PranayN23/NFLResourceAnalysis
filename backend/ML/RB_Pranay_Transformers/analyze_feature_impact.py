#!/usr/bin/env python
"""
Analyze the impact of new features on model performance
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.ML.RB_Pranay_Transformers.Player_Model_RB import add_ol_features, add_rolling_stats

print("="*80)
print("FEATURE IMPACT ANALYSIS")
print("="*80)

# Load data
rb_df = pd.read_csv('backend/ML/HB.csv')
rb_df = rb_df[rb_df['total_touches'] >= 50].copy()
rb_df.sort_values(by=['player', 'Year'], inplace=True)

# Basic engineering
rb_df["years_in_league"] = rb_df.groupby("player").cumcount()
rb_df["delta_grade"] = rb_df.groupby("player")["grades_offense"].diff().fillna(0)
rb_df["delta_yards"] = rb_df.groupby("player")["yards"].diff().fillna(0)
rb_df["delta_touches"] = rb_df.groupby("player")["total_touches"].diff().fillna(0)
rb_df['team_performance_proxy'] = rb_df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')

# Add new features
rb_df, ol_cols = add_ol_features(rb_df)
rb_df, rolling_cols = add_rolling_stats(rb_df)

print(f"\n[1] Feature Counts:")
print(f"  OL features: {len(ol_cols)}")
print(f"  Rolling stats: {len(rolling_cols)}")
print(f"  Total new features: {len(ol_cols) + len(rolling_cols)}")

# Check correlations
print(f"\n[2] Correlations with grades_offense:")
all_new_features = ol_cols + rolling_cols
correlations = []
for feat in all_new_features:
    if feat in rb_df.columns:
        corr = rb_df[[feat, 'grades_offense']].corr().iloc[0, 1]
        correlations.append((feat, corr))
        print(f"  {feat:35s}: {corr:6.3f}")

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n[3] Top correlations (absolute value):")
for feat, corr in correlations[:5]:
    print(f"  {feat:35s}: {corr:6.3f}")

# Check for problematic patterns
print(f"\n[4] Data Quality Checks:")

# Check missing data
for feat in ol_cols:
    if feat in rb_df.columns:
        missing_pct = (rb_df[feat].isna().sum() / len(rb_df)) * 100
        zero_pct = ((rb_df[feat] == 0).sum() / len(rb_df)) * 100
        print(f"  {feat}:")
        print(f"    Missing: {missing_pct:.1f}%")
        print(f"    Zero: {zero_pct:.1f}%")
        if zero_pct > 50:
            print(f"    ⚠️  WARNING: >50% zeros - feature may not be informative")

# Check variance
print(f"\n[5] Feature Variance (low variance = less informative):")
for feat in all_new_features:
    if feat in rb_df.columns:
        var = rb_df[feat].var()
        std = rb_df[feat].std()
        print(f"  {feat:35s}: var={var:8.2f}, std={std:6.2f}")
        if var < 1.0:
            print(f"    ⚠️  WARNING: Low variance - feature may not be informative")

# Check feature ranges
print(f"\n[6] Feature Value Ranges:")
for feat in all_new_features:
    if feat in rb_df.columns:
        values = rb_df[feat].dropna()
        if len(values) > 0:
            print(f"  {feat:35s}: [{values.min():6.2f}, {values.max():6.2f}]")

# Recommendations
print(f"\n[7] Recommendations:")
low_corr_features = [f for f, c in correlations if abs(c) < 0.1]
if low_corr_features:
    print(f"  ⚠️  Features with low correlation (<0.1):")
    for feat in low_corr_features[:5]:
        print(f"    - {feat}")

high_zero_features = []
for feat in ol_cols:
    if feat in rb_df.columns:
        zero_pct = ((rb_df[feat] == 0).sum() / len(rb_df)) * 100
        if zero_pct > 50:
            high_zero_features.append(feat)

if high_zero_features:
    print(f"  ⚠️  Features with >50% zeros:")
    for feat in high_zero_features:
        print(f"    - {feat}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Remove features with low correlation or high zero rates")
print("2. Check if OL features are properly calculated (snap-weighted)")
print("3. Verify rolling stats handle edge cases correctly")
print("4. Consider feature selection or dimensionality reduction")
