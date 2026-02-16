#!/usr/bin/env python
"""
Debug script to check OL feature quality and data issues
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.ML.RB_Pranay_Transformers.Player_Model_RB import add_ol_features, add_rolling_stats

# Load RB data
rb_df = pd.read_csv('backend/ML/HB.csv')
rb_df = rb_df[rb_df['total_touches'] >= 50].copy()
rb_df.sort_values(by=['player', 'Year'], inplace=True)

print("="*80)
print("DEBUGGING RB MODEL FEATURES")
print("="*80)

# Check OL features
print("\n[1] Checking OL Feature Integration...")
rb_df_test, ol_cols = add_ol_features(rb_df.copy())

print(f"  OL features added: {ol_cols}")
print(f"  Total rows: {len(rb_df_test)}")

# Check missing values
for col in ol_cols:
    missing = rb_df_test[col].isna().sum()
    pct_missing = (missing / len(rb_df_test)) * 100
    print(f"  {col}: {missing} missing ({pct_missing:.1f}%)")
    
    if missing > 0:
        # Check if missing values are filled
        filled = rb_df_test[col].notna().sum()
        print(f"    After fill: {filled} non-null")
        
        # Show sample values
        sample = rb_df_test[col].dropna().head(5)
        print(f"    Sample values: {sample.values}")

# Check rolling stats
print("\n[2] Checking Rolling Statistics...")
rb_df_test = add_rolling_stats(rb_df_test)
rolling_cols = ['rb_yards_rolling_std', 'rb_touches_rolling_std', 'rb_grades_rolling_std',
                'rb_yards_rolling_mean', 'rb_grades_rolling_mean']

for col in rolling_cols:
    if col in rb_df_test.columns:
        missing = rb_df_test[col].isna().sum()
        mean_val = rb_df_test[col].mean()
        std_val = rb_df_test[col].std()
        print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}, missing={missing}")

# Check feature distributions
print("\n[3] Feature Value Distributions...")
all_features = ol_cols + rolling_cols
for feat in all_features:
    if feat in rb_df_test.columns:
        values = rb_df_test[feat].dropna()
        if len(values) > 0:
            print(f"  {feat}:")
            print(f"    Min: {values.min():.2f}, Max: {values.max():.2f}")
            print(f"    Mean: {values.mean():.2f}, Median: {values.median():.2f}")
            print(f"    Zero count: {(values == 0).sum()} ({(values == 0).sum()/len(values)*100:.1f}%)")

# Check correlation with target
print("\n[4] Feature Correlations with grades_offense...")
if 'grades_offense' in rb_df_test.columns:
    for feat in all_features:
        if feat in rb_df_test.columns:
            corr = rb_df_test[[feat, 'grades_offense']].corr().iloc[0, 1]
            print(f"  {feat}: {corr:.3f}")

# Check specific problematic cases
print("\n[5] Checking Specific Cases...")
# Find players with high actual grades but low predictions
test_year = 2024
test_data = rb_df_test[rb_df_test['Year'] == test_year].copy()
if len(test_data) > 0:
    high_performers = test_data.nlargest(5, 'grades_offense')
    print(f"\n  Top 5 performers in {test_year}:")
    for _, row in high_performers.iterrows():
        print(f"    {row['player']}: Grade={row['grades_offense']:.1f}, Team={row.get('Team', 'N/A')}")
        for ol_col in ol_cols:
            if ol_col in row:
                print(f"      {ol_col}: {row[ol_col]:.2f}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
