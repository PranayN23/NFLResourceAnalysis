#!/usr/bin/env python
"""
Test script to validate RB model setup before running full training.
Checks data loading, OL feature integration, and rolling stats.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

print("=" * 60)
print("RB Model Setup Validation")
print("=" * 60)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import torch
    import joblib
    from sklearn.preprocessing import StandardScaler
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Data loading
print("\n[2/5] Testing data loading...")
try:
    data_path = 'backend/ML/HB.csv'
    if not os.path.exists(data_path):
        print(f"  ✗ Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df)} rows from HB.csv")
    print(f"  ✓ Columns: {len(df.columns)}")
except Exception as e:
    print(f"  ✗ Data loading error: {e}")
    sys.exit(1)

# Test 3: OL feature integration
print("\n[3/5] Testing OL feature integration...")
try:
    from backend.ML.RB_Pranay_Transformers.Player_Model_RB import add_ol_features
    
    # Test with small sample
    df_sample = df.head(100).copy()
    df_sample, ol_cols = add_ol_features(df_sample)
    
    if ol_cols:
        print(f"  ✓ OL features added: {ol_cols}")
        for col in ol_cols:
            if col in df_sample.columns:
                print(f"    - {col}: {df_sample[col].notna().sum()} non-null values")
    else:
        print("  ⚠ OL features not added (check G.csv, T.csv, C.csv)")
except Exception as e:
    print(f"  ✗ OL feature error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Rolling stats
print("\n[4/5] Testing rolling statistics...")
try:
    from backend.ML.RB_Pranay_Transformers.Player_Model_RB import add_rolling_stats
    
    df_sample = df.head(100).copy()
    df_sample = df_sample.sort_values(['player', 'Year'])
    df_sample, rolling_cols = add_rolling_stats(df_sample)
    
    if rolling_cols:
        print(f"  ✓ Rolling stats added: {rolling_cols}")
        for col in rolling_cols:
            if col in df_sample.columns:
                print(f"    - {col}: mean={df_sample[col].mean():.2f}")
except Exception as e:
    print(f"  ✗ Rolling stats error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Model architecture
print("\n[5/5] Testing model architecture...")
try:
    from backend.ML.RB_Pranay_Transformers.Player_Model_RB import PlayerTransformerRegressor
    
    # Test with expected feature count (31 features)
    model = PlayerTransformerRegressor(input_dim=31, seq_len=5)
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Input dim: 31, Seq len: 5")
    
    # Test forward pass
    import torch
    test_input = torch.randn(1, 5, 31)  # batch=1, seq=5, features=31
    test_mask = torch.zeros(1, 5, dtype=torch.bool)
    with torch.no_grad():
        output = model(test_input, mask=test_mask)
    print(f"  ✓ Forward pass successful, output shape: {output.shape}")
except Exception as e:
    print(f"  ✗ Model architecture error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Setup validation complete!")
print("=" * 60)
print("\nIf all tests passed, you can run:")
print("  python backend/ML/RB_Pranay_Transformers/Player_Model_RB.py")
print("\nIf you see errors, check:")
print("  1. Virtual environment is activated")
print("  2. All dependencies are installed")
print("  3. Data files (HB.csv, G.csv, T.csv, C.csv) exist")
