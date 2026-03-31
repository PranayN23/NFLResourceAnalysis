"""
Analyze grade distribution in training vs test data
"""
import pandas as pd
import numpy as np

# Load all OL data
df_g = pd.read_csv("backend/ML/G.csv")
df_c = pd.read_csv("backend/ML/C.csv")
df_t = pd.read_csv("backend/ML/T.csv")

df = pd.concat([df_g, df_c, df_t], axis=0, ignore_index=True)

print("="*100)
print("GRADE DISTRIBUTION ANALYSIS")
print("="*100)

# Overall stats
print("\nOVERALL grades_offense statistics:")
print(f"  Count: {df['grades_offense'].notna().sum()}")
print(f"  Mean: {df['grades_offense'].mean():.2f}")
print(f"  Median: {df['grades_offense'].median():.2f}")
print(f"  Std: {df['grades_offense'].std():.2f}")
print(f"  Min: {df['grades_offense'].min():.2f}")
print(f"  Max: {df['grades_offense'].max():.2f}")
percentiles = df['grades_offense'].quantile([0.25, 0.50, 0.75, 0.90, 0.95]).values
print(f"  Percentiles [25, 50, 75, 90, 95]: {percentiles}")

# Training vs Test split (as used in the script)
train_data = df[df["Year"] < 2024]
test_data = df[df["Year"] == 2024]

print(f"\n\nTRAINING DATA (Year < 2024):")
print(f"  Count: {train_data['grades_offense'].notna().sum()}")
print(f"  Mean: {train_data['grades_offense'].mean():.2f}")
print(f"  Std: {train_data['grades_offense'].std():.2f}")
print(f"  Min/Max: {train_data['grades_offense'].min():.2f} / {train_data['grades_offense'].max():.2f}")

print(f"\nTEST DATA (Year == 2024):")
print(f"  Count: {test_data['grades_offense'].notna().sum()}")
print(f"  Mean: {test_data['grades_offense'].mean():.2f}")
print(f"  Std: {test_data['grades_offense'].std():.2f}")
print(f"  Min/Max: {test_data['grades_offense'].min():.2f} / {test_data['grades_offense'].max():.2f}")

# By position
print(f"\n\nBY POSITION (All data):")
for pos in ['T', 'G', 'C']:
    df_pos = df[df['position'].str.upper() == pos]
    grades = df_pos['grades_offense'].dropna()
    print(f"  {pos}: Count={len(grades):4d}, Mean={grades.mean():6.2f}, Std={grades.std():5.2f}")

# Check specific elite players vs average
print(f"\n\nSPECIFIC PLAYERS - Grades Over Time:")
for player in ['Tristan Wirfs', 'Jordan Mailata', 'Penei Sewell']:
    p_data = df[(df['player'] == player) & (df['grades_offense'].notna())]
    if len(p_data) > 0:
        print(f"\n  {player}:")
        for _, row in p_data.sort_values('Year').iterrows():
            print(f"    {int(row['Year'])}: {row['grades_offense']:.1f}")

print("\n" + "="*100)
print("KEY FINDING")
print("="*100)
print("""
If the model is trained on grades with mean ~75 and std ~10,
but then predicts outputs in the 35-45 range for elite players,
this suggests the transformer model is COMPRESSING the output range,
not just the feature scaling.

This could be due to:
1. Sigmoid/tanh activation functions squashing output range
2. Model overfitting to mean prediction
3. Target variable distribution shift between train/test
4. Model not trained to distinguish between high-performing players
""")
