
import pandas as pd
import numpy as np


# Load Data
df = pd.read_csv("backend/ML/TightEnds/TE.csv")
df = df.replace("MISSING", np.nan)

# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Remove rows with no base-grade
df = df.dropna(subset=["weighted_grade"])
df = df.sort_values(["player_id", "Year"]).reset_index(drop=True)

# ==========================================
# DEFINE TARGETS
# ==========================================
# 1. TE_Value_Score (Current Best): (Grade - 50) * Snaps
if "TE_Value_Score" not in df.columns:
    df["TE_Value_Score"] = (df["grades_offense"].fillna(50) - 50) * df["total_snaps"].fillna(0) / 400

# 2. Weighted Grade (New experiment): Grade * Snaps
df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000

# Choose Target to Analyze
target_col = "weighted_grade" 
print(f"Analyzing Target: {target_col}")

# Shift Target to Next Year
df[f"{target_col}_next"] = df.groupby("player_id")[target_col].shift(-1)
df_trainable = df.dropna(subset=[f"{target_col}_next"]).reset_index(drop=True)

# Select numeric columns
numeric_df = df_trainable.select_dtypes(include=[np.number])

# Calculate Correlation with Target
correlations = numeric_df.corr()[f"{target_col}_next"].sort_values(ascending=False)

print(f"Top 20 Positive Correlations with {target_col}_next:")
print(correlations.head(20))
print(f"\nTop 20 Negative Correlations with {target_col}_next:")
print(correlations.tail(20))

# Also allow checking which features strongly correlate with each other (Multicollinearity)
print("\nGenerating Correlation Matrix for top features...")
# We can just pick the top 15 features and see their heat map values (printed)
top_features = correlations.index[1:16].tolist() # 0 is target itself
print(f"Top Features: {top_features}")

subset_corr = numeric_df[top_features].corr()
print("\nCorrelation between top predictors:")
print(subset_corr.to_string())
