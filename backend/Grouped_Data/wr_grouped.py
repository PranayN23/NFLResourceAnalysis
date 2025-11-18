import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\23tan\OneDrive - purdue.edu\Sophomore_Fall_2024_Semester\nfl-resource-allocation\NFLResourceAnalysis\backend\ML\WR.csv")
print(df[(df["Team"] == "Bengals") & (df["Year"] == 2022)][["touchdowns", "total_snaps", "player"]])

# Grouping and key columns
group_cols = ["Year", "Team"]
sum_cols = ["Cap_Space", "adjusted_value"]
keep_cols = ["Win %", "Net EPA"]
weight_col = "total_snaps"

# Convert numeric columns safely (keep_cols stay as-is)
for c in df.columns:
    if c not in group_cols + keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def weighted_avg(x: pd.Series):
    """
    x is the Series for a single column within a group.
    We fetch the corresponding weights from the original df via x.index.
    """
    w = df.loc[x.index, weight_col]
    mask = x.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return np.average(x[mask], weights=w[mask])

# Build aggregation dict:
# - default: weighted_avg for every column except group keys, sums, keeps, and the weight column itself
agg_dict = {
    c: (weighted_avg if c not in group_cols + sum_cols + keep_cols + [weight_col] else 'sum')
    for c in df.columns if c not in group_cols
}


# Keep columns: first non-null
for c in keep_cols:
    agg_dict[c] = "first"

# Sum columns: sum
for c in sum_cols:
    agg_dict[c] = "sum"

# (Optional) also sum the weight column so you can see total snaps per team-year
if weight_col in df.columns:
    agg_dict[weight_col] = "sum"

# Group and aggregate
result = df.groupby(group_cols).agg(agg_dict).reset_index()

# Save
#result.to_csv("Grouped_WR.csv", index=False)
#print("✅ Aggregated data saved to Grouped_WR.csv")
print(result[(result["Team"] == "Bengals") & (result["Year"] == 2022)][["touchdowns", "total_snaps", "player"]])
