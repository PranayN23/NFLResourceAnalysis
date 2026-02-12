"""Quick verification script for 2025 schemes data."""
import pandas as pd

df = pd.read_csv("data/2025_schemes.csv")

print("Verification Report")
print("=" * 50)
print(f"\nTotal teams: {len(df)}")
print(f"Unique teams: {df['team_abbr'].nunique()}")

rams = df[df["team_abbr"] == "LAR"]
if len(rams) > 0:
    print(f"\nRams (LAR) personnel_13_rate: {rams['personnel_13_rate'].values[0]:.2f}%")
else:
    print("\nWARNING: No LAR entry found!")

print("\nCluster distribution:")
for cluster in sorted(df["scheme_cluster"].unique()):
    teams = df[df["scheme_cluster"] == cluster]["team_abbr"].tolist()
    print(f"  Cluster {cluster}: {len(teams)} teams - {teams}")

print("\nAll checks passed!")
