
import pandas as pd
import numpy as np

def analyze_progression(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Clean numeric columns
    cols_to_numeric = ["weighted_grade", "Year", "career_year"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If weighted_grade is missing (it might be calculated in the other script), recalculate rough version
    if "weighted_grade" not in df.columns or df["weighted_grade"].isna().all():
         df["grades_offense"] = pd.to_numeric(df["grades_offense"], errors="coerce")
         df["total_snaps"] = pd.to_numeric(df["total_snaps"], errors="coerce")
         df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000.0

    # Sort
    df = df.sort_values(["player_id", "Year"])
    
    # Calculate Career Year if missing (Assuming min year for player is year 0 or 1)
    # Note: dataset may not have rookie year for everyone. 
    # We will assume the earliest year present is year 1 IF age < 24? Or just trust the diff.
    # Actually Time2Vec script calculates career_year. Let's replicate simple version.
    df["career_year_calc"] = df.groupby("player_id")["Year"].transform(lambda x: x - x.min() + 1)
    
    print("\n==== ANALYZING GROWTH FACTORS ====")
    
    growth_stats = []
    
    for y_start, y_end in [(1, 2), (2, 3), (3, 4)]:
        # Get pairs
        t1 = df[df["career_year_calc"] == y_start][["player_id", "weighted_grade", "age"]]
        t2 = df[df["career_year_calc"] == y_end][["player_id", "weighted_grade"]]
        
        merged = pd.merge(t1, t2, on="player_id", suffixes=("_start", "_end"))
        
        # Filter for meaningful production (ignore benchwarmers who stay 0)
        # Assuming "High Potential" -> Grade > 10 in previous year
        meaningful = merged[merged["weighted_grade_start"] > 10]
        
        avg_growth_abs = (meaningful["weighted_grade_end"] - meaningful["weighted_grade_start"]).mean()
        avg_growth_pct = (meaningful["weighted_grade_end"] / meaningful["weighted_grade_start"]).mean()
        retention = meaningful["weighted_grade_end"].mean() / meaningful["weighted_grade_start"].mean()
        
        print(f"\nTransition Year {y_start} -> {y_end}:")
        print(f"  Sample Size: {len(meaningful)}")
        print(f"  Avg Absolute Change: {avg_growth_abs:.4f}")
        print(f"  Avg Growth Multiplier (Mean of Ratios): {avg_growth_pct:.4f}")
        print(f"  Retention Rate (Ratio of Means): {retention:.4f}")
        
        growth_stats.append({
            "Transition": f"{y_start}->{y_end}",
            "Retention": retention
        })

    print("\n==== RECCOMENDED CONSTANTS ====")
    for stat in growth_stats:
        print(f"GROWTH_{stat['Transition'].replace('->', '_')} = {stat['Retention']:.4f}")

    # Team History Analysis
    print("\n==== TEAM TE EPA HISTORY ====")
    # Calculate avg Net EPA for each team per year
    df["Net EPA"] = pd.to_numeric(df["Net EPA"], errors="coerce")
    team_epa = df.groupby(["Team", "Year"])["Net EPA"].mean().reset_index()
    print("Team EPA Head:")
    print(team_epa.head())

if __name__ == "__main__":
    analyze_progression("backend/ML/TightEnds/TE.csv")
