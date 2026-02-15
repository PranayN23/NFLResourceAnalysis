
import pandas as pd
import numpy as np

def analyze_progression(filepath):
    print("Loading WR data...")
    df = pd.read_csv(filepath)
    
    # Clean numeric columns
    cols_to_numeric = ["weighted_grade", "Year", "career_year"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recalculate WR Weighted Grade (Grade * Snaps / 1000)
    if "grades_offense" in df.columns and "total_snaps" in df.columns:
         df["grades_offense"] = pd.to_numeric(df["grades_offense"], errors="coerce")
         df["total_snaps"] = pd.to_numeric(df["total_snaps"], errors="coerce")
         df["weighted_grade"] = df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0) / 1000.0

    # Sort
    df = df.sort_values(["player_id", "Year"])
    
    # Calculate Career Year 
    df["career_year_calc"] = df.groupby("player_id")["Year"].transform(lambda x: x - x.min() + 1)
    
    print("\n==== ANALYZING WR GROWTH FACTORS ====")
    
    growth_stats = []
    
    # Analyze transitions: Year 1->2, 2->3, 3->4
    for y_start, y_end in [(1, 2), (2, 3), (3, 4)]:
        t1 = df[df["career_year_calc"] == y_start][["player_id", "weighted_grade", "age"]]
        t2 = df[df["career_year_calc"] == y_end][["player_id", "weighted_grade"]]
        
        merged = pd.merge(t1, t2, on="player_id", suffixes=("_start", "_end"))
        
        # Filter for meaningful production (e.g. > 10 score ~ 200 snaps)
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

    print("\n==== RECOMMENDED CONSTANTS (Raw Retention) ====")
    for stat in growth_stats:
        print(f"GROWTH_{stat['Transition'].replace('->', '_')} = {stat['Retention']:.4f}")

if __name__ == "__main__":
    analyze_progression("backend/ML/WR.csv")
