
import pandas as pd
import numpy as np

def add_value_metric(filepath):
    df = pd.read_csv(filepath)
    
    # Ensure numeric
    df["weighted_grade"] = pd.to_numeric(df["weighted_grade"], errors="coerce")
    df["grades_offense"] = pd.to_numeric(df["grades_offense"], errors="coerce")
    df["total_snaps"] = pd.to_numeric(df["total_snaps"], errors="coerce")
    
    # 1. Total Value Metric
    # Formula: (Grade - Replacement_Level) * Snaps
    # Replacement level set conservatively at 50 (PFF grades usually center around 60, 50-60 is average/backup)
    
    replacement_level = 50
    # Use 'grades_offense' (PFF Grade 0-100) instead of 'weighted_grade' (which appears to be pre-scaled).
    df["TE_Value"] = (df["grades_offense"].fillna(replacement_level) - replacement_level) * df["total_snaps"].fillna(0)
    
    # Scale to 0-100ish for readability
    # Max possible: (92-50) * 1100 = 46,200
    # Div by 400 => ~115 max.
    df["TE_Value_Score"] = df["TE_Value"] / 400
    
    # 2. Production Only Score (Fantasy-like) for reference
    # Yards + 20*TDs + 10*FirstDowns
    df["yards"] = pd.to_numeric(df["yards"], errors="coerce").fillna(0)
    df["touchdowns"] = pd.to_numeric(df["touchdowns"], errors="coerce").fillna(0)
    df["first_downs"] = pd.to_numeric(df["first_downs"], errors="coerce").fillna(0)
    
    df["Production_Score"] = df["yards"] + (20 * df["touchdowns"]) + (10 * df["first_downs"])
    
    # Save back
    df.to_csv(filepath, index=False)
    print(f"Updated {filepath} with 'TE_Value' and 'TE_Value_Score'.")
    
    # Print Top 20 All-Time
    print("\nTop 20 Seasons (TE_Value_Score):")
    print(df[["player", "Year", "Team", "TE_Value_Score", "grades_offense", "total_snaps"]].sort_values("TE_Value_Score", ascending=False).head(20))
    
    # Print Top 20 from 2024
    print("\nTop 20 Seasons (2024):")
    print(df[df["Year"] == 2024][["player", "Year", "Team", "TE_Value_Score", "grades_offense", "total_snaps"]].sort_values("TE_Value_Score", ascending=False).head(20))

if __name__ == "__main__":
    add_value_metric("backend/ML/TightEnds/TE.csv")
