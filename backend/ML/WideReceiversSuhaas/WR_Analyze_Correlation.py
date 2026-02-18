
import pandas as pd
import numpy as np

DATA_FILE = "backend/ML/WR.csv"

def analyze_correlation():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # 1. Normalize Weighted Grade (if needed)
    # Check current scale
    mean_wg = df["weighted_grade"].dropna().mean()
    print(f"Mean Raw Weighted Grade: {mean_wg:.2f}")
    
    # Re-calculate to ensure consistency with TE model (Grade * Snaps / 1000)
    # This keeps values in a readable ~30-80 range instead of ~30,000
    df["weighted_grade_norm"] = (df["grades_offense"].fillna(0) * df["total_snaps"].fillna(0)) / 1000.0
    
    # Filter for significant seasons (e.g., > 100 snaps) to reduce noise from bench players? 
    # For correlation, we generally want all valid data, but let's see.
    # We'll stick to the raw year-to-year correlation first.
    
    df = df.sort_values(["player_id", "Year"])
    
    # Create Lag Features
    df["prev_grade"] = df.groupby("player_id")["weighted_grade_norm"].shift(1)
    df["prev_grade_2"] = df.groupby("player_id")["weighted_grade_norm"].shift(2)
    
    # 1-Year Correlation
    valid_1y = df.dropna(subset=["weighted_grade_norm", "prev_grade"])
    corr_1y = valid_1y["weighted_grade_norm"].corr(valid_1y["prev_grade"])
    
    print(f"\n==== CORRELATION ANALYSIS ====")
    print(f"Year-to-Year Correlation (N={len(valid_1y)}): {corr_1y:.4f}")
    
    # 2-Year Correlation
    valid_2y = df.dropna(subset=["weighted_grade_norm", "prev_grade_2"])
    corr_2y = valid_2y["weighted_grade_norm"].corr(valid_2y["prev_grade_2"])
    print(f"2-Year Lag Correlation (N={len(valid_2y)}):   {corr_2y:.4f}")
    
    # Check Consistency by Year
    print("\n==== YEAR-BY-YEAR STABILITY ====")
    years = sorted(df["Year"].unique())
    print(f"{'Year':<6} | {'N':<6} | {'Corr (Yr-1)':<12}")
    print("-" * 30)
    
    for year in years:
        if year == years[0]: continue
        year_data = df[df["Year"] == year]
        # Need to join with prev year manually or use the shifted column
        # The shifted column is already aligned by player since we sorted and grouped
        
        # Filter for this year
        curr = df[df["Year"] == year]
        
        if len(curr) > 10:
            c = curr["weighted_grade_norm"].corr(curr["prev_grade"])
            print(f"{year:<6} | {len(curr.dropna(subset=['prev_grade'])):<6} | {c:.4f}")

if __name__ == "__main__":
    analyze_correlation()
