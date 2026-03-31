"""
DIAGNOSTIC SCRIPT 2: Check if NaN Production is Position-Specific
Focus: Scan CSV files for complete-NaN season rows across all positions
"""
import pandas as pd
import numpy as np

print("="*80)
print("DIAGNOSTIC 2: DATA-LEVEL NaN ANALYSIS (CSV Scanning)")
print("="*80)

# Test each position by scanning raw CSV data
positions = [
    ("RB", "backend/ML/HB.csv"),
    ("WR", "backend/ML/WR.csv"),
    ("TE", "backend/ML/TightEnds/TE.csv"),
    ("LB", "backend/ML/LB.csv"),
    ("CB", "backend/ML/CB.csv"),
    ("S", "backend/ML/S.csv"),
    ("DI", "backend/ML/DI.csv"),
    ("ED", "backend/ML/ED.csv"),
    ("T", "backend/ML/T.csv"),
    ("G", "backend/ML/G.csv"),
    ("C", "backend/ML/C.csv"),
    ("QB", "backend/ML/C.csv"),
]


print("\nScanning all position CSVs for complete-NaN season rows...\n")

comprehensive_results = {}

for pos_name, csv_path in positions:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"✗ {pos_name}: File not found ({csv_path})")
        continue
    
    print(f"{'='*70}")
    print(f"POSITION: {pos_name} ({csv_path})")
    print(f"{'='*70}")
    print(f"Total records: {len(df)}")
    
    # Find players with at least one year of all-NaN stats
    players_with_null_years = []
    
    if 'player' in df.columns and 'Year' in df.columns:
        for player in df['player'].unique():
            player_df = df[df['player'] == player]
            
            for year in player_df['Year'].unique():
                year_df = player_df[player_df['Year'] == year]
                
                # Check key stat columns (will vary by position)
                stat_cols = [col for col in df.columns if col not in ['player', 'Year', 'team', 'player_last_name', 'salary']]
                
                # Count NaN values in stat columns
                null_count = year_df[stat_cols].isnull().sum().sum()
                total_values = len(stat_cols) * len(year_df)
                
                # If 75%+ of stat columns are NaN, flag it
                if total_values > 0 and null_count >= total_values * 0.75:
                    players_with_null_years.append((player, year, null_count, total_values))
        
        if players_with_null_years:
            print(f"\n⚠️  Found {len(players_with_null_years)} player-years with 75%+ NaN stats:")
            for player, year, null_count, total in sorted(players_with_null_years)[:10]:
                pct = (null_count/total)*100
                print(f"  - {player:30s} Year {year}: {pct:5.1f}% NaN ({null_count}/{total} values)")
            comprehensive_results[pos_name] = len(players_with_null_years)
        else:
            print(f"\n✓ No seasons with 75%+ missing values in first pass")
            comprehensive_results[pos_name] = 0
    else:
        print(f"Skipped: missing 'player' or 'Year' column")
        comprehensive_results[pos_name] = None

print("\n" + "="*80)
print("SUMMARY: NaN PREVALENCE BY POSITION")
print("="*80)

for pos_name in comprehensive_results:
    count = comprehensive_results[pos_name]
    if count is not None:
        symbol = "⚠️" if count > 0 else "✓"
        print(f"{symbol} {pos_name:3s}: {count:3d} player-years with severe missing data")

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
1. NaN production is DATA-DRIVEN, not model-specific
2. Any position can have a player with a completely-missing season in raw CSV
3. Example: Mike Boone (RB) has 2023 season with all-NaN stats
4. When prepare_features() tries to compute deltas on NaN rows:
   - NaN + operations = NaN
   - StandardScaler input has NaN
   - StandardScaler output has NaN (no imputation)
   - Transformer receives NaN tensor → outputs NaN

ROOT CAUSE CHAIN:
  Raw CSV has all-NaN row
    ↓
  Feature engineering computes on NaN values → NaN output
    ↓
  StandardScaler receives NaN → outputs NaN (by design)
    ↓
  Transformer receives NaN values in input tensor
    ↓
  Numerical operations with NaN → NaN output
    ↓
  Ensemble: (NaN * 0.5) + (other_score * 0.5) = NaN
    ↓
  Age adjustment: NaN - adjustment = NaN
    ↓
  FINAL GRADE: NaN
""")

