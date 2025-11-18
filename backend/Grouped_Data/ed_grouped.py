import pandas as pd

# Load CSV
df = pd.read_csv("backend/ML/ED.csv")

# Convert numeric-looking columns to proper float
numeric_cols = [
    'Cap_Space', 'adjusted_value', 'Net EPA', 'age', 'player_game_count',
    'assists', 'batted_passes', 'declined_penalties', 'forced_fumbles',
    'franchise_id', 'fumble_recoveries', 'grades_defense', 'grades_defense_penalty',
    'grades_pass_rush_defense', 'grades_run_defense', 'grades_tackle', 'hits',
    'hurries', 'missed_tackle_rate', 'missed_tackles', 'penalties', 'sacks',
    'snap_counts_defense', 'snap_counts_pass_rush', 'snap_counts_run_defense',
    'snap_counts_dl', 'snap_counts_dl_outside_t', 'snap_counts_dl_over_t',
    'stops', 'tackles', 'tackles_for_loss', 'total_pressures',
    'weighted_grade', 'weighted_average_grade'
]

# Force them to be numeric; convert errors to NaN
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Helper for weighted average
def weighted_avg(series, weights):
    # Filter out NaN values in both series and weights
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any() or weights[mask].sum() == 0:
        return 0.0  # Return 0 instead of NaN if no valid data
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()

pd.set_option('display.max_columns', None)
print(df[(df['Team'] == "Ravens") & (df['Year'] == 2022)][["sacks", "player", "snap_counts_defense"]])

# Group by Team and Year
grouped = (
    df.groupby(['Team', 'Year'])
    .apply(lambda g: pd.Series({
        'Cap_Space': g['Cap_Space'].sum() if g['Cap_Space'].notna().any() else 0.0,
        'adjusted_value': g['adjusted_value'].sum() if g['adjusted_value'].notna().any() else 0.0,
        'Net EPA': g['Net EPA'].iloc[0] if g['Net EPA'].notna().any() else 0.0,
        'Win %': g['Win %'].iloc[0] if g['Win %'].notna().any() else '',
        'franchise_id': g['franchise_id'].iloc[0] if g['franchise_id'].notna().any() else 0.0,
        
        # Weighted averages using snap_counts_defense as weight
        'age': weighted_avg(g['age'], g['snap_counts_defense']),
        'player_game_count': weighted_avg(g['player_game_count'], g['snap_counts_defense']),
        'assists': weighted_avg(g['assists'], g['snap_counts_defense']),
        'batted_passes': weighted_avg(g['batted_passes'], g['snap_counts_defense']),
        'declined_penalties': weighted_avg(g['declined_penalties'], g['snap_counts_defense']),
        'forced_fumbles': weighted_avg(g['forced_fumbles'], g['snap_counts_defense']),
        'fumble_recoveries': weighted_avg(g['fumble_recoveries'], g['snap_counts_defense']),
        'grades_defense': weighted_avg(g['grades_defense'], g['snap_counts_defense']),
        'grades_defense_penalty': weighted_avg(g['grades_defense_penalty'], g['snap_counts_defense']),
        'grades_pass_rush_defense': weighted_avg(g['grades_pass_rush_defense'], g['snap_counts_defense']),
        'grades_run_defense': weighted_avg(g['grades_run_defense'], g['snap_counts_defense']),
        'grades_tackle': weighted_avg(g['grades_tackle'], g['snap_counts_defense']),
        'hits': weighted_avg(g['hits'], g['snap_counts_defense']),
        'hurries': weighted_avg(g['hurries'], g['snap_counts_defense']),
        'missed_tackle_rate': weighted_avg(g['missed_tackle_rate'], g['snap_counts_defense']),
        'missed_tackles': weighted_avg(g['missed_tackles'], g['snap_counts_defense']),
        'penalties': weighted_avg(g['penalties'], g['snap_counts_defense']),
        'sacks': weighted_avg(g['sacks'], g['snap_counts_defense']),
        'snap_counts_defense': g['snap_counts_defense'].sum(),  # Sum total snaps
        'snap_counts_pass_rush': weighted_avg(g['snap_counts_pass_rush'], g['snap_counts_defense']),
        'snap_counts_run_defense': weighted_avg(g['snap_counts_run_defense'], g['snap_counts_defense']),
        'snap_counts_dl': weighted_avg(g['snap_counts_dl'], g['snap_counts_defense']),
        'snap_counts_dl_outside_t': weighted_avg(g['snap_counts_dl_outside_t'], g['snap_counts_defense']),
        'snap_counts_dl_over_t': weighted_avg(g['snap_counts_dl_over_t'], g['snap_counts_defense']),
        'stops': weighted_avg(g['stops'], g['snap_counts_defense']),
        'tackles': weighted_avg(g['tackles'], g['snap_counts_defense']),
        'tackles_for_loss': weighted_avg(g['tackles_for_loss'], g['snap_counts_defense']),
        'total_pressures': weighted_avg(g['total_pressures'], g['snap_counts_defense']),
        'weighted_grade': weighted_avg(g['weighted_grade'], g['snap_counts_defense']),
        'weighted_average_grade': weighted_avg(g['weighted_average_grade'], g['snap_counts_defense']),
    }))
    .reset_index()
)

print(grouped[(grouped['Team'] == "Ravens") & (grouped['Year'] == 2022)]["sacks"])

#print(grouped.head())
grouped.to_csv('backend/Grouped_Data/Grouped_ED.csv', index=False)

