import pandas as pd

# Load CSV
df = pd.read_csv("backend/ML/QB.csv")
# Convert numeric-looking columns to proper float
numeric_cols = [
    'Cap_Space','adjusted_value','Net EPA','accuracy_percent','aimed_passes',
    'attempts','avg_depth_of_target','avg_time_to_throw','bats','big_time_throws',
    'btt_rate','completion_percent','completions','declined_penalties',
    'def_gen_pressures','drop_rate','dropbacks','drops','first_downs',
    'grades_hands_fumble','grades_offense','grades_pass','grades_run',
    'hit_as_threw','interceptions','passing_snaps','penalties',
    'pressure_to_sack_rate','qb_rating','sack_percent','sacks','scrambles',
    'spikes','thrown_aways','touchdowns','turnover_worthy_plays',
    'twp_rate','yards','ypa'
]

# Force them to be numeric; convert errors to NaN, then fill NaN with 0 for aggregation
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Helper for weighted average
def weighted_avg(series, weights):
    if weights.sum() == 0:
        return series.mean()
    return (series * weights).sum() / weights.sum()

pd.set_option('display.max_columns', None)
print(df[(df['Team'] == "Ravens") & (df['Year'] == 2022)][["touchdowns", "player", "passing_snaps"]])

# Group by position, player, and year
grouped = (
    df.groupby(['Team', 'Year'])
    .apply(lambda g: pd.Series({
        'Cap_Space': g['Cap_Space'].sum(),
        'adjusted_value': g['adjusted_value'].sum(),
        'Net EPA': g['Net EPA'].iloc[0],
        'Win %': g['Win %'].iloc[0],
        'franchise_id': g['franchise_id'].iloc[0],
        
        # Counting statistics - SUM
        'aimed_passes': g['aimed_passes'].sum(),
        'attempts': g['attempts'].sum(),
        'completions': g['completions'].sum(),
        'touchdowns': g['touchdowns'].sum(),
        'interceptions': g['interceptions'].sum(),
        'sacks': g['sacks'].sum(),
        'scrambles': g['scrambles'].sum(),
        'first_downs': g['first_downs'].sum(),
        'yards': g['yards'].sum(),
        'dropbacks': g['dropbacks'].sum(),
        'drops': g['drops'].sum(),
        'big_time_throws': g['big_time_throws'].sum(),
        'turnover_worthy_plays': g['turnover_worthy_plays'].sum(),
        'bats': g['bats'].sum(),
        'declined_penalties': g['declined_penalties'].sum(),
        'penalties': g['penalties'].sum(),
        'hit_as_threw': g['hit_as_threw'].sum(),
        'thrown_aways': g['thrown_aways'].sum(),
        'spikes': g['spikes'].sum(),
        
        # Rate/percentage statistics - WEIGHTED AVERAGE
        'accuracy_percent': weighted_avg(g['accuracy_percent'], g['passing_snaps']),
        'completion_percent': weighted_avg(g['completion_percent'], g['passing_snaps']),
        'btt_rate': weighted_avg(g['btt_rate'], g['passing_snaps']),
        'twp_rate': weighted_avg(g['twp_rate'], g['passing_snaps']),
        'drop_rate': weighted_avg(g['drop_rate'], g['passing_snaps']),
        'qb_rating': weighted_avg(g['qb_rating'], g['passing_snaps']),
        'sack_percent': weighted_avg(g['sack_percent'], g['passing_snaps']),
        'pressure_to_sack_rate': weighted_avg(g['pressure_to_sack_rate'], g['passing_snaps']),
        'avg_depth_of_target': weighted_avg(g['avg_depth_of_target'], g['passing_snaps']),
        'avg_time_to_throw': weighted_avg(g['avg_time_to_throw'], g['passing_snaps']),
        'ypa': weighted_avg(g['ypa'], g['passing_snaps']),
        'def_gen_pressures': weighted_avg(g['def_gen_pressures'], g['passing_snaps']),
        
        # Grades - WEIGHTED AVERAGE
        'grades_hands_fumble': weighted_avg(g['grades_hands_fumble'], g['passing_snaps']),
        'grades_offense': weighted_avg(g['grades_offense'], g['passing_snaps']),
        'grades_pass': weighted_avg(g['grades_pass'], g['passing_snaps']),
        'grades_run': weighted_avg(g['grades_run'], g['passing_snaps']),
    }), include_groups=False)
    .reset_index()
)
print("\n")
print(grouped[(grouped['Team'] == "Ravens") & (grouped['Year'] == 2022)]["touchdowns"])

grouped.to_csv('backend/Grouped_Data/Grouped_QB.csv')