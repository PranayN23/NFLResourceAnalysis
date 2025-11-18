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



# Force them to be numeric; convert errors to NaN
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')


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
        
        # Weighted averages
        'accuracy_percent': weighted_avg(g['accuracy_percent'], g['passing_snaps']),
        'aimed_passes': weighted_avg(g['aimed_passes'], g['passing_snaps']),
        'attempts': weighted_avg(g['attempts'], g['passing_snaps']),
        'avg_depth_of_target': weighted_avg(g['avg_depth_of_target'], g['passing_snaps']),
        'avg_time_to_throw': weighted_avg(g['avg_time_to_throw'], g['passing_snaps']),
        'bats': weighted_avg(g['bats'], g['passing_snaps']),
        'big_time_throws': weighted_avg(g['big_time_throws'], g['passing_snaps']),
        'btt_rate': weighted_avg(g['btt_rate'], g['passing_snaps']),
        'completion_percent': weighted_avg(g['completion_percent'], g['passing_snaps']),
        'completions': weighted_avg(g['completions'], g['passing_snaps']),
        'declined_penalties': weighted_avg(g['declined_penalties'], g['passing_snaps']),
        'def_gen_pressures': weighted_avg(g['def_gen_pressures'], g['passing_snaps']),
        'drop_rate': weighted_avg(g['drop_rate'], g['passing_snaps']),
        'dropbacks': weighted_avg(g['dropbacks'], g['passing_snaps']),
        'drops': weighted_avg(g['drops'], g['passing_snaps']),
        'first_downs': weighted_avg(g['first_downs'], g['passing_snaps']),
        'grades_hands_fumble': weighted_avg(g['grades_hands_fumble'], g['passing_snaps']),
        'grades_offense': weighted_avg(g['grades_offense'], g['passing_snaps']),
        'grades_pass': weighted_avg(g['grades_pass'], g['passing_snaps']),
        'grades_run': weighted_avg(g['grades_run'], g['passing_snaps']),
        'hit_as_threw': weighted_avg(g['hit_as_threw'], g['passing_snaps']),
        'interceptions': weighted_avg(g['interceptions'], g['passing_snaps']),
        'penalties': weighted_avg(g['penalties'], g['passing_snaps']),
        'pressure_to_sack_rate': weighted_avg(g['pressure_to_sack_rate'], g['passing_snaps']),
        'qb_rating': weighted_avg(g['qb_rating'], g['passing_snaps']),
        'sack_percent': weighted_avg(g['sack_percent'], g['passing_snaps']),
        'sacks': weighted_avg(g['sacks'], g['passing_snaps']),
        'scrambles': weighted_avg(g['scrambles'], g['passing_snaps']),
        'spikes': weighted_avg(g['spikes'], g['passing_snaps']),
        'thrown_aways': weighted_avg(g['thrown_aways'], g['passing_snaps']),
        'touchdowns': weighted_avg(g['touchdowns'], g['passing_snaps']),
        'turnover_worthy_plays': weighted_avg(g['turnover_worthy_plays'], g['passing_snaps']),
        'twp_rate': weighted_avg(g['twp_rate'], g['passing_snaps']),
        'yards': weighted_avg(g['yards'], g['passing_snaps']),
        'ypa': weighted_avg(g['ypa'], g['passing_snaps']),
    }))
    .reset_index()
)
print(grouped[(grouped['Team'] == "Ravens") & (grouped['Year'] == 2022)]["touchdowns"])

#print(grouped.head())
grouped.to_csv('backend/Grouped_Data/Grouped_QB.csv')
