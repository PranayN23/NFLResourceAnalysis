import pandas as pd

# Load Tackle data
df = pd.read_csv("backend/ML/T.csv")

# Columns we don't want to coerce to numeric
non_numeric_cols = ['player_id', 'Team', 'Year', 'position_x', 'player', 'position', 'franchise_id']
numeric_cols = [c for c in df.columns if c not in non_numeric_cols]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

weight_col = 'snap_counts_offense'


def weighted_avg(series: pd.Series, weights: pd.Series) -> float:
    weights = weights.fillna(0)
    if weights.sum() == 0:
        return series.mean()
    return (series * weights).sum() / weights.sum()


# Columns to sum across players
sum_cols = [
    'Cap_Space',
    'adjusted_value',
    'declined_penalties',
    'hits_allowed',
    'hurries_allowed',
    'non_spike_pass_block',
    'penalties',
    'pressures_allowed',
    'sacks_allowed',
    'snap_counts_block',
    'snap_counts_ce',
    'snap_counts_lg',
    'snap_counts_lt',
    'snap_counts_offense',
    'snap_counts_pass_block',
    'snap_counts_pass_play',
    'snap_counts_rg',
    'snap_counts_rt',
    'snap_counts_run_block',
    'snap_counts_te',
    'weighted_grade',
]

# Everything else that's numeric (and not summed or delegated) gets a snap-weighted average
weighted_cols = [
    col
    for col in numeric_cols
    if col not in sum_cols + ['Net EPA', 'Win %', weight_col]
]


def aggregate_group(g: pd.DataFrame) -> pd.Series:
    weights = g[weight_col].fillna(0)
    aggregated = {
        'Cap_Space': g['Cap_Space'].sum(),
        'adjusted_value': g['adjusted_value'].sum(),
        'Net EPA': g['Net EPA'].iloc[0],
        'Win %': g['Win %'].iloc[0],
        'franchise_id': g['franchise_id'].iloc[0],
    }

    for col in sum_cols:
        if col in aggregated:
            continue
        aggregated[col] = g[col].sum()

    for col in weighted_cols:
        aggregated[col] = weighted_avg(g[col], weights)

    return pd.Series(aggregated)


grouped = (
    df.groupby(['Team', 'Year'])
    .apply(aggregate_group, include_groups=False)
    .reset_index()
)

# Quick sanity check for a known team/year if present
sample_team = 'Chiefs'
sample_year = 2022
print(grouped[(grouped['Team'] == sample_team) & (grouped['Year'] == sample_year)].head())

grouped.to_csv('backend/Grouped_Data/Grouped_T.csv', index=False)
