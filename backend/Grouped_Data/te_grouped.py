import pandas as pd
import numpy as np

df = pd.read_csv("backend/ML/TE.csv")

# ===============================================================
# 0. Replace literal "MISSING" strings with NaN
# ===============================================================

df = df.replace("MISSING", np.nan)

# ===============================================================
# 1. Numeric columns
# ===============================================================

numeric_cols = [
    'Cap_Space','adjusted_value','Net EPA','avg_depth_of_target','avoided_tackles',
    'caught_percent','contested_catch_rate','contested_receptions','contested_targets',
    'declined_penalties','drop_rate','drops','first_downs','fumbles','grades_hands_drop',
    'grades_hands_fumble','grades_offense','grades_pass_block','grades_pass_route',
    'inline_rate','inline_snaps','interceptions','longest','pass_block_rate','pass_blocks',
    'pass_plays','penalties','receptions','route_rate','routes','slot_rate','slot_snaps',
    'targeted_qb_rating','targets','touchdowns','wide_rate','wide_snaps','yards',
    'yards_after_catch','yards_after_catch_per_reception','yards_per_reception',
    'yprr','snap_counts_pass_block','snap_counts_run_block','total_snaps',
    'weighted_grade','weighted_average_grade'
]

# Convert numeric columns safely
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ===============================================================
# 2. Players with NaN total_snaps → treat as 0 snaps (but keep cap space)
# ===============================================================

df['total_snaps'] = df['total_snaps'].fillna(0)

# ===============================================================
# 3. Weighted average function (zero-snap players ignored automatically)
# ===============================================================

def weighted_avg(series, weights):
    weights = weights.fillna(0)              # NaN snaps → 0
    mask = (~series.isna()) & (weights > 0)  # only real players
    if mask.sum() == 0:
        return np.nan
    return np.average(series[mask], weights=weights[mask])

# ===============================================================
# 4. Group and aggregate
# ===============================================================

grouped = (
    df.groupby(['Team', 'Year'])
    .apply(lambda g: pd.Series({

        # Totals (include all players)
        'Cap_Space': g['Cap_Space'].sum(),
        'adjusted_value': g['adjusted_value'].sum(),
        'Net EPA': g['Net EPA'].iloc[0],
        'Win %': g['Win %'].iloc[0],
        'franchise_id': g['franchise_id'].iloc[0],

        # Weighted averages (snap-weighted, ignores 0-snap players)
        'avg_depth_of_target': weighted_avg(g['avg_depth_of_target'], g['total_snaps']),
        'avoided_tackles': weighted_avg(g['avoided_tackles'], g['total_snaps']),
        'caught_percent': weighted_avg(g['caught_percent'], g['total_snaps']),
        'contested_catch_rate': weighted_avg(g['contested_catch_rate'], g['total_snaps']),
        'drop_rate': weighted_avg(g['drop_rate'], g['total_snaps']),
        'grades_offense': weighted_avg(g['grades_offense'], g['total_snaps']),
        'grades_pass_block': weighted_avg(g['grades_pass_block'], g['total_snaps']),
        'grades_pass_route': weighted_avg(g['grades_pass_route'], g['total_snaps']),
        'inline_rate': weighted_avg(g['inline_rate'], g['total_snaps']),
        'pass_block_rate': weighted_avg(g['pass_block_rate'], g['total_snaps']),
        'route_rate': weighted_avg(g['route_rate'], g['total_snaps']),
        'slot_rate': weighted_avg(g['slot_rate'], g['total_snaps']),
        'targeted_qb_rating': weighted_avg(g['targeted_qb_rating'], g['total_snaps']),
        'wide_rate': weighted_avg(g['wide_rate'], g['total_snaps']),
        'yards_after_catch_per_reception': weighted_avg(g['yards_after_catch_per_reception'], g['total_snaps']),
        'yards_per_reception': weighted_avg(g['yards_per_reception'], g['total_snaps']),
        'yprr': weighted_avg(g['yprr'], g['total_snaps']),
        'weighted_grade': weighted_avg(g['weighted_grade'], g['total_snaps']),
        'weighted_average_grade': weighted_avg(g['weighted_average_grade'], g['total_snaps']),

        # Production (weighted)
        'touchdowns': weighted_avg(g['touchdowns'], g['total_snaps']),
        'yards': weighted_avg(g['yards'], g['total_snaps']),
        'receptions': weighted_avg(g['receptions'], g['total_snaps']),
        'first_downs': weighted_avg(g['first_downs'], g['total_snaps']),
        'targets': weighted_avg(g['targets'], g['total_snaps']),
    }))
    .reset_index()
)

# ===============================================================
# 5. Save result
# ===============================================================

grouped.to_csv('backend/Grouped_Data/Grouped_TE.csv', index=False)

print("Done! Grouped TE dataset written correctly.")
