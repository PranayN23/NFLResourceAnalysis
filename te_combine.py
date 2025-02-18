import pandas as pd

years = [2018, 2019, 2020, 2021, 2022]
pff = []
for year in years:
    tight_ends = pd.read_csv('PFF/Receiving' + str(year) + '.csv')
    positions_of_interest = ['TE']
    tight_ends = tight_ends[tight_ends['position'].isin(positions_of_interest)]

    # List of columns of interest
    columns = [
        "avg_depth_of_target", "avoided_tackles", "caught_percent",
        "contested_catch_rate", "contested_receptions", "contested_targets",
        "declined_penalties", "drop_rate", "drops", "first_downs",
        "franchise_id", "fumbles", "grades_hands_drop", "grades_hands_fumble",
        "grades_offense", "grades_pass_block", "grades_pass_route",
        "inline_rate", "inline_snaps", "interceptions", "longest",
        "pass_block_rate", "pass_blocks", "pass_plays", "penalties",
        "receptions", "route_rate", "routes", "slot_rate", "slot_snaps",
        "targeted_qb_rating", "targets", "touchdowns", "wide_rate",
        "wide_snaps", "yards", "yards_after_catch", "yards_after_catch_per_reception",
        "yards_per_reception", "yprr"
    ]

    for column in columns:
        if column in tight_ends.columns:
            tight_ends['weighted_' + column] = tight_ends[column] * tight_ends['slot_snaps']

    # Group by team_name and position, summing the weighted columns and total snaps
    aggregation_dict = {
        'slot_snaps': 'sum'  # Total snaps for normalization
    }

    for column in columns:
        if column in tight_ends.columns:
            aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

    te_grouped = tight_ends.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()

    # Calculate the weighted averages
    for column in columns:
        if 'weighted_' + column in te_grouped.columns:
            te_grouped['weighted_avg_' + column] = te_grouped['weighted_' + column] / te_grouped['slot_snaps']

    # Create the team mapping
    team_mapping = {
        'WAS': 'Commanders', 'TEN': 'Titans', 'TB': 'Buccaneers',
        'SF': '49ers', 'SEA': 'Seahawks', 'PIT': 'Steelers',
        'PHI': 'Eagles', 'NYJ': 'Jets', 'NYG': 'Giants',
        'NO': 'Saints', 'NE': 'Patriots', 'MIN': 'Vikings',
        'MIA': 'Dolphins', 'LV': 'Raiders', 'LAC': 'Chargers',
        'LA': 'Rams', 'KC': 'Chiefs', 'JAX': 'Jaguars',
        'IND': 'Colts', 'HST': 'Texans', 'GB': 'Packers',
        'DET': 'Lions', 'DEN': 'Broncos', 'DAL': 'Cowboys',
        'CLV': 'Browns', 'CIN': 'Bengals', 'CHI': 'Bears',
        'CAR': 'Panthers', 'ATL': 'Falcons', 'ARZ': 'Cardinals',
        'BLT': 'Ravens', 'BUF': 'Bills', 'OAK': 'Raiders'
    }

    te_grouped['Team'] = te_grouped['team_name'].replace(team_mapping)

    # Select relevant columns for output
    output_columns = ['position'] + ['weighted_avg_' + column for column in columns if
                                     'weighted_' + column in te_grouped.columns] + ['Team']
    te_grouped = te_grouped[output_columns]
    te_grouped['Year'] = year
    pff.append(te_grouped)
pd.concat(pff)

result = pd.concat(pff, ignore_index=True)
for column in columns:
    result['Previous_' + column] = result.groupby(['Team', 'position'])['weighted_avg_' + column].shift(1)
result.to_csv("TEPFF.csv")

