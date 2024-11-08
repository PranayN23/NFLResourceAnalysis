import pandas as pd

def get_pff():
    years = [2018, 2019, 2020, 2021, 2022]
    pff = []
    for year in years:
        OL = pd.read_csv('PFF/OL' + str(year) + '.csv')
        positions_of_interest = ['C', 'G', 'T', 'TE']
        OL = OL[OL['position'].isin(positions_of_interest)]

        # List of columns of interest
        columns = [
            "block_percent", "declined_penalties", "franchise_id", "grades_offense",
            "grades_pass_block", "grades_run_block", "hits_allowed", "hurries_allowed",
            "non_spike_pass_block", "non_spike_pass_block_percentage", "pass_block_percent",
            "pbe", "penalties", "pressures_allowed", "sacks_allowed", "snap_counts_block",
            "snap_counts_ce", "snap_counts_lg", "snap_counts_lt", "snap_counts_offense",
            "snap_counts_pass_block", "snap_counts_pass_play", "snap_counts_rg",
            "snap_counts_rt", "snap_counts_run_block", "snap_counts_te"
        ]
        # Initialize weighted columns
        for column in columns:
            if column in OL.columns:
                OL['weighted_' + column] = OL[column] * OL['block_percent']

        # Group by team_name and position, summing the weighted columns and total snaps
        aggregation_dict = {
            'block_percent': 'sum'  # Total snaps for normalization
        }

        for column in columns:
            if column in OL.columns:
                aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

        OL_grouped = OL.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()

        # Calculate the weighted averages
        for column in columns:
            if 'weighted_' + column in OL_grouped.columns:
                OL_grouped['weighted_avg_' + column] = OL_grouped['weighted_' + column] / OL_grouped['block_percent']

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

        OL_grouped['Team'] = OL_grouped['team_name'].replace(team_mapping)

        # Select relevant columns for output
        output_columns = ['position'] + ['weighted_avg_' + column for column in columns if
                                         'weighted_' + column in OL_grouped.columns] + ['Team']
        OL_grouped = OL_grouped[output_columns]
        OL_grouped['Year'] = year
        pff.append(OL_grouped)
    pd.concat(pff)

    result = pd.concat(pff, ignore_index=True)
    for column in columns:
        result['Previous_' + column] = result.groupby(['Team', 'position'])['weighted_avg_' + column].shift(1)
    file_path = "/Users/vihaanchadha/downloads/OLPFF.csv"
    result.to_csv(file_path)
    result.to_csv("OLPFF.csv")


get_pff()