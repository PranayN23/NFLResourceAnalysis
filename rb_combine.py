import pandas as pd

def get_pff():
    years = [2018, 2019, 2020, 2021, 2022]
    pff = []
    for year in years:
        rb = pd.read_csv('PFF/RB' + str(year) + '.csv')
        positions_of_interest = ['HB']
        rb = rb[rb['position'].isin(positions_of_interest)]

        # List of columns of interest
        columns = [
            'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent',
            'breakaway_yards', 'declined_penalties', 'designed_yards', 'drops',
            'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive',
            'first_downs', 'franchise_id', 'fumbles', 'gap_attempts', 'grades_hands_fumble',
            'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block',
            'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties',
            'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles',
            'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact',
            'yco_attempt', 'ypa', 'yprr', 'zone_attempts'
        ]
        # Initialize weighted columns
        for column in columns:
            if column in rb.columns:
                rb['weighted_' + column] = rb[column] * rb['attempts']

        # Group by team_name and position, summing the weighted columns and total snaps
        aggregation_dict = {
            'attempts': 'sum'  # Total snaps for normalization
        }

        for column in columns:
            if column in rb.columns:
                aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

        rb_grouped = rb.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()

        # Calculate the weighted averages
        for column in columns:
            if 'weighted_' + column in rb_grouped.columns:
                rb_grouped['weighted_avg_' + column] = rb_grouped['weighted_' + column] / rb_grouped['attempts']

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

        rb_grouped['Team'] = rb_grouped['team_name'].replace(team_mapping)

        # Select relevant columns for output
        output_columns = ['position'] + ['weighted_avg_' + column for column in columns if
                                         'weighted_' + column in rb_grouped.columns] + ['Team']
        rb_grouped = rb_grouped[output_columns]
        rb_grouped['Year'] = year
        pff.append(rb_grouped)
    pd.concat(pff)

    result = pd.concat(pff, ignore_index=True)
    for column in columns:
        result['Previous_' + column] = result.groupby(['Team', 'position'])['weighted_avg_' + column].shift(1)
    result.to_csv("RBPFF.csv")


get_pff()