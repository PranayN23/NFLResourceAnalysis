import pandas as pd

def get_pff():
    years = [2018, 2019, 2020, 2021, 2022]
    pff = []
    for year in years:
        wr = pd.read_csv('PFF/Receiving' + str(year) + '.csv')
        positions_of_interest = ['WR']
        wr = wr[wr['position'].isin(positions_of_interest)]
        
        # List of columns of interest
        columns = [
            'avg_depth_of_target', 'avoided_tackles', 'caught_percent', 'contested_catch_rate',
            'contested_receptions', 'contested_targets', 'declined_penalties', 'drop_rate', 'drops',
            'first_downs', 'franchise_id', 'fumbles', 'grades_hands_drop', 'grades_hands_fumble',
            'grades_offense', 'grades_pass_block', 'grades_pass_route', 'inline_rate',
            'interceptions', 'longest', 'pass_block_rate', 'pass_blocks', 'pass_plays',
            'penalties,receptions', 'route_rate', 'routes', 'slot_rate', 'targeted_wr_rating',
            'targets', 'touchdowns', 'wide_rate', 'yards', 'yards_after_catch',
            'yards_after_catch_per_reception', 'yards_per_reception', 'yprr'
        ]

        wr['total_snaps'] = wr['inline_snaps'] + wr['slot_snaps'] + wr['wide_snaps']
        
        # Initialize weighted columns
        for column in columns:
            if column in wr.columns:
                wr['weighted_' + column] = wr[column] * wr['total_snaps']

        # Group by team_name and position, summing the weighted columns and total snaps
        aggregation_dict = {
            'total_snaps': 'sum'  # Total snaps for normalization
        }
        
        for column in columns:
            if column in wr.columns:
                aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

        wr_grouped = wr.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()

        # Calculate the weighted averages
        for column in columns:
            if 'weighted_' + column in wr_grouped.columns:
                wr_grouped['weighted_avg_' + column] = wr_grouped['weighted_' + column] / wr_grouped['total_snaps']
        
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
        
        wr_grouped['Team'] = wr_grouped['team_name'].replace(team_mapping)

        # Select relevant columns for output
        output_columns = ['position'] + ['weighted_avg_' + column for column in columns if 'weighted_' + column in wr_grouped.columns] + ['Team']
        wr_grouped = wr_grouped[output_columns]
        wr_grouped['Year'] = year
        pff.append(wr_grouped)
    pd.concat(pff)
    result = pd.concat(pff, ignore_index=True)
    result.to_csv("wrPFF.csv")
get_pff()