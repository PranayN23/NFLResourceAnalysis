import pandas as pd

def get_pff():
    years = [2018, 2019, 2020, 2021, 2022]
    pff = []
    for year in years:
        qb = pd.read_csv('PFF/QB' + str(year) + '.csv')
        positions_of_interest = ['QB']
        qb = qb[qb['position'].isin(positions_of_interest)]
        
        # List of columns of interest
        columns = [
            'accuracy_percent', 'aimed_passes', 'attempts', 'avg_depth_of_target',
            'avg_time_to_throw', 'bats', 'big_time_throws', 'btt_rate', 
            'completion_percent', 'completions', 'declined_penalties', 
            'def_gen_pressures', 'drop_rate', 'dropbacks', 'drops', 
            'first_downs', 'franchise_id', 'grades_hands_fumble', 
            'grades_offense', 'grades_pass', 'grades_run', 'hit_as_threw', 
            'interceptions', 'passing_snaps', 'penalties', 
            'pressure_to_sack_rate', 'qb_rating', 'sack_percent', 
            'sacks', 'scrambles', 'spikes', 'thrown_aways', 
            'touchdowns', 'turnover_worthy_plays', 'twp_rate', 
            'yards', 'ypa'
        ]
        
        # Initialize weighted columns
        for column in columns:
            if column in qb.columns:
                qb['weighted_' + column] = qb[column] * qb['passing_snaps']

        # Group by team_name and position, summing the weighted columns and total snaps
        aggregation_dict = {
            'passing_snaps': 'sum'  # Total snaps for normalization
        }
        
        for column in columns:
            if column in qb.columns:
                aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

        qb_grouped = qb.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()

        # Calculate the weighted averages
        for column in columns:
            if 'weighted_' + column in qb_grouped.columns:
                qb_grouped['weighted_avg_' + column] = qb_grouped['weighted_' + column] / qb_grouped['passing_snaps']
        
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
        
        qb_grouped['Team'] = qb_grouped['team_name'].replace(team_mapping)

        # Select relevant columns for output
        output_columns = ['position'] + ['weighted_avg_' + column for column in columns if 'weighted_' + column in qb_grouped.columns] + ['Team']
        qb_grouped = qb_grouped[output_columns]
        qb_grouped['Year'] = year
        pff.append(qb_grouped)
    pd.concat(pff)
        

    result = pd.concat(pff, ignore_index=True)
    for column in columns:
        result['Previous_' + column] = result.groupby(['Team', 'position'])['weighted_avg_' + column].shift(1)
    result.to_csv("QBPFF.csv")
get_pff()
