import pandas as pd

def get_secondary_defense_data():
    # Starting from 2019 with 2018 as baseline for previous year data
    years = [2018, 2019, 2020, 2021, 2022]
    defense_data = []
    positions_of_interest = ['CB', 'S', 'ED', 'LB', 'DI']
    
    for year in years:
        # Load the CSV for the current year
        df = pd.read_csv(f'PFF/Defense{year}.csv')

        df = df.drop(columns=['catch_rate', 'yards_per_reception', 'longest', 'yards_after_catch'])
        
        # Filter for positions of interest
        df = df[df['position'].isin(positions_of_interest)]
        
        # Include all relevant columns for CB and S, excluding identifiers
        columns = [col for col in df.columns if col not in ['team_name', 'position', 'Year', 'snap_counts_defense']]
        
        # Convert all relevant columns to numeric, coercing errors to NaN
        df[columns] = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Initialize weighted columns using 'snap_counts_defense'
        for column in columns:
            if column in df.columns:
                df['weighted_' + column] = df[column] * df['snap_counts_defense']

        # Group by team_name and position, summing the weighted columns and total snaps
        aggregation_dict = {
            'snap_counts_defense': 'sum'  # Total snaps for normalization
        }
        
        for column in columns:
            if 'weighted_' + column in df.columns:
                aggregation_dict['weighted_' + column] = 'sum'  # Sum of weighted values

        # Aggregate data
        df_grouped = df.groupby(['team_name', 'position']).agg(aggregation_dict).reset_index()
        
        # Assign the current year to a new column before selecting output columns
        df_grouped['Year'] = year

        # Calculate weighted averages for each column
        for column in columns:
            if 'weighted_' + column in df_grouped.columns:
                df_grouped['weighted_avg_' + column] = df_grouped['weighted_' + column] / df_grouped['snap_counts_defense']
        
        # Replace team abbreviations with full names
        team_mapping = {
            'WAS': 'Commanders', 'TEN': 'Titans', 'TB': 'Buccaneers', 'SF': '49ers',
            'SEA': 'Seahawks', 'PIT': 'Steelers', 'PHI': 'Eagles', 'NYJ': 'Jets',
            'NYG': 'Giants', 'NO': 'Saints', 'NE': 'Patriots', 'MIN': 'Vikings',
            'MIA': 'Dolphins', 'LV': 'Raiders', 'LAC': 'Chargers', 'LA': 'Rams',
            'KC': 'Chiefs', 'JAX': 'Jaguars', 'IND': 'Colts', 'HST': 'Texans',
            'GB': 'Packers', 'DET': 'Lions', 'DEN': 'Broncos', 'DAL': 'Cowboys',
            'CLV': 'Browns', 'CIN': 'Bengals', 'CHI': 'Bears', 'CAR': 'Panthers',
            'ATL': 'Falcons', 'ARZ': 'Cardinals', 'BLT': 'Ravens', 'BUF': 'Bills', 'OAK': 'Raiders'
        }
        
        df_grouped['Team'] = df_grouped['team_name'].replace(team_mapping)

        # Select relevant columns for output, keeping 'Year' as one of the first columns
        output_columns = ['Year', 'position', 'Team', 'snap_counts_defense'] + ['weighted_avg_' + column for column in columns if 'weighted_' + column in df_grouped.columns]
        df_grouped = df_grouped[output_columns]
        
        # Append the yearly data to the list
        defense_data.append(df_grouped)
    
    # Concatenate all years including 2018 as baseline, and keep data from 2019 onwards in final output
    combined_data = pd.concat(defense_data, ignore_index=True)
    
    # Add previous year's statistics for each team and position
    for column in columns:
        combined_data['Previous_' + column] = combined_data.groupby(['Team', 'position'])['weighted_avg_' + column].shift(1)
    
    # Filter to keep only 2019 onward in final output, after shifting for previous data
    combined_data = combined_data[combined_data['Year'] > 2018]
    
    # Save to CSV, overwriting the existing file
    combined_data.to_csv("SecondaryDefenseData.csv", index=False)

get_secondary_defense_data()
