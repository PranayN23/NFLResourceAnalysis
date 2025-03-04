import pandas as pd
import numpy as np

def main():
    get_pff()
    result = collect_data()
    position_mapping = {
            'RB/FB': 'HB',
        }
    result['Position'] = result['Position'].map(position_mapping).fillna(result['Position'])
    result.to_csv('data.csv')
    

def get_pff():
    years = list(range(2010, 2025))
    for year in years:
        pff = []
        defense = pd.read_csv('PFF/Defense' + str(year) + '.csv')
        defense = defense[['player', 'position', 'team_name', 'grades_defense', 'snap_counts_defense']]
        positions_of_interest = ['LB', 'DI', 'ED', 'CB', 'S']
        defense = defense[defense['position'].isin(positions_of_interest)]

        # Map positions to broader categories
        position_mapping = {
            'DI': 'DL',
            'ED': 'DL',
            'CB': 'DB',
            'S': 'DB',
            'LB': 'LB'
        }

        defense['position'] = defense['position'].replace(position_mapping)

        defense['weighted_grade'] = defense['grades_defense'] * defense['snap_counts_defense']
        

        defense = defense.groupby(['team_name', 'position']).agg(
            weighted_avg_grades=('weighted_grade', 'sum'),
            total_snaps=('snap_counts_defense', 'sum')
        )

        # Compute the weighted average by dividing the sum of weighted grades by the total snaps
        defense['weighted_avg_grades'] = defense['weighted_avg_grades'] / defense['total_snaps']

        # Drop the 'total_snaps' column if you only want the weighted average in the final result
        defense = defense.drop(columns=['total_snaps']).reset_index()
        ol = pd.read_csv('PFF/OL' + str(year) + '.csv')
        ol = ol[['player', 'position', 'team_name', 'grades_offense', 'snap_counts_offense']]
        positions_of_interest = ['T', 'G', 'C']
        ol = ol[ol['position'].isin(positions_of_interest)]

        # Map positions to broader categories
        position_mapping = {
            'T': 'OL',
            'G': 'OL',
            'C': 'OL',
        }
        ol['position'] = ol['position'].replace(position_mapping)

        ol['weighted_grade'] = ol['grades_offense'] * ol['snap_counts_offense']
        

        ol = ol.groupby(['team_name', 'position']).agg(
            weighted_avg_grades=('weighted_grade', 'sum'),
            total_snaps=('snap_counts_offense', 'sum')
        )
         # Compute the weighted average by dividing the sum of weighted grades by the total snaps
        ol['weighted_avg_grades'] = ol['weighted_avg_grades'] / ol['total_snaps']

        # Drop the 'total_snaps' column if you only want the weighted average in the final result
        ol = ol.drop(columns=['total_snaps']).reset_index()
        

        rb = pd.read_csv('PFF/RB' + str(year) + '.csv')
        rb = rb[['player', 'position', 'team_name', 'grades_offense', 'total_touches']]
        positions_of_interest = ['HB']
        rb = rb[rb['position'].isin(positions_of_interest)]
        rb['weighted_grade'] = rb['grades_offense'] * rb['total_touches']
        

        rb = rb.groupby(['team_name', 'position']).agg(
            weighted_avg_grades=('weighted_grade', 'sum'),
            total_snaps=('total_touches', 'sum')
        )
         # Compute the weighted average by dividing the sum of weighted grades by the total snaps
        rb['weighted_avg_grades'] = rb['weighted_avg_grades'] / rb['total_snaps']

        # Drop the 'total_snaps' column if you only want the weighted average in the final result
        rb = rb.drop(columns=['total_snaps']).reset_index()        
        position_mapping = {
            'HB': 'RB/FB',
        }
        rb['position'] = rb['position'].map(position_mapping)

        catchers = pd.read_csv('PFF/Receiving' + str(year) + '.csv')
        catchers['snap_counts_offense'] = catchers['slot_snaps'] + catchers['wide_snaps'] + catchers['inline_snaps']
        catchers = catchers[['player', 'position', 'team_name', 'grades_offense', 'snap_counts_offense']]
        positions_of_interest = ['WR', 'TE']
        catchers = catchers[catchers['position'].isin(positions_of_interest)]
        catchers['weighted_grade'] = catchers['grades_offense'] * catchers['snap_counts_offense']
        

        catchers = catchers.groupby(['team_name', 'position']).agg(
            weighted_avg_grades=('weighted_grade', 'sum'),
            total_snaps=('snap_counts_offense', 'sum')
        )
         # Compute the weighted average by dividing the sum of weighted grades by the total snaps
        catchers['weighted_avg_grades'] = catchers['weighted_avg_grades'] / catchers['total_snaps']

        # Drop the 'total_snaps' column if you only want the weighted average in the final result
        catchers = catchers.drop(columns=['total_snaps']).reset_index()
        


        qb = pd.read_csv('PFF/QB' + str(year) + '.csv')
        qb = qb[['player', 'position', 'team_name', 'grades_offense', 'passing_snaps']]
        positions_of_interest = ['QB']
        qb = qb[qb['position'].isin(positions_of_interest)]
        qb['weighted_grade'] = qb['grades_offense'] * qb['passing_snaps']
        

        qb = qb.groupby(['team_name', 'position']).agg(
            weighted_avg_grades=('weighted_grade', 'sum'),
            total_snaps=('passing_snaps', 'sum')
        )
         # Compute the weighted average by dividing the sum of weighted grades by the total snaps
        qb['weighted_avg_grades'] = qb['weighted_avg_grades'] / qb['total_snaps']

        # Drop the 'total_snaps' column if you only want the weighted average in the final result
        qb = qb.drop(columns=['total_snaps']).reset_index()
        result = pd.concat([defense, ol, rb, catchers, qb], axis=0)
        team_mapping = {
    'WAS': 'Commanders',
    'TEN': 'Titans',
    'TB': 'Buccaneers',
    'SF': '49ers',
    'SEA': 'Seahawks',
    'PIT': 'Steelers',
    'PHI': 'Eagles',
    'NYJ': 'Jets',
    'NYG': 'Giants',
    'NO': 'Saints',
    'SL': 'Saints',
    'SD' : 'Chargers',
    'NE': 'Patriots',
    'MIN': 'Vikings',
    'MIA': 'Dolphins',
    'LV': 'Raiders',
    'LAC': 'Chargers',
    'LA': 'Rams',
    'KC': 'Chiefs',
    'JAX': 'Jaguars',
    'IND': 'Colts',
    'HST': 'Texans',
    'GB': 'Packers',
    'DET': 'Lions',
    'DEN': 'Broncos',
    'DAL': 'Cowboys',
    'CLV': 'Browns',
    'CIN': 'Bengals',
    'CHI': 'Bears',
    'CAR': 'Panthers',
    'ATL': 'Falcons',
    'ARZ': 'Cardinals',
    'BLT': 'Ravens',
    'BUF': 'Bills',
    'OAK': 'Raiders'
}
        result['Team'] = result['team_name'].replace(team_mapping)
        result = result[['position', 'weighted_avg_grades', 'Team']]
        result.to_csv('PFF' + str(year) + '.csv')


def collect_data():

    # Sample data loading
    # Replace these with your actual data loading mechanism
    pff_data_dfs = {year: pd.read_csv(f'PFF{year}.csv') for year in range(2018, 2023)}
    cap_space_dfs = {year: pd.read_csv(f'cap_data{year}.csv') for year in range(2018, 2023)}
    draft_data_dfs = {year: pd.read_csv(f'draft_data_{year}.csv') for year in range(2018, 2023)}
    av_data_dfs = {year: pd.read_csv(f'value_data_{year}.csv') for year in range(2018, 2023)}

    # Reshape dataframes
    positions = ['QB', 'RB/FB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K/P/LS']
    reshaped_cap_space = pd.concat([reshape_df(df, year, positions) for year, df in cap_space_dfs.items()])
    reshaped_draft_data = pd.concat([reshape_df(df, year, positions) for year, df in draft_data_dfs.items()])

    reshaped_av_data = pd.concat([reshape_df(df, year, positions) for year, df in av_data_dfs.items()])
    reshaped_pff_data = pd.concat([shape_df(df, year, positions) for year, df in pff_data_dfs.items()])

    # Combine dataframes
    combined_df = pd.merge(reshaped_cap_space, reshaped_draft_data, on=['Team', 'Year', 'Position'], suffixes=('_cap_space', '_draft_data'))
    combined_df = pd.merge(combined_df, reshaped_av_data, on=['Team', 'Year', 'Position'])
    combined_df = pd.merge(combined_df, reshaped_pff_data, on=['Team', 'Year', 'Position'])
    combined_df.rename(columns={'Value': 'Current_AV'}, inplace=True)
    combined_df.rename(columns={'weighted_avg_grades': 'Current_PFF'}, inplace=True)

    # Compute previous year AV
    combined_df['Previous_AV'] = combined_df.groupby(['Team', 'Position'])['Current_AV'].shift(1)
    combined_df['Previous_PFF'] = combined_df.groupby(['Team', 'Position'])['Current_PFF'].shift(1)

    team_dfs = {year: pd.read_csv(f'team_success_{year}.csv') for year in range(2019, 2023)}
    reshaped_team = pd.concat([fix_df(df, year) for year, df in team_dfs.items()])
    combined_df = pd.merge(combined_df, reshaped_team, on=['Team', 'Year'])
    # Filter for years past 2019
    final_df = combined_df[combined_df['Year'] >= 2019]
    final_df = final_df[['Team', 'Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Current_AV', 'Previous_PFF', 'Current_PFF', 'Total DVOA', 'win-loss-pct', 'Net EPA']]
    return final_df

def fix_df(df, year):
    df['Year'] = year
    df = df.rename(columns = {'TEAM' : 'Team', 'TOTAL DVOA' : 'Total DVOA'})
    return df

def reshape_df(df, year, position_cols):
    df = df.melt(id_vars=['Team'], value_vars=position_cols, var_name='Position', value_name='Value')
    df['Year'] = year
    return df

def shape_df(df, year, position_cols):
    df['Year'] = year
    df.rename(columns={'position': 'Position'}, inplace=True)

    return df

if __name__ == "__main__":
    main()

