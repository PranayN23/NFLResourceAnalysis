import pandas as pd
import re
from rapidfuzz import process, fuzz

from rapidfuzz import process, fuzz


data = pd.read_csv("non_pff.csv")
data.loc[(data['player_name'] == 'Coby Bryant') & (data['Year'].isin([2023, 2024])), 'position'] = 'S'

positions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C', 'DI', 'ED', 'LB', 'CB', 'S']
# Drop columns safely (only if they exist in the DataFrame)
# Merge each position's PFF data using an outer join
sum = 0
s = 0
for pos in positions:
    random = pd.read_csv(pos + 'PFF.csv')
    s += len(random)
    pff = pd.read_csv(pos + '_no_cap.csv')
    columns_to_drop = [
        'Unnamed: 0.1', 'Unnamed: 0_x', 'team', 'position_y', 'entry_year',
        'years_exp', 'pick', 
        'rookie_year', 'draft_club', 'pff', 'stuart', 'johnson', 'hill', 'otc',
        'Year_x', 'merge_player_x', 'fuzzy_matched_player', 'final_player_name',
        'Unnamed: 0_y', 'player', 'team_name'
    ]



    
    if pos == 'QB':
        pos_data = data[data['Position'] == 'QB']
        pos_data.loc[data['player_name'] == "Terrelle Pryor", 'Position'] = 'QB'
    if pos == 'HB':
        pos_data = data[data['Position'] == 'RB']
    if pos == 'WR':
        pos_data = data[data['Position'] == 'WR']
    if pos == 'TE':
        pos_data = data[data['Position'] == 'TE']
    if pos == 'T':
        pos_data = data[data['Position'].isin(['LT/T', 'LT', 'T', 'RT', 'RT/T', 'RT/G', 'T/G', 'T/RT', 
                                                        'T/C', 'T/RT', 'LT/G', 'TE/T'])]
    if pos == 'G':
        pos_data = data[data['Position'].isin(['G', 'G/C', 'G/LT', 'G/RT', 'G/T', 'LT/G'])]
    if pos == 'C':
        pos_data = data[data['Position'].isin(['C', 'C/G'])]
    if pos == 'DI':
        pos_data = data[data['Position'].isin(['DT', 'DT/FB', 'DT/DE', 'DE/DT', 'DE'])]
    if pos == 'ED':
        pos_data = data[data['Position'].isin(['DE', 'OLB/DE', 'DE/LB', 'OLB', 'DE/OLB', 'DE/DT'])]
    if pos == 'LB':
        pos_data = data[data['Position'].isin(['ILB', 'OLB', 'OLB/LB', 'LB', 'ILB/LB', 'OLB/S', 'ILB/S'])]
    if pos == 'CB':
        pos_data = data[data['Position'].isin(['CB', 'CB/FS', 'CB/PR', 'CB/S'])]
    if pos == 'S':
        pos_data = data[data['Position'].isin(['SS/S', 'S', 'FS/S', 'SS', 'FS', 'SS/CB', 'FS/CB', 'SS/LB', 'S/CB'])]

    # Merge using fuzzy-matched names
    position_df = pos_data.merge(pff, left_on=['player_id', 'Team', 'Year'], 
                                right_on=['player_id', 'Team', 'Year'], how='left')
    position_df.drop(columns=['Unnamed: 0', 'age_x', 'Net EPA_x', 'Win %_x', 'Position', 'player_name', 'pff_id_x', 'pff_id_y'
                              ,'adjusted_value_x'], inplace=True)
    position_df.rename(columns={'position_y' : 'position', 'age_y': 'age', 'adjusted_value_y' : 'adjusted_value', 'Net EPA_y' : 'Net EPA', 'Win %_y' : 'Win %'}, inplace=True)

    # Add a column to count NaN values in each row
    position_df['nan_count'] = position_df.isna().sum(axis=1)

    # Sort by 'nan_count' to prioritize rows with fewer NaNs
    position_df = position_df.sort_values(by='nan_count')

    # Drop duplicate rows, keeping the first occurrence (which will be the one with fewer NaNs)
    position_df = position_df.drop_duplicates(subset=['player_id', 'Team', 'Year'], keep='first')

    # Drop the 'nan_count' column
    position_df.drop(columns=['nan_count'], inplace=True)
    position_df.dropna(subset=['player'], inplace=True)


    print(f"{pos} Merged:", len(position_df))

   
    sum += len(position_df)
    position_df.to_csv(f'{pos}.csv', index=False)
cap_data = pd.read_csv('cap_data.csv')
draft_data = pd.read_csv('draft_data.csv')
df = pd.read_csv('nfl_epa.csv')
df2 = pd.read_csv('nfl_win.csv')


print(f"Cap: {len(cap_data)}")
print(f"Draft: {len(draft_data)}")
print(f"EPA: {len(df)}")
print(f"Win: {len(df2)}")
print(f"PFF: {s}")
print(f"Final: {sum}")






