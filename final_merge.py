import pandas as pd
import re
from rapidfuzz import process, fuzz

from rapidfuzz import process, fuzz


data = pd.read_csv("non_pff.csv")
data = data[['Team', 'Year', 'Player', 'player_id', 'adjusted_value', 'Cap_Space', 'position']]
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
    # Drop specified columns
    pff.drop(columns=columns_to_drop, inplace=True)
    pff.rename(columns={'player_id_x': 'player_id'}, inplace=True)
    pff.rename(columns={'position_x': 'position'}, inplace=True)
    pff.rename(columns={'season': 'Year'}, inplace=True)


    
    if pos == 'QB':
        pos_data = data[data['position'] == 'QB']
    if pos == 'HB':
        pos_data = data[data['position'] == 'RB']
    if pos == 'WR':
        pos_data = data[data['position'] == 'WR']
    if pos == 'TE':
        pos_data = data[data['position'] == 'TE']
    if pos == 'T':
        pos_data = data[data['position'] == 'OL']
    if pos == 'G':
        pos_data = data[data['position']== 'OL']
    if pos == 'C':
        pos_data = data[data['position']== 'OL']
    if pos == 'DI':
        pos_data = data[data['position']== 'DL']
    if pos == 'ED':
        pos_data = data[data['position'].isin(['DL', 'LB'])]
    if pos == 'LB':
        pos_data = data[data['position'] == 'LB']
    if pos == 'CB':
        pos_data = data[data['position'] == 'CB']
    if pos == 'S':
        pos_data = data[data['position'] == 'S']

    # Merge using fuzzy-matched names
    position_df = pos_data.merge(pff, left_on=['player_id', 'Team', 'Year'], 
                                right_on=['player_id', 'Team', 'Year'], how='inner')

    # Add a column to count NaN values in each row
    position_df['nan_count'] = position_df.isna().sum(axis=1)

    # Sort by 'nan_count' to prioritize rows with fewer NaNs
    position_df = position_df.sort_values(by='nan_count')

    # Drop duplicate rows, keeping the first occurrence (which will be the one with fewer NaNs)
    position_df = position_df.drop_duplicates(subset=['player_id', 'Team', 'Year'], keep='first')

    # Drop the 'nan_count' column
    position_df.drop(columns=['nan_count'], inplace=True)

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






