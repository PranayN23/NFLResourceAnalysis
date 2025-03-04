import pandas as pd
import re
from rapidfuzz import process, fuzz

from rapidfuzz import process, fuzz


def normalize_name(name):
    if pd.isna(name):  # Handle NaN values
        return name
    name = name.lower().strip()  # Convert to lowercase and strip spaces
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', name)  # Remove common suffixes
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name

def fuzzy_match_name(name, choices, threshold=69):
    if pd.isna(name):
        return None  # Skip NaNs
    
    # First, check for exact match
    exact_match = next((choice for choice in choices if name.lower() == choice.lower()), None)
    if exact_match:
        return exact_match
    
    # If no exact match, use fuzzy matching
    match, score, _ = process.extractOne(name, choices, scorer=fuzz.ratio)
    return match if score >= threshold else None

data = pd.read_csv("draft.csv")
data.fillna("MISSING", inplace=True)
print(data['position'].unique())

positions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C', 'DI', 'ED', 'LB', 'CB', 'S']
# Drop columns safely (only if they exist in the DataFrame)
# Merge each position's PFF data using an outer join
sum = 0
s = 0
for pos in positions:
    pff = pd.read_csv(pos + 'PFF.csv')
    if pos == 'HB':
        pff.loc[pff['player'] == 'Beanie Wells', 'player'] = 'Chris Wells'
        pff.loc[pff['player'] == 'Cadillac Williams', 'player'] = 'Carnell Williams'
    pff.fillna("MISSING", inplace=True)
    print(f"{pos} PFF:", len(pff))
    s += len(pff)
    pff['merge_player'] = pff['player'].apply(normalize_name)
    pff['team_name'] = pff['team_name'].str.strip()
    pff_names = pff['merge_player'].unique()
    data['merge_player'] = data['player_name'].apply(normalize_name)

    # Apply fuzzy matching to map player names in data
    data['fuzzy_matched_player'] = data['merge_player'].apply(lambda x: fuzzy_match_name(x, pff_names))
    
    # If no match is found, keep the original name
    data['final_player_name'] = data['fuzzy_matched_player'].fillna(data['merge_player'])
    pff = pff[pff['position'] == pos]
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
        pos_data = data[data['position'] == 'CB']
    print(pos_data.columns)
    print(pff.columns)

    # Merge using fuzzy-matched names
    position_df = pos_data.merge(pff, left_on=['final_player_name', 'Team', 'Year'], 
                                right_on=['merge_player', 'Team', 'Year'], how='left')
    print(f"{pos} Merged:", len(position_df))
        # Merge using fuzzy-matched names
    position_df.drop(columns=['Unnamed: 0_x', 'position_x', 'player_name', 'merge_player_x','fuzzy_matched_player','final_player_name','Unnamed: 0_y', 'player_id_y','team_name', 'merge_player_y'], inplace=True)
    position_df.rename(columns={'player_id_x': 'player_id', 'position_y': 'position'}, inplace=True)


    sum += len(position_df)
    position_df.to_csv(f'{pos}_no_cap.csv', index=False)
print(sum)