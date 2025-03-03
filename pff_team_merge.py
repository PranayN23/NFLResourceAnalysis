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
        pos_data = data[data['position'] == 'S']

    # Merge using fuzzy-matched names
    position_df = pos_data.merge(pff, left_on=['final_player_name', 'Team', 'season'], 
                                right_on=['merge_player', 'Team', 'Year'], how='right')
    print(f"{pos} Merged:", len(position_df))

    # Save NaN rows after each position merge
    pos_nan_rows = position_df[position_df.isna().any(axis=1)]
    pos_nan_rows.to_csv(f'{pos}_rows_with_nans.csv', index=False)
    print(f"Rows with NaN values in {pos} Merged: {len(pos_nan_rows)}")
    players_with_multiple_nans = pos_nan_rows['merge_player_x'].value_counts()

    # Filter players who have more than one missing data row
    players_with_multiple_nans = players_with_multiple_nans[players_with_multiple_nans > 1]

    # Print the players who have multiple rows with missing data (excluding those with 'MISSING' in any row)
    print("Players with multiple rows containing NaNs (excluding those with 'MISSING' in any row):", players_with_multiple_nans)

    print(len(players_with_multiple_nans))

    sum += len(position_df)
    position_df.to_csv(f'{pos}_no_cap.csv', index=False)
print(sum)