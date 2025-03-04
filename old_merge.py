import pandas as pd
import re
from rapidfuzz import process, fuzz

from rapidfuzz import process, fuzz



# Load datasets
df = pd.read_csv('nfl_epa.csv')
df['Team'] = df['Team'].str.strip()
df.fillna("MISSING", inplace=True)  # Fill existing NaNs
print('EPA:', len(df))

df2 = pd.read_csv('nfl_win.csv')
df2.fillna("MISSING", inplace=True)
print('Win:', len(df2))

team = df.merge(df2, on=['Team', 'Year'], how='outer')
print('Team:', len(team))


draft = pd.read_csv('draft_data.csv')
draft.fillna("MISSING", inplace=True)
print('Draft:', len(draft))
data = draft.merge(team, left_on=['season', 'team'], right_on=['Year', 'Team'], how='outer')
print('Data:', len(data))
data.to_csv("draft.csv")


# Save NaN rows after second merge (draft + team)
data_nan_rows = data[data.isna().any(axis=1)]
data_nan_rows.to_csv('data_rows_with_nans.csv', index=False)
print(f"Rows with NaN values in Data merge: {len(data_nan_rows)}")

def normalize_name(name):
    if pd.isna(name):  # Handle NaN values
        return name
    name = name.lower().strip()  # Convert to lowercase and strip spaces
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', name)  # Remove common suffixes
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name

cap = pd.read_csv('cap_data.csv')
cap.fillna("MISSING", inplace=True)

cap['Player'].replace({'Matt Judon': 'Matthew Judon', 
                       'Christopher Carson' : 'Chris Carson',
                        'Trenton Brown': 'Trent Brown', 
                        'Ron Leary': 'Ronald Leary', 'Matt Slater' : 'Matthew Slater',
                        'Chris Hubbard' : 'Christopher Hubbard',
                        'Robbie Anderson' : 'Robby Anderson', 
                        'Evan Dietrich-Smith' : 'Evan Smith',
                        'Benjamin Ijalana' : 'Ben Ijalana',
                        'Michael Mitchell' : 'Mike Mitchell',
                        'Travis Carrie' : 'T.J. Carrie',
                        'Nickell Robey' : 'Nickell Robey-Coleman',
                        'Michael Person' : 'Mike Person',
                        'Adam-Pacman Jones' : 'Adam Jones',
                        'HaHa Clinton-Dix' : 'Ha Ha Clinton-Dix',
                        'Christopher Conte' : 'Chris Conte',
                        'William Beatty' : 'Will Beatty',
                        'Joshua Mauga' : 'Josh Mauga',
                        'Ricky Wagner' : 'Rick Wagner',
                        'Stevie Johnson' : 'Steve Johnson',
                        'Joseph Noteboom' : 'Joe Noteboom',
                        'John Cyprien' : 'Johnathan Cyprien',
                        'Ced Wilson': 'Cedrick Wilson',
                        'Nyheim Miller-Hines' : 'Nyheim Hines',
                        'Sebastian Joseph' : 'Sebastian Joseph-Day',
                        'Johnathan Ford' : 'Rudy Ford',
                        'Nathan Stupar' : 'Nate Stupar',
                        'Foley Fatukasi' : 'Folorunso Fatukasi',
                        'Joseph Jones' : 'Joe Jones',
                        'Sean Bunting' : 'Sean Murphy-Bunting',
                         'Darius Leonard' : 'Shaquille Leonard',
                        'Ugochukwu Amadi' : 'Ugo Amadi',
                        'Joshua Thomas' : 'Josh Thomas',
                         'Gregory Toler' : 'Greg Toler',
                         'Justin March' : 'Justin March-Lillard',
                         'Deonte Harris' : 'Deonte Harty',
                         'Johnny Newton' : 'Jer\'Zhan Newton'}, inplace=True)

# Replace player names in the data DataFrame
data['player_name'].replace({'Matt Judon': 'Matthew Judon', 
                             'Chris Wells' : 'Beanie Wells',
                        'Trenton Brown': 'Trent Brown', 
                        'Ron Leary': 'Ronald Leary', 'Matt Slater' : 'Matthew Slater',
                        'Chris Hubbard' : 'Christopher Hubbard',
                        'Robbie Anderson' : 'Robby Anderson', 
                        'Evan Dietrich-Smith' : 'Evan Smith',
                        'Benjamin Ijalana' : 'Ben Ijalana',
                        'Michael Mitchell' : 'Mike Mitchell',
                        'Travis Carrie' : 'T.J. Carrie',
                        'Nickell Robey' : 'Nickell Robey-Coleman',
                        'Michael Person' : 'Mike Person',
                        'Adam-Pacman Jones' : 'Adam Jones',
                        'HaHa Clinton-Dix' : 'Ha Ha Clinton-Dix',
                        'Christopher Conte' : 'Chris Conte',
                        'William Beatty' : 'Will Beatty',
                        'Joshua Mauga' : 'Josh Mauga',
                        'Ricky Wagner' : 'Rick Wagner',
                        'Stevie Johnson' : 'Steve Johnson',
                        'Joseph Noteboom' : 'Joe Noteboom',
                         'John Cyprien' : 'Johnathan Cyprien',
                         'Ced Wilson': 'Cedrick Wilson',
                         'Nyheim Miller-Hines' : 'Nyheim Hines',
                         'Sebastian Joseph' : 'Sebastian Joseph-Day',
                         'Johnathan Ford' : 'Rudy Ford',
                         'Nathan Stupar' : 'Nate Stupar',
                         'Foley Fatukasi' : 'Folorunso Fatukasi',
                         'Joseph Jones' : 'Joe Jones',
                         'Sean Bunting' : 'Sean Murphy-Bunting',
                         'Darius Leonard' : 'Shaquille Leonard',
                         'Ugochukwu Amadi' : 'Ugo Amadi',
                         'Joshua Thomas' : 'Josh Thomas',
                         'Gregory Toler' : 'Greg Toler',
                         'Justin March' : 'Justin March-Lillard',
                         'Deonte Harris' : 'Deonte Harty',
                         'Johnny Newton' : 'Jer\'Zhan Newton'}, inplace=True)

cap['merge_player'] = cap['Player'].apply(normalize_name)

cap = cap[~cap['Position'].isin(['P', 'K', 'LS'])]
data['merge_player'] = data['player_name'].apply(normalize_name)

cap['Team'] = cap['Team'].str.strip()
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


# Get unique player names from cap_data for fuzzy matching
cap_names = cap["merge_player"].unique()

# Apply fuzzy matching to player names in team_data
data["fuzzy_matched_player"] = data["merge_player"].apply(lambda x: fuzzy_match_name(x, cap_names))

# If no match is found, keep the original name
data["final_player_name"] = data["fuzzy_matched_player"].fillna(data["merge_player"])

# Merge using the fuzzy-matched names
non_pff = data.merge(cap, left_on=["Team", "Year", "final_player_name"], 
                              right_on=["Team", "year", "merge_player"], how="left")
print('Non-PFF:', len(non_pff))
non_pff.to_csv('non_pff.csv')



# Remove columns containing 'position' in their names
# non_pff = non_pff.loc[:, ~non_pff.columns.str.contains('position', case=False)]
# List of positions
positions = ['QB', 'HB', 'WR', 'TE', 'T', 'G', 'C', 'DI', 'ED', 'LB', 'CB', 'S']
columns_to_drop = [
    'position', 'Unnamed: 0_x', 'player_name', 'merge_player_x', 'fuzzy_matched_player', 'final_player_name', 'merge_player_x', 'season', 'years_exp', 'entry_year', 'rookie_year',
    'draft_club', 'age', 'pick', 'pff', 'stuart', 'johnson', 'hill', 'otc', 'Unnamed: 0_y', 'merge_player_y', 'Year']
temp = ['SS/S' 'S' 'CB' 'FS/S', 'DE' 'DT' 'FB' 'ILB' 'OLB/DE' 'ILB/LB'
 'OLB/LB' 'DE/LB' 'LB' 'RT/T' 'G' 'LT/T' 'C' 'QB' 'RB' 'TE' 'WR' 'T'
 'RT/G' 'RT' 'CB/FS' 'DT/DE' 'G/C' 'OLB' 'DE/OLB' 'G/T' 'LT' 'FS' 'T/G'
 'LT/G' 'SS/CB' 'TE/DE' 'FS/CB' 'KR' 'ILB/S' 'TE/T' 'DE/DT' 'T/RT' 'SS/LB'
 'TE/LS' 'G/LT' 'G/RT' 'WR/KR' 'SS' 'C/G' 'OL' 'T/C' 'CB/PR' 'DT/FB'
 'S/CB' 'OLB/S' 'CB/S' 'FB/TE']
# Drop columns safely (only if they exist in the DataFrame)
non_pff = non_pff.drop(columns=[col for col in columns_to_drop if col in non_pff.columns])
non_pff.loc[non_pff['Player'] == "Tim Tebow", 'Position'] = 'QB'




# Merge each position's PFF data using an outer join
sum = 0
s = 0
for pos in positions:
    pff = pd.read_csv(pos + 'PFF.csv')
    pff.fillna("MISSING", inplace=True)
    print(f"{pos} PFF:", len(pff))
    s += len(pff)
    pff['merge_player'] = pff['player'].apply(normalize_name)
    pff['team_name'] = pff['team_name'].str.strip()

    
    
    pff_names = pff['merge_player'].unique()









    non_pff['merge_player'] = non_pff['Player'].apply(normalize_name)

    # Apply fuzzy matching to map player names in non_pff
    non_pff['fuzzy_matched_player'] = non_pff['merge_player'].apply(lambda x: fuzzy_match_name(x, pff_names))
    
    # If no match is found, keep the original name
    non_pff['final_player_name'] = non_pff['fuzzy_matched_player'].fillna(non_pff['merge_player'])
    pff = pff[pff['position'] == pos]
    if pos == 'QB':
        pos_non_pff = non_pff[non_pff['Position'] == 'QB']
        pos_non_pff.loc[non_pff['Player'] == "Terrelle Pryor", 'Position'] = 'QB'
    if pos == 'HB':
        pos_non_pff = non_pff[non_pff['Position'] == 'RB']
    if pos == 'WR':
        pos_non_pff = non_pff[non_pff['Position'] == 'WR']
    if pos == 'TE':
        pos_non_pff = non_pff[non_pff['Position'] == 'TE']
    if pos == 'T':
        pos_non_pff = non_pff[non_pff['Position'].isin(['T', 'RT/T', 'RT/G', 'G/T', 'T/G', 'T/RT', 'T/C', 'G/T', 'T/RT'])]
    if pos == 'G':
        pos_non_pff = non_pff[non_pff['Position'].isin(['G', 'G/C', 'G/LT', 'G/RT', 'G/T', 'C/G'])]
    if pos == 'C':
        pos_non_pff = non_pff[non_pff['Position'].isin(['C', 'C/G', 'T/C'])]
    if pos == 'DI':
        pos_non_pff = non_pff[non_pff['Position'].isin(['DT', 'DT/DE', 'DE/DT'])]
    if pos == 'ED':
        pos_non_pff = non_pff[non_pff['Position'].isin(['DE', 'OLB/DE', 'DE/LB', 'OLB', 'DE/OLB', 'TE/DE', 'DE/DT'])]
    if pos == 'LB':
        pos_non_pff = non_pff[non_pff['Position'].isin(['ILB', 'OLB', 'OLB/LB', 'ILB/LB', 'OLB/S', 'ILB/S', 'SS/LB'])]
    if pos == 'CB':
        pos_non_pff = non_pff[non_pff['Position'].isin(['CB', 'CB/FS', 'CB/PR', 'S/CB', 'CB/S'])]
    if pos == 'S':
        pos_non_pff = non_pff[non_pff['Position'].isin(['SS/S', 'FS/S', 'SS', 'FS', 'SS/CB', 'FS/CB'])]

    # Merge using fuzzy-matched names
    position_df = pos_non_pff.merge(pff, left_on=['final_player_name', 'Team', 'year'], 
                                right_on=['merge_player', 'Team', 'Year'], how='left')
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
    position_df.to_csv(f'{pos}.csv', index=False)
print('EPA:', len(df))
print('Win:', len(df2))
print('Draft:', len(draft))
print('Cap:', len(cap))
print('Pre-PFF Merge', len(non_pff))
print('PFF', s)
print('Total', sum)
