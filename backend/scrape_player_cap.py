from bs4 import BeautifulSoup
import requests
import pandas as pd


all_data = []  # List to hold the data for all years
years = list(range(2010, 2025))
for year in years:
# Example: If you're scraping from a webpage
    url = "https://www.spotrac.com/nfl/rankings/player/_/year/" + str(year) + "/sort/cap_total"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    print(year)
    # Select all li elements under #table
    items = soup.select("#table li")
    data = []
    for index, item in enumerate(items, start=1):
        player = item.select_one(".link").text.strip() if item.select_one(".link") else "N/A"
        
        # Check if team and position elements are present
        team_pos_element = item.select_one("small")
        if not team_pos_element:
            continue  # Skip this item if team and position are missing
        
        team_pos = team_pos_element.text.strip().split(", ")
        team = team_pos[0].split()[-1]  # Extract team
        position = team_pos[1]  # Extract position
        
        # Safely extract cap space info
        cap_space_element = item.select_one(".medium")
        cap_space = cap_space_element.text.strip().replace("$", "").replace(",", "") if cap_space_element else "0"
        if cap_space == 'Post June 1st Designation\nPost 6/1\n                                \n                4100000':
            cap_space = 4100000
        data.append([player, team, position, int(cap_space)])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Player", "Team", "Position", "Cap_Space"])
    team_mapping = {
        'ARI': 'Cardinals',
        'ATL': 'Falcons',
        'BAL': 'Ravens',
        'BUF': 'Bills',
        'CAR': 'Panthers',
        'CHI': 'Bears',
        'CIN': 'Bengals',
        'CLE': 'Browns',
        'DAL': 'Cowboys',
        'DEN': 'Broncos',
        'DET': 'Lions',
        'GB': 'Packers',
        'HOU': 'Texans',
        'IND': 'Colts',
        'JAX': 'Jaguars',
        'KC': 'Chiefs',
        'LV': 'Raiders',
        'LAC': 'Chargers',
        'LAR': 'Rams',
        'MIA': 'Dolphins',
        'MIN': 'Vikings',
        'NE': 'Patriots',
        'NO': 'Saints',
        'NYG': 'Giants',
        'NYJ': 'Jets',
        'PHI': 'Eagles',
        'PIT': 'Steelers',
        'SF': '49ers',
        'SEA': 'Seahawks',
        'TB': 'Buccaneers',
        'TEN': 'Titans',
        'WAS': 'Commanders',
        'OAK': 'Raiders',
        'SD' : 'Chargers',
        'STL': 'Rams'
    }
    def adjust_value(row):
        if row['year'] == 2022:
            return row['Cap_Space'] / 208200000 * 100
        elif row['year'] == 2021:
            return row['Cap_Space'] / 182500000 * 100
        elif row['year'] == 2020:
            return row['Cap_Space'] / 198200000 * 100
        elif row['year'] == 2019:
            return row['Cap_Space'] / 188200000 * 100
        elif row['year'] == 2018:
            return row['Cap_Space'] / 177200000 * 100
        elif row['year'] == 2017:
            return row['Cap_Space'] / 167000000 * 100
        elif row['year'] == 2016:
            return row['Cap_Space'] / 155270000 * 100
        elif row['year'] == 2015:
            return row['Cap_Space'] / 143280000 * 100
        elif row['year'] == 2014:
            return row['Cap_Space'] / 133000000 * 100
        elif row['year'] == 2013:
            return row['Cap_Space'] / 123600000 * 100
        elif row['year'] == 2012:
            return row['Cap_Space'] / 120600000 * 100
        elif row['year'] == 2011:
            return row['Cap_Space'] / 120375000 * 100
        elif row['year'] == 2010:
            return row['Cap_Space'] / 123000000 * 100  
        elif row['year'] == 2023:
            return row['Cap_Space'] / 224800000 * 100
        elif row['year'] == 2024:
            return row['Cap_Space'] / 255400000 * 100
        else:
            return row['Cap_Space']  # In case there is no matching year
    df['year'] = year
    #print(df['Cap_Space'])
    df['Cap_Space'] = df.apply(adjust_value, axis=1)
    #print(df['Cap_Space'])

    df['Team'] = df['Team'].map(team_mapping)
    
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

# Output the final DataFrame
print(f"Total rows extracted: {len(final_df)}")
print(final_df.head())
final_df.to_csv('cap_data.csv')