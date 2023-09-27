from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import pandas as pd
import re
import numpy as np

def main():
    web_scrape()

def web_scrape():
    cap_data2022 = web_scrape_data('2022')
    cap_data2022.to_csv('cap_data2022.csv')
    cap_data2021 = web_scrape_data('2021')
    cap_data2021.to_csv('cap_data2021.csv')
    draft_data = web_scrape_rosters()
    draft_data.to_csv('draft_data.csv')

def web_scrape_rosters():
    nfl_teams = ["ARZ", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
                 "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", 
                 "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
                   "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"]
    dataframes = []
    for team in nfl_teams:
        dataframes.append(add_roster(team))
    return convert_dataframes(dataframes)
    
    
def convert_dataframes(dataframes):
    draft_data = pd.concat(dataframes, ignore_index=True)
    draft_data.replace('', np.nan, inplace=True)
    draft_data.fillna(0, inplace=True)
    draft_data['RB/FB'] = draft_data["RB"] + draft_data['FB']
    draft_data['TE'] = draft_data["TE"] + draft_data['TE/FB']
    draft_data["OL"] = draft_data["OC"] + draft_data['OC/OG'] + draft_data['OT/OC'] 
    + draft_data['OG/OC'] + draft_data["OT/OG"] + draft_data["OG/OT"]
    + draft_data["OG"] + draft_data["OT"] 
    draft_data['DL'] = draft_data['DT'] + draft_data['DE'] + draft_data['NT']
    draft_data['LB'] = draft_data['LB'] + draft_data['ILB'] + draft_data['OLB']
    draft_data['DB'] = draft_data['S'] + draft_data['CB'] + draft_data['FS']
    draft_data["K/P/LS"] = draft_data['PK'] + draft_data['PT'] + draft_data['LS']
    draft_data = draft_data[['Team', 'QB', 'RB/FB', 'WR', "TE", "OL", "DL", "LB", "DB", "K/P/LS"]]
    return draft_data


def add_roster(team):
    url = 'https://www.ourlads.com/nfldepthcharts/roster/' + team
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    headers = []
    data = []
    for th in table.find('thead').find_all('th'):
        headers.append(th.text.strip())
    for row in table.find('tbody').find_all('tr'):
        row_data = []
        for td in row.find_all('td'):
            t = td.text.strip()
            if (t != 'Active Players' and t != 'Reserves' and t != 'Practice Squad'):
                row_data.append(t)
                data.append(row_data)
    roster_data = pd.DataFrame(data, columns=headers)
    unique_lists = list(set(tuple(x) for x in data))
    # Convert the lists back to their original format (optional)
    data = [list(x) for x in unique_lists]
    return clean_roster_data(roster_data, team)

def clean_roster_data(roster_data, team):
    roster_data = roster_data.drop_duplicates()
    roster_data.dropna()
    roster_data = roster_data[['Player', 'Pos.', 'Orig. Team', 'Draft Status']]   
    # we remove the original team filter for now and will come back to it later
    # roster_data = roster_data[roster_data['Orig. Team'].str.contains(team)]
    roster_data['Draft Year'] = roster_data['Draft Status'].apply(getYear)
    # we remove the year filter for now and will come back to it later
    # roster_data = roster_data[(roster_data['Draft Year'] >= 19) & (roster_data['Draft Year'] <= 22)]
    roster_data['Draft Value'] = roster_data['Draft Status'].apply(getValue)
    total_value = roster_data['Draft Value'].sum()
    grouped = roster_data.groupby('Pos.')['Draft Value'].sum()
    grouped = grouped / total_value * 100
    grouped = grouped.reset_index()
    grouped['Team'] = pd.Series(team, index=range(len(grouped)))
    grouped = grouped.pivot(index='Team', columns='Pos.', values='Draft Value')
    grouped.reset_index(inplace=True)
    grouped = grouped.fillna(0)
    # Rename the columns to match your desired format
    grouped.columns.name = None  # Remove the columns' name
    grouped.columns = grouped.columns.str.replace(' ', '')  # Remove spaces in column names
    team_mapping = {
        'ARZ': 'Cardinals',
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
        'LAC': 'Chargers',
        'LAR': 'Rams',
        'LV': 'Raiders',
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
        'WAS': 'Commanders'
    }
    # Map the 'Team' column to full names using the mapping dictionary
    grouped['Team'] = grouped['Team'].map(team_mapping)
    return grouped
    
def getYear(info):
    year = info.split()[0]
    if year == '22/7':
        return 22
    elif year == 'SF23':
        return 23
    else:
        return int(year)

def getValue(info):
    tokens = info.split(' ')
    pick_value = pd.read_csv('nfl_pick_value.csv')
    try:
        if len(tokens) == 2 or len(tokens) == 0 or tokens[0] == '22/7' or tokens[2] == '' or tokens[2] == '020R':
            return 0
        else:
            pick = int(tokens[2])
            if (pick < len(pick_value)):
                return pick_value.loc[pick, 'Value']
            else:
                return 0
    except IndexError as e:
        return 0

def web_scrape_data(year):
    url = 'https://www.spotrac.com/nfl/positional/breakdown/' + year + '/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    headers = []
    data = []
    for th in table.find('thead').find_all('th'):
        headers.append(th.text.strip())
    for row in table.find('tbody').find_all('tr'):
        row_data = []
        for td in row.find_all('td'):
            t = td.text.strip()
            if ('M' in t and 'Miami' not in t and 'Minnesota' not in t):
                t = t.split(' ')[0]
                t = float(t)
                if (year == '2022'):
                    t = t / 208200000 * 100
                else:
                    t = t / 182500000 * 100
                row_data.append(t)
            else:
                row_data.append(td.text.strip())
            data.append(row_data)
    unique_lists = list(set(tuple(x) for x in data))
    # Convert the lists back to their original format (optional)
    data = [list(x) for x in unique_lists]
    cap_data = pd.DataFrame(data, columns=headers)
    cap_data['Team'] = cap_data['Team'].apply(lambda x: x.split()[len(x.split()) - 1])
    return cap_data


if __name__ == "__main__":
    main()