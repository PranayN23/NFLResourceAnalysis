"""
Pranay Nandkeolyar

This module web scrapes data from different web sites
regarding nfl team resource stats, such as draft capital spent 
per position and cap space spent per position. 
This required the use of many libraries including urllib (web scraping), regex (re),
pandas (CSV processing), numpy (data manipulation)
BeautifulSoup (html parsing), requests (web scraping). 
"""
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import pandas as pd
import re
import numpy as np

def main():
    """
    We call our web scrape method here
    """
    web_scrape()

def web_scrape():
    """
    We call our specific web scrape methods here
    and store the resulting data as csv files
    """
    # we use one method to scrape cap space data 
    # and another for draft dara
    cap_data2022 = web_scrape_data('2022')
    cap_data2022.to_csv('cap_data2022.csv')
    cap_data2021 = web_scrape_data('2021')
    cap_data2021.to_csv('cap_data2021.csv')
    cap_data2022 = web_scrape_data('2020')
    cap_data2022.to_csv('cap_data2020.csv')
    cap_data2021 = web_scrape_data('2019')
    cap_data2021.to_csv('cap_data2019.csv')
    draft_data = web_scrape_rosters(False)
    draft_data.to_csv('draft_data.csv')
    draft_data = web_scrape_rosters(True)
    draft_data.to_csv('draft_data_last_4.csv')

def web_scrape_rosters(last_4):
    """
    Here we webscrape every team's roster
    for their draft data
    @param last_4 - a boolean representing whether or not we want the last 4
    years only or all draft data
    returns a dataframe of each team's releative draft resources spent
    on each position
    """
    # we make a list of every team's shorthand
    nfl_teams = ["ARZ", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
                 "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", 
                 "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
                   "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"]
    # we loop through every team and scrape their roster page
    dataframes = []
    for team in nfl_teams:
        dataframes.append(add_roster(team, last_4))
    return convert_dataframes(dataframes)
    
    
def convert_dataframes(dataframes):
    """
    Takes a list of each teams roster data and 
    returns it as one dataframe with the corrected positional information
    returns a dataframe of each team's releative draft resources spent
    on each position
    """
    draft_data = pd.concat(dataframes, ignore_index=True)
    # we fill empty values
    draft_data.replace('', np.nan, inplace=True)
    draft_data.fillna(0, inplace=True)
    # we rework each position's definition to match our desired output
    draft_data['RB/FB'] = draft_data["RB"] + draft_data['FB']
    draft_data['TE'] = draft_data["TE"] + draft_data['TE/FB']
    draft_data["OL"] = draft_data["OC"] + draft_data['OC/OG'] + draft_data['OT/OC'] 
    + draft_data['OG/OC'] + draft_data["OT/OG"] + draft_data["OG/OT"]
    + draft_data["OG"] + draft_data["OT"] 
    draft_data['DL'] = draft_data['DT'] + draft_data['DE'] + draft_data['NT']
    draft_data['LB'] = draft_data['LB'] + draft_data['ILB'] + draft_data['OLB']
    draft_data['DB'] = draft_data['S'] + draft_data['CB'] + draft_data['FS']
    draft_data["K/P/LS"] = draft_data['PK'] + draft_data['PT'] + draft_data['LS']
    # we remove excess columns
    draft_data = draft_data[['Team', 'QB', 'RB/FB', 'WR', "TE", "OL", "DL", "LB", "DB", "K/P/LS"]]
    return draft_data


def add_roster(team, last_4):
    """
    Scrapes a teams roster
    @param team - the team's roster to scrape
    @param last_4 - a boolean representing whether or not we want the last 4
    years only or all draft data
    returns a dataframe of a team's draft info
    """
    # we get our website and set up beautiful soup
    url = 'https://www.ourlads.com/nfldepthcharts/roster/' + team
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    headers = []
    data = []
    # we get our headers/column names
    for th in table.find('thead').find_all('th'):
        headers.append(th.text.strip())
    # we process each row here
    for row in table.find('tbody').find_all('tr'):
        row_data = []
        for td in row.find_all('td'):
            t = td.text.strip()
            # we prevent edge breaks
            if (t != 'Active Players' and t != 'Reserves' and t != 'Practice Squad'):
                row_data.append(t)
                data.append(row_data)
    # we make sure each row is unique
    unique_lists = list(set(tuple(x) for x in data))
    # Convert the lists back to their original format (optional)
    data = [list(x) for x in unique_lists]
    # we convert our data to a dataframe
    roster_data = pd.DataFrame(data, columns=headers)
    return clean_roster_data(roster_data, team, last_4)

def clean_roster_data(roster_data, team, last_4):
    """
    Cleans a specific teams roster
    @param roster_data - the team's scraped roster info 
    @param team - the name of the team 
    @param last_4 - a boolean representing whether or not we want the last 4
    years only or all draft data
    returns a cleaned dataframe of a team's draft info
    """
    # we remove dupicates and na values
    roster_data = roster_data.drop_duplicates()
    roster_data.dropna()
    # we filter down to data we want
    roster_data = roster_data[['Player', 'Pos.', 'Orig. Team', 'Draft Status']] 
    # we get the draft year for each player 
    roster_data['Draft Year'] = roster_data['Draft Status'].apply(getYear) 
    # if we want only the last 4 years of data
    if (last_4):
        # we make sure the team has actually drafted that player
        # we filter down to only include the draft years we want
        roster_data = roster_data[roster_data['Orig. Team'].str.contains(team)]
        roster_data = roster_data[(roster_data['Draft Year'] >= 19) & (roster_data['Draft Year'] <= 22)]
    # we get the draft value
    roster_data['Draft Value'] = roster_data['Draft Status'].apply(getValue)
    # we convert the draft value to be a percentage
    total_value = roster_data['Draft Value'].sum()
    grouped = roster_data.groupby('Pos.')['Draft Value'].sum()
    grouped = grouped / total_value * 100
    # we pivot the index and make the index of each row the team name
    # instead of the player
    grouped = grouped.reset_index()
    grouped['Team'] = pd.Series(team, index=range(len(grouped)))
    grouped = grouped.pivot(index='Team', columns='Pos.', values='Draft Value')
    grouped.reset_index(inplace=True)
    grouped = grouped.fillna(0)
    # Renamed the columns to match your desired format
    grouped.columns.name = None  
    grouped.columns = grouped.columns.str.replace(' ', '')  # Remove spaces in column names
    # we convert back from our shortened team name to our longer one
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
    """
    we get the year value
    @param info - the draft infor for each player
    returns the year the player was drafted
    """
    year = info.split()[0]
    # our first 2 conditionals are edge cases we handle
    # from malformed data input
    if year == '22/7':
        return 22
    elif year == 'SF23':
        return 23
    else:
        return int(year)

def getValue(info):
    """
    Gets the draft value for a particular pick
    @param info - the draft info for each player
    returns the draft value invested in that player
    """
    tokens = info.split(' ')
    # we read in our pick values dataframe
    pick_value = pd.read_csv('nfl_pick_value.csv')
    # for any malformed inout, we return 0 as they are
    # undrafted free agents
    try:
        if len(tokens) == 2 or len(tokens) == 0 or tokens[0] == '22/7' or tokens[2] == '' or tokens[2] == '020R':
            return 0
        else:
            # we get the pick and return the value
            pick = int(tokens[2])
            if (pick < len(pick_value)):
                return pick_value.loc[pick, 'Value']
            else:
                return 0
    except IndexError as e:
        return 0

def web_scrape_data(year):
    """
    Takes a year and scrapes the cap data for that year
    @param year - the year of data to scrape
    returns a cleaned dataframe containing cap space info
    """
    # scrape the website using Beautiful Soup
    url = 'https://www.spotrac.com/nfl/positional/breakdown/' + year + '/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    headers = []
    data = []
    # get the column names
    for th in table.find('thead').find_all('th'):
        headers.append(th.text.strip())
    # process each row
    for row in table.find('tbody').find_all('tr'):
        row_data = []
        for td in row.find_all('td'):
            t = td.text.strip()
            # we remove the Million symbol for money rows
            if ('M' in t and 'Miami' not in t and 'Minnesota' not in t):
                # we convert the text to a float
                t = t.split(' ')[0]
                t = float(t)
                # we get the percentage of the cap spent on that position 
                # and change the total number by year as the cap space number changes 
                if (year == '2022'):
                    t = t / 208200000 * 100
                elif (year == '2021'):
                    t = t / 182500000 * 100
                elif (year == '2020'):
                    t = t / 198200000 * 100
                else:
                    t = t / 188200000 * 100
                row_data.append(t)
            else:
                row_data.append(td.text.strip())
            data.append(row_data)
    # we make sure we only got unique data
    unique_lists = list(set(tuple(x) for x in data))
    data = [list(x) for x in unique_lists]
    # we make a dataframe out of our info
    cap_data = pd.DataFrame(data, columns=headers)
    # we remove the city name from the team name
    cap_data['Team'] = cap_data['Team'].apply(lambda x: x.split()[len(x.split()) - 1])
    # we return the cap data
    cap_data = cap_data[['QB', "RB/FB", 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K/P/LS', 'Team']]
    return cap_data


if __name__ == "__main__":
    main()