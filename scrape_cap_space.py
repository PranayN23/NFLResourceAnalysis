"""
Pranay Nandkeolyar

This module web scrapes data from different web sites
regarding nfl team resource stats, such as draft capital spent 
per position and cap space spent per position. 
This required the use of many libraries including urllib (web scraping), regex (re),
pandas (CSV processing), numpy (data manipulation)
BeautifulSoup (html parsing), requests (web scraping). 
"""
import math
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
    
    cap_data2010 = web_scrape_data('2010')
    cap_data2010.to_csv('cap_data2010.csv')
    cap_data2011 = web_scrape_data('2011')
    cap_data2011.to_csv('cap_data2011.csv')
    cap_data2012 = web_scrape_data('2012')
    cap_data2012.to_csv('cap_data2012.csv')
    cap_data2013 = web_scrape_data('2013')
    cap_data2013.to_csv('cap_data2013.csv')
    cap_data2014 = web_scrape_data('2014')
    cap_data2014.to_csv('cap_data2014.csv')
    cap_data2015 = web_scrape_data('2015')
    cap_data2015.to_csv('cap_data2015.csv')
    cap_data2016 = web_scrape_data('2016')
    cap_data2016.to_csv('cap_data2016.csv')
    cap_data2017 = web_scrape_data('2017')
    cap_data2017.to_csv('cap_data2017.csv')
    cap_data2023 = web_scrape_data('2023')
    cap_data2023.to_csv('cap_data2023.csv')
    cap_data2024 = web_scrape_data('2024')
    cap_data2024.to_csv('cap_data2024.csv')

    """ cap_data2022 = web_scrape_data('2022')
    cap_data2022.to_csv('cap_data2022.csv')
    cap_data2021 = web_scrape_data('2021')
    cap_data2021.to_csv('cap_data2021.csv')
    cap_data2020 = web_scrape_data('2020')
    cap_data2020.to_csv('cap_data2020.csv')
    cap_data2019 = web_scrape_data('2019')
    cap_data2019.to_csv('cap_data2019.csv')
    cap_data2018 = web_scrape_data('2018')
    cap_data2018.to_csv('cap_data2018.csv')
     """
"""     s = set()
    draft2022, value2022 = web_scrape_rosters(2022, s)
    draft2022.to_csv('draft_data_2022.csv')
    value2022.to_csv('value_data_2022.csv')
    draft2021, value2021 = web_scrape_rosters(2021, s)
    draft2021.to_csv('draft_data_2021.csv')
    value2021.to_csv('value_data_2021.csv')
    draft2020, value2020 = web_scrape_rosters(2020, s)
    draft2020.to_csv('draft_data_2020.csv')
    value2020.to_csv('value_data_2020.csv')
    draft2019, value2019 = web_scrape_rosters(2019, s)
    draft2019.to_csv('draft_data_2019.csv')
    value2019.to_csv('value_data_2019.csv')
    draft2018, value2018 = web_scrape_rosters(2018, s)
    draft2018.to_csv('draft_data_2018.csv')
    value2018.to_csv('value_data_2018.csv') """
    
def web_scrape_rosters(year, s):
    """
    Here we webscrape every team's roster
    for their draft data
    @param last_4 - a boolean representing whether or not we want the last 4
    years only or all draft data
    returns a dataframe of each team's releative draft resources spent
    on each position
    """
    # we make a list of every team's shorthand
    nfl_teams = [ "Cardinals", "Falcons", "Ravens", "Bills", "Panthers", "Bears",
    "Bengals", "Browns", "Cowboys", "Broncos", "Lions", "Packers", "Texans", "Colts",
    "Jaguars", "Chiefs", "Raiders", "Chargers", "Rams", "Dolphins", "Vikings", 
    "Patriots", "Saints", "Giants", "Jets", "Eagles", "Steelers", "49ers",
    "Seahawks", "Buccaneers", "Titans", "Commanders"]
    # we loop through every team and scrape their roster page
    draft = []
    value = []
    for team in nfl_teams:
        draftDF, valueDF = add_roster(team, year)
        draft.append(draftDF)
        value.append(valueDF)
    return convert_dataframes(draft, s), convert_dataframes(value, s)
    
    
import pandas as pd
import numpy as np

def convert_dataframes(dataframes, s):
    """
    Takes a list of each team's roster data and 
    returns it as one dataframe with the corrected positional information
    returns a dataframe of each team's relative draft resources spent
    on each position.
    """
    draft_data = pd.concat(dataframes, ignore_index=True)
    s.update(draft_data.columns.tolist())
    # Fill empty values
    draft_data.replace('', np.nan, inplace=True)
    draft_data.fillna(0, inplace=True)
    
    # Rework each position's definition to match our desired output
    draft_data['RB/FB'] = draft_data.get("RB", 0) + draft_data.get('FB', 0)
    draft_data["OL"] = (draft_data.get('T', 0) + draft_data.get('OL', 0) 
                        + draft_data.get('G', 0) + draft_data.get('LG/C', 0)
                        + draft_data.get('RT/LT', 0) + draft_data.get('RT', 0)
                        + draft_data.get('RG/C', 0) + draft_data.get('C', 0)
                        + draft_data.get('LG', 0) + draft_data.get('C/LG', 0)
                        + draft_data.get('RG', 0) + draft_data.get('OT', 0)
                        + draft_data.get('LT/RT', 0) + draft_data.get('LT', 0))
    draft_data['DL'] = (draft_data.get('DT', 0) + draft_data.get('DE', 0) 
                        + draft_data.get('RDT/LDT', 0) + draft_data.get('NT', 0)
                        + draft_data.get('RDE', 0) + draft_data.get('LDT', 0)
                        + draft_data.get('LDE', 0) + draft_data.get('LDE/RDE', 0)
                        + draft_data.get('DL', 0) + draft_data.get('RDT', 0)
                        + draft_data.get('RDE/LDE', 0) + draft_data.get('LDT/RDT', 0) + + draft_data.get('OLB', 0) )
    draft_data['LB'] = (draft_data.get('LB', 0)
                        + draft_data.get('LOLB', 0) + draft_data.get('LLB', 0) 
                        + draft_data.get('LILB', 0) + draft_data.get('RLB', 0)
                        + draft_data.get('LILB/RILB', 0) + draft_data.get('MLB/RLB', 0)
                        + draft_data.get('MLB', 0) + draft_data.get('ROLB/RILB', 0)
                        + draft_data.get('RILB', 0) + draft_data.get('RLB/LLB', 0))
    draft_data['DB'] = (draft_data.get('S', 0) + draft_data.get('CB', 0) 
                        + draft_data.get('SS/FS', 0) + draft_data.get('LCB/RCB', 0)
                        + draft_data.get('RCB/LCB', 0) + draft_data.get('FS', 0)
                        + draft_data.get('LCB', 0) + draft_data.get('RCB', 0)
                        + draft_data.get('DB', 0) + draft_data.get('SS', 0)
                        + draft_data.get('FS/SS', 0))
    draft_data["K/P/LS"] = draft_data.get('K', 0) + draft_data.get('P', 0) + draft_data.get('LS', 0)
    
    # Remove excess columns
    draft_data = draft_data[['Team', 'QB', 'RB/FB', 'WR', "TE", "OL", "DL", "LB", "DB", "K/P/LS"]]
    
    return draft_data


def add_roster(team, year):
    path = 'rosters/' + str(year) + '/' + team + '.csv'
    return clean_roster_data(pd.read_csv(path), team, year)

def clean_roster_data(roster_data, team, year):
    """
    Cleans a specific team's roster
    @param roster_data - the team's scraped roster info 
    @param team - the name of the team 
    @param last_4 - a boolean representing whether or not we want the last 4
    years only or all draft data
    returns a cleaned DataFrame of a team's draft info
    """
    # Remove rows with missing values in the DataFrame
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    roster_data = roster_data.copy()
    roster_data.replace('', np.nan, inplace=True)
    roster_data.fillna(0, inplace=True)
    # Select specific columns from the DataFrame
    roster_data = roster_data[['Player', 'Pos', 'AV', 'Drafted (tm/rnd/yr)']]
    # Extract the draft year for each player using .loc
    roster_data['Draft Year'] = roster_data['Drafted (tm/rnd/yr)'].apply(getYear)
    # Extract the team each player was drafted for using .loc
    roster_data['Orig. Team'] = roster_data['Drafted (tm/rnd/yr)'].apply(getOGTeam, args=(team,))
    # Convert 'Draft Year' to numeric and filter based on it
    # Filter the DataFrame based on 'Draft Year' and 'Orig. Team'
    appx = roster_data.copy()
    # roster_data = roster_data[(roster_data['Draft Year'] >= (year - 3)) & (roster_data['Draft Year'] <= year)]
    # roster_data = roster_data[roster_data['Orig. Team'] == team]
    # Calculate the 'Draft Value' for each player
    roster_data['Draft Value'] = roster_data['Drafted (tm/rnd/yr)'].apply(getValue, year = year)
    # Calculate 'draft' and 'appx' using the 'getData' function
    draft = getData(roster_data, 'Draft Value', team)
    appx = getData(appx, 'AV', team)
    return draft, appx


def getData(data, column_name, team):
    if column_name == 'Draft Value':
        total_value = data[column_name].sum()
        grouped = data.groupby('Pos')[column_name].sum()
        # grouped = grouped / total_value * 100
    else:
        grouped = data.groupby('Pos')[column_name].sum()
    # we pivot the index and make the index of each row the team name
    # instead of the player
    grouped = grouped.reset_index()
    grouped['Team'] = pd.Series(team, index=range(len(grouped)))
    grouped = grouped.pivot(index='Team', columns='Pos', values=column_name)
    grouped.reset_index(inplace=True)
    grouped = grouped.fillna(0)
    # Renamed the columns to match your desired format
    grouped.columns.name = None
    grouped.columns = grouped.columns.str.replace(' ', '')
    return grouped

def getOGTeam(info, team):
    """
    we get the year value
    @param info - the draft infor for each player
    returns the year the player was drafted
    """
    if isinstance(info, str) and len(info.strip()) > 0:
        if ('Washington' in info):
            return 'Commanders'
        else:
            l = info.split()
            index = l.index('/')
            return l[index - 1]
    else:
        return team


def getYear(info):
    """
    we get the year value
    @param info - the draft infor for each player
    returns the year the player was drafted
    """
    if isinstance(info, str) and len(info.strip()) > 0:
        l = info.split()
        try:
            r =  int(l[len(l) - 1])
        except ValueError:
            r = int(l[len(l) - 2])
        return r
    else:
        return 0

def getValue(info, year):
    """
    Gets the draft value for a particular pick
    @param info - the draft info for each player
    returns the draft value invested in that player
    """
    # we read in our pick values dataframe
    pick_value = pd.read_csv('nfl_pick_value.csv')
    # for any malformed inout, we return 0 as they are
    # undrafted free agents
    try:
        tokens = info.split(' ')
        index = tokens.index('/')
        pick_str = tokens[index + 3]
        pick_str = pick_str[:len(pick_str) - 2]
        pick = int(pick_str)
        year_diff = year - int(tokens[-1])
        k = 1 / 4
        if (pick < len(pick_value)):
            return pick_value.loc[pick - 1, 'Value'] * np.exp(-k * year_diff)
        else:
            return 0
    except Exception as e:
        return 0

def web_scrape_data(year):
    """
    Takes a year and scrapes the cap data for that year
    @param year - the year of data to scrape
    returns a cleaned dataframe containing cap space info
    """
    # scrape the website using Beautiful Soup
    url = 'https://www.spotrac.com/nfl/position/_/year/' + year + '/'
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
            if ('M' in t and 'MIA' not in t and 'MIN' not in t):
                # we convert the text to a float
                t = t.replace('$', '').replace('M', '')
    
                # Convert to float and multiply by 1,000,000
                t = float(t) * 1_000_000
                # we get the percentage of the cap spent on that position 
                # and change the total number by year as the cap space number changes 
                if (year == '2022'):
                    t = t / 208200000 * 100
                elif (year == '2021'):
                    t = t / 182500000 * 100
                elif (year == '2020'):
                    t = t / 198200000 * 100
                elif (year == '2019'):
                    t = t / 188200000 * 100
                elif (year == '2018'):
                    t = t / 177200000 * 100
                elif (year == '2017'):
                    t = t / 167000000 * 100
                elif (year == '2016'):
                    t = t / 155270000 * 100
                elif (year == '2015'):
                    t = t / 143280000 * 100
                elif (year == '2014'):
                    t = t / 133000000 * 100
                elif (year == '2013'):
                    t = t / 123600000 * 100
                elif (year == '2012'):
                    t = t / 120600000 * 100
                elif (year == '2011'):
                    t = t / 120375000 * 100
                elif (year == '2010'):
                    t = t / 123000000 * 100  
                elif (year == '2023'):
                    t = t / 224800000 * 100
                elif (year == '2024'):
                    t = t / 255400000 * 100    
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
    cap_data = cap_data[['QB', "RB", 'WR', 'TE', 'OL', 'DL', 'LB', 'SEC', 'K', 'P', 'LS', 'Team']]
    cap_data = cap_data.replace('-', 0)
    cap_data['K/P/LS'] = cap_data['K'] + cap_data['P'] + cap_data['LS']
    cap_data['RB/FB'] = cap_data['RB']
    cap_data['DB'] = cap_data['SEC']
    cap_data = cap_data[['QB', "RB/FB", 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K/P/LS', 'Team']]
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
    cap_data['Team'] = cap_data['Team'].map(team_mapping)
    return cap_data


if __name__ == "__main__":
    main()