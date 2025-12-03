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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


def main():
    years = [2019, 2020, 2021, 2022]
    for year in years:
        dvoa = pd.read_csv("DVOA " + str(year) + ".csv")
        epa = pd.read_csv("EPA " + str(year) + ".csv")
        epa = epa[["Abbr", "Net"]]
        dvoa = dvoa[["TEAM", "TOTAL DVOA", "W-L"]]
        dvoa['win-loss-pct'] = dvoa['W-L'].apply(make_percent)
        merged = dvoa.merge(epa, left_on='TEAM', right_on='Abbr')
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
    'WAS': 'Commanders'
}
        merged['TEAM'] = merged['TEAM'].map(team_mapping)
        merged = merged[['TEAM', 'TOTAL DVOA', 'win-loss-pct', 'Net']]
        merged = merged.rename(columns={'Net': 'Net EPA'})
        merged.to_csv("team_success_" + str(year) + ".csv")
 

def make_percent(record):
    results = record.split('-')
    if len(results) > 2:
        return int(results[0]) / ((int(results[0])) + int(results[1]) + int(results[2]))
    else:
        return int(results[0]) / ((int(results[0])) + int(results[1]))

if __name__ == "__main__":
    main()