"""
Pranay Nandkeolyar

This module calculates the correlations with winning
given different resource (cap space and draft pick) data.
This required the use of many libraries including matplotlib (plotting), regex (re),
pandas (CSV processing), numpy (data manipulation)
Sea Born (plotting), scipy (statistical analysis). 
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
 
def main():
    """
    We get our data and call our correlations method
    """
    years = [2019, 2020, 2021, 2022]
    for year in years:
        value = pd.read_csv(f'value_data_{year}.csv')
        # we get the correlations of each resource dataset with winning
        # in the year it relates to
        corrs = correlations(value, year)
        title = f'Positional Success Correlation with Winning Percentage ({year})'
        # we print the corrleation data frames
        fig, ax = plt.subplots(1, figsize=(15,7))
        ax.bar(corrs.index, corrs['win-loss-pct'])
        ax.set_title(title)
        ax.set_xlabel('Position')
        ax.set_ylabel('Correlation with Winning')
        plt.savefig(title + '.png')


def correlations(data, year):
    """
    We get the correlation between each of our datasets 
    and the respectvie stats for that year
    @param datasets - our resources dataframes
    returns a list of correlation dataframes
    """
    team_stats = pd.read_csv(f'team_success_{year}.csv')
    team_stats = team_stats[['TEAM', 'win-loss-pct']]
    merged2022 = data.merge(team_stats, left_on='Team', right_on='TEAM')
    numeric_columns = merged2022.select_dtypes(include=['number'])
    correlation = numeric_columns.corr()[['win-loss-pct']]
    correlation = correlation.drop(['Unnamed: 0', 'win-loss-pct'], axis=0)
    return correlation


if __name__ == "__main__":
    main()