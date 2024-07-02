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
    value2022 = pd.read_csv('value_data_2022.csv')
    value2021 = pd.read_csv('value_data_2021.csv')
    datasets = [value2022, value2021]
    # we get the correlations of each resource dataset with winning
    # in the year it relates to
    corrs = correlations(datasets)
    titles = ['Positional Success Correlation with Winning Percentage (2022)',
              'Positional Success Correlation with Winning Percentage (2021)']
    # we print the corrleation data frames
    for i in range(len(corrs)):
        fig, ax = plt.subplots(1, figsize=(15,7))
        ax.bar(corrs[i].index, corrs[i]['win-loss-pct'])
        ax.set_title(titles[i])
        ax.set_xlabel('Position')
        ax.set_ylabel('Correlation with Winning')
        plt.savefig(titles[i] + '.png')


def correlations(datasets):
    """
    We get the correlation between each of our datasets 
    and the respectvie stats for that year
    @param datasets - our resources dataframes
    returns a list of correlation dataframes
    """
    count = 0
    correlations = []
    for data in datasets:
        if (count == 1):
            team_stats = pd.read_csv('team_success_2021.csv')
            print(team_stats.columns)
        else:
            team_stats = pd.read_csv('team_success_2022.csv')
        count += 1
        team_stats = team_stats[['TEAM', 'win-loss-pct']]
        merged2022 = data.merge(team_stats, left_on='Team', right_on='TEAM')
        numeric_columns = merged2022.select_dtypes(include=['number'])
        correlation = numeric_columns.corr()[['win-loss-pct']]
        correlation = correlation.drop(['Unnamed: 0', 'win-loss-pct'], axis=0)
        correlations.append(correlation)
    return correlations


if __name__ == "__main__":
    main()