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
    cap_data2022 = pd.read_csv('cap_data2022.csv')
    cap_data2021 = pd.read_csv('cap_data2021.csv')
    draft_data = pd.read_csv('draft_data.csv')
    draft_data_last_4 = pd.read_csv('draft_data_last_4.csv')
    datasets = [cap_data2022, cap_data2021, draft_data, draft_data_last_4]
    # we get the correlations of each resource dataset with winning
    # in the year it relates to
    corrs = correlations(datasets)
    ylabels = ['%% of Cap Space Spent', '%% of Cap Space Spent)', '%% of draft capital spent', '%% of draft capital spent']
    titles = ['Percentage of Cap Space Spent vs Winning Percentage (2022)',
              'Percentage of Cap Space Spent vs Winning Percentage (2021)',
              'Percentage of Draft Capital Spent vs Winning Percentage',
              'Percentage of Draft Capital Spent vs Winning Percentage Over The Last 4 Years']
    # we print the corrleation data frames
    for i in range(len(corrs)):
        fig, ax = plt.subplots(1, figsize=(15,7))
        ax.bar(corrs[i].index, corrs[i]['win_pct'])
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
            team_stats = pd.read_csv('nfl2021.csv')
        else:
            team_stats = pd.read_csv('nfl_stats.csv')
        count += 1
        team_stats = team_stats[['team', 'win_pct']]
        merged2022 = data.merge(team_stats, left_on='Team', right_on='team')
        numeric_columns = merged2022.select_dtypes(include=['number'])
        correlation = numeric_columns.corr()[['win_pct']]
        correlation = correlation.drop(['Unnamed: 0', 'win_pct'], axis=0)
        correlations.append(correlation)
    return correlations


if __name__ == "__main__":
    main()