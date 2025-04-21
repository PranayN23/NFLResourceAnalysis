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
    cols = ['TOTAL DVOA', 'win-loss-pct', 'Net EPA']
    count = 0
    for col in cols:
        years = [2019, 2020, 2021, 2022]
        for year in years:
            value = pd.read_csv(f'value_data_{year}.csv')
            # we get the correlations of each resource dataset with winning
            # in the year it relates to
            corrs = correlations(value, year, col)
            if count == 0:
                title = f'Positional Success Correlation with DVOA ({year})'
            elif count == 1:
                title = f'Positional Success Correlation with Winning ({year})'
            else:
                title = f'Positional Success Correlation with EPA ({year})'
            # we print the corrleation data frames
            fig, ax = plt.subplots(1, figsize=(15,7))
            ax.bar(corrs.index, corrs[col])
            ax.set_title(title)
            ax.set_xlabel('Position')
            if count == 0:
                ax.set_ylabel('Correlation with DVOA')
            elif count == 1:
                ax.set_ylabel('Correlation with Winning')
            else:
                ax.set_ylabel('Correlation with DVOA')
            plt.savefig(title + '.png')
        count += 1


def correlations(data, year, col):
    """
    We get the correlation between each of our datasets 
    and the respectvie stats for that year
    @param datasets - our resources dataframes
    returns a list of correlation dataframes
    """
    team_stats = pd.read_csv(f'team_success_{year}.csv')
    team_stats['TOTAL DVOA'] = team_stats['TOTAL DVOA'].str.rstrip('%').astype('float') / 100
    team_stats = team_stats[['TEAM', col]]
    merged2022 = data.merge(team_stats, left_on='Team', right_on='TEAM')
    numeric_columns = merged2022.select_dtypes(include=['number'])
    correlation = numeric_columns.corr()[[col]]
    correlation = correlation.drop(['Unnamed: 0', col], axis=0)
    return correlation


if __name__ == "__main__":
    main()