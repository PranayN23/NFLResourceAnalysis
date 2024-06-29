import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
#from scipy.stats import linregress


def main():
    #Read data
    cap_2021 = pd.read_csv("cap_data2021.csv")
    value_2021 = pd.read_csv("value_data_2021.csv")

    #Positions
    positions = ["QB", "RB/FB", "WR", "TE", "OL", "DL", "LB", "DB", "K/P/LS"]

    #Generate graph by position
    for position in positions:

        #Take out all 0 values for respective position
        clean_cap = cap_2021[position]
        clean_cap = clean_cap[clean_cap != 0]

        clean_value = value_2021[position]
        clean_value = clean_value[clean_value != 0]

        #Save and Plot graphs for each position
        plt.figure(figsize=(8, 6))
        sns.regplot(x = clean_cap, y = clean_value)
        plt.title(f'{position} Investment vs Performance 2021')
        plt.xlabel(f'Percentage of Cap Space Invested in {position}')
        plt.ylabel(f'{position} Performance')
        if position == 'RB/FB':
            plt.savefig('RB Investment vs Performance 2021.png')
        elif position == "K/P/LS":
             plt.savefig('ST Investment vs Performance 2021.png')
        else:
            plt.savefig(f'{position} Investment vs Performance 2021.png')
        plt.close()

        #print(clean_cap.head())
        #print(clean_value.head())
if __name__ == "__main__":
    main()