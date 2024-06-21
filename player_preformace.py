import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():

    # Read data
    cap2019 = pd.read_csv("cap_data2019.csv")
    value2019 = pd.read_csv("value_data_2019.csv")
    draft2019 = pd.read_csv("draft_data_2019.csv")

    # Merge dataframes
    merged = cap2019.merge(value2019, on="Team")
    merged = merged.merge(draft2019, on="Team")

    # Filter out rows with 0 values
    merged = merged[(merged["QB_x"] != 0) & (merged["QB_y"] != 0) & (merged["QB"] != 0)]

    # Plot Quarterback Investment vs Performance 2019
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x="QB_x", y="QB_y")
    plt.title('Quarterback Investment vs Performance 2019')
    plt.xlabel('Percentage of Cap Space Invested in QB')
    plt.ylabel('QB Performance')
    plt.savefig('Quarterback Investment vs Performance 2019.png')

    # Plot Quarterback Draft Investment vs Performance 2019
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x="QB", y="QB_y")
    plt.title('Quarterback Draft Investment vs Performance 2019')
    plt.xlabel('Percentage of Draft Capital Invested in QB')
    plt.ylabel('QB Performance')
    plt.savefig('Quarterback Draft Investment vs Performance 2019.png')

    plt.show()

if __name__ == "__main__":
    main()