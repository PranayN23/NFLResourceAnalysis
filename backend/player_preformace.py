import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

def sanitize_filename(filename):
    """Sanitize filenames by replacing problematic characters."""
    return re.sub(r'[^\w\-_\. ]', '_', filename)


def main():
    years = [2019, 2020, 2021, 2022]
    for year in years:
    # Read data
        cap2019 = pd.read_csv(f'cap_data{year}.csv')
        value2019 = pd.read_csv(f"value_data_{year}.csv")
        draft2019 = pd.read_csv(f"draft_data_{year}.csv")

        # Merge dataframes
        merged = cap2019.merge(value2019, on="Team")
        merged = merged.merge(draft2019, on="Team")

        ## List of positions to analyze
        positions = ["QB", "RB/FB", "WR", "TE", "OL", "DL", "LB", "DB", "K/P/LS"]

        for position in positions:
            # Filter out rows with 0 values for the current position
            filtered = merged[(merged[f"{position}_x"] != 0) & (merged[f"{position}_y"] != 0) & (merged[position] != 0)]
            
            # Plot Cap Space Investment vs Performance
            plt.figure(figsize=(8, 6))
            sns.regplot(data=filtered, x=f"{position}_x", y=f"{position}_y")
            plt.title(f'{position} Investment vs Performance {year}')
            plt.xlabel(f'Percentage of Cap Space Invested in {position}')
            plt.ylabel(f'{position} Performance')
            if position == 'RB/FB':
                plt.savefig(f'RB Investment vs Performance {year}.png')
            elif position == "K/P/LS":
                plt.savefig(f'ST Investment vs Performance {year}.png')
            else:
                plt.savefig(f'{position} Investment vs Performance {year}.png')
            plt.close()
            # Plot Draft Investment vs Performance
            plt.figure(figsize=(8, 6))
            sns.regplot(data=filtered, x=position, y=f"{position}_y")
            plt.title(f'{position} Draft Investment vs Performance {year}')
            plt.xlabel(f'Percentage of Draft Capital Invested in {position}')
            plt.ylabel(f'{position} Performance')
            if position == 'RB/FB':
                plt.savefig(f'RB Draft Investment vs Performance {year}.png')
            elif position == "K/P/LS":
                plt.savefig(f'ST Draft Investment vs Performance {year}.png')
            else:
                plt.savefig(f'{position} Draft Investment vs Performance {year}.png')
            plt.close()

if __name__ == "__main__":
    main()