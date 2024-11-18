import pandas as pd

# Load the CSV files
df = pd.read_csv('data.csv')
df = df[df['Position'] == "WR"]

data = pd.read_csv('wrPFF.csv')
data = data.rename(columns={'weighted_avg_grades_offense': 'Current_PFF', 'position' : 'Position'})

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.loc[:, df.columns != '']
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.loc[:, data.columns != '']
df = df.merge(data, on=['Team', 'Year', 'Current_PFF', 'Position'])
df.to_csv("Combined_WR.csv")
