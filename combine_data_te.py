#%%
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
#%%
!pip install tensorflow==2.18.0
!pip install scikit-learn
!pip install 'urllib3<2'
#%%
data1 = pd.read_csv('TEPFF.csv')
combined_df = pd.DataFrame(data1)

# Filter for rows where the 'year' is between 2010 and 2020 (inclusive)
combined_df = combined_df[(combined_df['Year'] >= 2019) & (combined_df['Year'] <= 2022)]

# Display the filtered DataFrame
print(combined_df['Year'])
#%%
import warnings
warnings.filterwarnings("ignore")

# Merge all AV values
#%%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
for year in range(2019, 2023):
    # Read the CSV file for the current year
    data = pd.read_csv(f'value_data_{year}.csv')
    
    # Create a DataFrame with the selected columns
    av = data[['Team', 'TE']]
    
    # Add the "Year" column
    av['Year'] = year
    
    # Merge with combined_df on 'Team' and 'Year' columns
    combined_df = pd.merge(combined_df, av, on=['Team', 'Year'], how='left', suffixes=('', f'_{year}'))

print(combined_df)


combined_df['Current_AV'] = combined_df[['TE', 'TE_2020', 'TE_2021', 'TE_2022']].bfill(axis=1).iloc[:, 0]


combined_df = combined_df.drop(columns=['TE', 'TE_2020', 'TE_2021', 'TE_2022'])


print(combined_df)
#%%
combined_df = combined_df.drop(columns= 'Unnamed: 0')

print(combined_df.columns)

#%%
for year in range(2018, 2022):
    # Read the CSV file for the current year
    data = pd.read_csv(f'value_data_{year}.csv')
    
    # Create a DataFrame with the selected columns
    av = data[['Team', 'TE']]
    
    # Add the "Year" column
    av['Year'] = year + 1
    
    # Merge with combined_df on 'Team' and 'Year' columns
    combined_df = pd.merge(combined_df, av, on=['Team', 'Year'], how='left', suffixes=('', f'_{year}'))

print(combined_df)


combined_df['Previous_AV'] = combined_df[['TE', 'TE_2019', 'TE_2020', 'TE_2021']].bfill(axis=1).iloc[:, 0]


combined_df = combined_df.drop(columns=['TE', 'TE_2019', 'TE_2020', 'TE_2021'])


print(combined_df)
#%%
# Team Success Additions
#%%
data = pd.read_csv('data.csv')
te_data = pd.read_csv('TEPFF.csv')
filtered_data = data[data['Position'] == 'TE']
print(filtered_data.columns)
print(te_data.columns)
#%%
combined_df = pd.merge(filtered_data, te_data, on=['Team', 'Year'], how='left')
combined_df = combined_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
print (combined_df)
#%%
combined_df.to_csv("Combined_TE.csv")
#%%
print(combined_df.head)
#%%
