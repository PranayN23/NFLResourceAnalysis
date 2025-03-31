#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
#%%
te_df = pd.read_csv('Combined_TE.csv')
#%%
print(te_df.head())
#%%
def check_correlation(df, metric):
    pd.set_option('display.max_rows', None)

    features = [col for col in df.columns if
                col != metric and col != 'weighted_avg_franchise_id' and col != 'weighted_avg_spikes' and col != 'Team' and col != 'Year' and col != 'Position']
    prev = [x for x in features if 'Previous' in x]
    prev.append('Current_' + metric)
    curr = [x for x in features if 'Previous' not in x]
    df['Total DVOA'] = df['Total DVOA'].astype(str).str.rstrip('%').astype(float) / 100.0
    l = [curr, prev]
    for item in l:
        # Filter only the relevant columns
        corr_df = df[item]

        # Compute the correlation matrix
        corr_matrix = corr_df.corr()
        target_corr = corr_matrix[['Current_' + metric]].drop('Current_' + metric).sort_values(by='Current_' + metric,
                                                                                               ascending=False)  # Select correlation with 'metric' and exclude itself

        # Print the correlation matrix
        print(f'Correlation Matrix for {metric}:\n', target_corr, '\n')
        pd.reset_option('display.max_rows')

#%%
te_df = te_df.drop(columns=['Unnamed: 0', 'Team', 'Year', 'Position', 'position'])
#%%
check_correlation(te_df, 'PFF')
#%%
te_df = pd.read_csv('Combined_TE.csv')
#%%
te_df = te_df.sort_values(by=['Team', 'Year'])
te_df = te_df.drop(columns=['Unnamed: 0'])

# Display the sorted data
print(te_df.head())

#%%
team_data = te_df.groupby('Team')
print(team_data.head)
#%%
import numpy as np

sequences = []
targets = []

# Iterate over each team and its respective data
for team, group in team_data:
    # Ensure the team has at least 4 years of data
    if len(group) >= 4:
        print(f"Processing team: {team}, data length: {len(group)}")  # Debugging: check length of data for each team
        
        # Iterate through the data to create sequences for 3 years
        for i in range(len(group) - 3):
            # Select the relevant columns for the sequence
            sequence = group.iloc[i:i+3][['Previous_yprr', 'Previous_grades_pass_route', 'Previous_PFF', 
            'Previous_grades_offense', 'Previous_yards', 'Previous_first_downs']]  # Add more columns if needed
            
            # The target is the next year's Current_PFF
            target = group.iloc[i+3]['Current_PFF']  
            
            sequences.append(sequence.values)  # Add the sequence to the list
            targets.append(target)  # Add the target to the list

# Convert lists to numpy arrays
X = np.array(sequences)
y = np.array(targets)



print(f"Total sequences generated: {len(sequences)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
#%%
# Check shape before reshaping
print(X)
print(X.shape)

# Reshape X to (samples, timesteps, features)
X = X.reshape(X.shape[0], 3, -1)  # 3 years per sequence, features will be inferred


print('hi')  # Should now be (samples, 3, features)
print(X)