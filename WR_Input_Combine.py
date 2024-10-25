import pandas as pd

# Load the CSV files
wrPFF = pd.read_csv('wrPFF.csv', index_col=0)
data = pd.read_csv('data.csv', index_col=0)

# Merge on 'Position', 'Team', and 'Year' columns
merged_df = pd.merge(wrPFF, data, on=['Position', 'Team', 'Year'], how='inner')  # Adjust 'how' as needed
# Display or save the merged DataFrame
merged_df = merged_df.reset_index()
print(merged_df.head())
merged_df.to_csv('WR_data_updated.csv', index=False)
