import pandas as pd

def main():
    result = collect_data()
    result.to_csv('data.csv')



def collect_data():

    # Sample data loading
    # Replace these with your actual data loading mechanism
    cap_space_dfs = {year: pd.read_csv(f'cap_data{year}.csv') for year in range(2018, 2023)}
    draft_data_dfs = {year: pd.read_csv(f'draft_data_{year}.csv') for year in range(2018, 2023)}
    av_data_dfs = {year: pd.read_csv(f'value_data_{year}.csv') for year in range(2018, 2023)}

    # Reshape dataframes
    positions = ['QB', 'RB/FB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K/P/LS']
    reshaped_cap_space = pd.concat([reshape_df(df, year, positions) for year, df in cap_space_dfs.items()])
    reshaped_draft_data = pd.concat([reshape_df(df, year, positions) for year, df in draft_data_dfs.items()])

    reshaped_av_data = pd.concat([reshape_df(df, year, positions) for year, df in av_data_dfs.items()])
    # Combine dataframes
    combined_df = pd.merge(reshaped_cap_space, reshaped_draft_data, on=['Team', 'Year', 'Position'], suffixes=('_cap_space', '_draft_data'))
    combined_df = pd.merge(combined_df, reshaped_av_data, on=['Team', 'Year', 'Position'])
    combined_df.rename(columns={'Value': 'Current_AV'}, inplace=True)
    # Compute previous year AV
    combined_df['Previous_AV'] = combined_df.groupby(['Team', 'Position'])['Current_AV'].shift(1)
    team_dfs = {year: pd.read_csv(f'team_success_{year}.csv') for year in range(2019, 2023)}
    reshaped_team = pd.concat([fix_df(df, year) for year, df in team_dfs.items()])
    combined_df = pd.merge(combined_df, reshaped_team, on=['Team', 'Year'])
    # Filter for years past 2019
    final_df = combined_df[combined_df['Year'] >= 2019]
    final_df = final_df[['Team', 'Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Current_AV', 'Total DVOA', 'win-loss-pct', 'Net EPA']]
    return final_df

def fix_df(df, year):
    df['Year'] = year
    df = df.rename(columns = {'TEAM' : 'Team', 'TOTAL DVOA' : 'Total DVOA'})
    return df

def reshape_df(df, year, position_cols):
    df = df.melt(id_vars=['Team'], value_vars=position_cols, var_name='Position', value_name='Value')
    df['Year'] = year
    return df


if __name__ == "__main__":
    main()

