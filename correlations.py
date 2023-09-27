""" 

team_stats = pd.read_csv('nfl_stats.csv')
    temp = team_stats[['team', 'win_pct']]
    merged2022 = cap_data.merge(temp, left_on='Team', right_on='team')
    numeric_columns = merged2022.select_dtypes(include=['number'])
    correlations = numeric_columns.corr()[['win_pct']]
    return(correlations, cap_data, team_stats)
"""