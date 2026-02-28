import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def calculate_weighted_averages_by_team_year(df):
    """
    Calculate weighted averages for all LB stats per team per year, weighted by snap_counts_defense
    Groups by: Team, Position (LB), Year
    Sums: Cap_Space, adjusted_value
    Keeps: Win %, Net EPA (team-level)
    Weighted averages: All key stats
    """
    print("\nCalculating weighted averages by team, position, and year...")

    # Define stats to calculate weighted averages for
    stats_to_weight = [
        # Performance grades
        'grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense',
        'grades_run_defense', 'grades_tackle', 'missed_tackle_rate',

        # Counting stats
        'tackles', 'assists', 'sacks', 'stops', 'tackles_for_loss',
        'total_pressures', 'hits', 'hurries', 'interceptions', 'pass_break_ups',
        'forced_fumbles', 'fumble_recoveries', 'missed_tackles',

        # Situational snap counts
        'snap_counts_box', 'snap_counts_offball', 'snap_counts_pass_rush',
        'snap_counts_run_defense',

        # Additional metrics
        'penalties', 'targets', 'declined_penalties', 'age', 'player_game_count'
    ]

    # Group by team, position, and year
    team_pos_year_data = []

    for team in sorted(df['Team'].unique()):
        for year in sorted(df['Year'].unique()):
            for position in sorted(df['position'].unique()):
                team_pos_year_df = df[(df['Team'] == team) &
                                      (df['Year'] == year) &
                                      (df['position'] == position)].copy()

                # Filter out players with no snap counts
                team_pos_year_df = team_pos_year_df[team_pos_year_df['snap_counts_defense'].notna() &
                                                    (team_pos_year_df['snap_counts_defense'] > 0)]

                if len(team_pos_year_df) == 0:
                    continue

                print(f"  {team} {position} {year}: {len(team_pos_year_df)} players, total snaps: {team_pos_year_df['snap_counts_defense'].sum():.0f}")

                # Initialize result dictionary
                result = {
                    'Team': team,
                    'position': position,
                    'Year': year
                }

                # Total snap counts for this team-position-year
                total_snaps = team_pos_year_df['snap_counts_defense'].sum()
                result['total_snap_counts_defense'] = total_snaps
                result['total_players'] = len(team_pos_year_df)

                # SUM Cap_Space and adjusted_value
                result['sum_Cap_Space'] = team_pos_year_df['Cap_Space'].sum() if 'Cap_Space' in team_pos_year_df.columns else None
                result['sum_adjusted_value'] = team_pos_year_df['adjusted_value'].sum() if 'adjusted_value' in team_pos_year_df.columns else None

                # Keep team-level stats as-is (take first value since they're same for all players on team)
                result['Win_Percent'] = team_pos_year_df['Win %'].iloc[0] if 'Win %' in team_pos_year_df.columns and len(team_pos_year_df) > 0 else None
                result['Net_EPA'] = team_pos_year_df['Net EPA'].iloc[0] if 'Net EPA' in team_pos_year_df.columns and len(team_pos_year_df) > 0 else None

                # Calculate weighted average for each stat
                for stat in stats_to_weight:
                    if stat in team_pos_year_df.columns:
                        # Get valid data (non-null values)
                        valid_data = team_pos_year_df[[stat, 'snap_counts_defense']].dropna()

                        if len(valid_data) > 0:
                            # Weighted average = sum(stat * weight) / sum(weight)
                            weighted_sum = (valid_data[stat] * valid_data['snap_counts_defense']).sum()
                            weight_sum = valid_data['snap_counts_defense'].sum()

                            if weight_sum > 0:
                                result[f'weighted_avg_{stat}'] = weighted_sum / weight_sum
                            else:
                                result[f'weighted_avg_{stat}'] = None
                        else:
                            result[f'weighted_avg_{stat}'] = None
                    else:
                        result[f'weighted_avg_{stat}'] = None

                team_pos_year_data.append(result)

    # Convert to DataFrame
    result_df = pd.DataFrame(team_pos_year_data)

    print(f"\nCreated weighted averages for {len(result_df)} team-position-year combinations")

    # Add lagged features (previous year's data) for LSTM
    result_df = add_lagged_features(result_df, stats_to_weight)

    return result_df

def add_lagged_features(df, stats_to_weight):
    """
    Add lagged features (previous year's weighted averages) for LSTM training
    For each team-position combination, create prev_weighted_avg_* columns
    """
    print("\nAdding lagged features (previous year's data)...")

    # Sort by Team, position, and Year
    df = df.sort_values(['Team', 'position', 'Year']).reset_index(drop=True)

    # Create lagged columns for each weighted average stat
    lagged_columns = []
    for stat in stats_to_weight:
        col_name = f'weighted_avg_{stat}'
        prev_col_name = f'prev_weighted_avg_{stat}'

        if col_name in df.columns:
            # Group by Team and position, then shift by 1 year
            df[prev_col_name] = df.groupby(['Team', 'position'])[col_name].shift(1)
            lagged_columns.append(prev_col_name)

    # Also lag the summary stats
    df['prev_total_snap_counts_defense'] = df.groupby(['Team', 'position'])['total_snap_counts_defense'].shift(1)
    df['prev_total_players'] = df.groupby(['Team', 'position'])['total_players'].shift(1)
    df['prev_sum_Cap_Space'] = df.groupby(['Team', 'position'])['sum_Cap_Space'].shift(1)
    df['prev_sum_adjusted_value'] = df.groupby(['Team', 'position'])['sum_adjusted_value'].shift(1)

    # Count how many rows have complete lagged data
    complete_rows = df[lagged_columns].notna().all(axis=1).sum()
    print(f"  Added {len(lagged_columns)} lagged feature columns")
    print(f"  {complete_rows} rows have complete previous year data")
    print(f"  {len(df) - complete_rows} rows missing previous year data (first year for team-position)")

    return df

def pull_all_lb_data():
    """
    Pull all LB data from MongoDB for the past 15 years
    """
    # MongoDB connection
    mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)

    # Access LB database
    lb_db = client['LB']

    # Get all team collections
    team_collections = lb_db.list_collection_names()
    print(f"Found {len(team_collections)} team collections")

    all_lb_data = []

    # Define the fields we want for LB (based on actual LB data structure)
    lb_fields = [
        # Basic info
        "player_id", "Year", "Team", "player", "position", "age", "franchise_id", "player_game_count",

        # Core defensive metrics
        "grades_defense", "snap_counts_defense", "assists", "forced_fumbles",
        "grades_coverage_defense", "grades_pass_rush_defense", "grades_run_defense",
        "hits", "hurries", "missed_tackle_rate", "sacks", "stops", "tackles",
        "tackles_for_loss", "total_pressures",

        # Additional LB-specific metrics
        "grades_defense_penalty", "grades_tackle", "interceptions", "pass_break_ups",
        "penalties", "snap_counts_box", "snap_counts_offball", "snap_counts_pass_rush",
        "snap_counts_run_defense", "targets", "fumble_recoveries", "declined_penalties",
        "missed_tackles",

        # Team/contract info
        "Cap_Space", "adjusted_value", "Net EPA", "Win %",

        # Additional useful metrics
        "weighted_grade", "weighted_average_grade"
    ]
    
    # Pull data from each team collection
    for team in team_collections:
        print(f"Processing team: {team}")
        collection = lb_db[team]
        
        # Query for all years (2010-2024) and filter for LB position
        cursor = collection.find({
            "Year": {"$gte": 2010, "$lte": 2024},
            "position": "LB"  # Only get LB players
        })
        
        for doc in cursor:
            # Extract only the fields we want and clean the data
            filtered_doc = {}
            for field in lb_fields:
                value = doc.get(field)
                
                # Handle MongoDB number types and convert to appropriate Python types
                if isinstance(value, dict) and '$numberDouble' in value:
                    try:
                        filtered_doc[field] = float(value['$numberDouble'])
                    except (ValueError, TypeError):
                        filtered_doc[field] = None
                elif isinstance(value, dict) and '$numberInt' in value:
                    try:
                        filtered_doc[field] = int(value['$numberInt'])
                    except (ValueError, TypeError):
                        filtered_doc[field] = None
                elif value == "MISSING" or value == "NaN":
                    filtered_doc[field] = None
                else:
                    filtered_doc[field] = value
            
            all_lb_data.append(filtered_doc)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_lb_data)
    
    print(f"Pulled {len(df)} LB records")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    
    # Data cleaning
    print("\nData cleaning...")
    
    # Convert numeric columns
    numeric_columns = [
        'Year', 'age', 'snap_counts_defense', 'assists', 'forced_fumbles', 'hits', 'hurries', 
        'sacks', 'stops', 'tackles', 'tackles_for_loss', 'total_pressures', 'interceptions', 
        'pass_break_ups', 'penalties', 'snap_counts_box', 'snap_counts_offball', 
        'snap_counts_pass_rush', 'snap_counts_run_defense', 'targets', 'fumble_recoveries', 
        'declined_penalties', 'missed_tackles', 'Cap_Space', 'adjusted_value', 'Net EPA', 
        'weighted_grade', 'player_game_count', 'franchise_id'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert grade columns
    grade_columns = [
        'grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense', 
        'grades_run_defense', 'grades_defense_penalty', 'grades_tackle', 'missed_tackle_rate'
    ]
    
    for col in grade_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean Win % column
    if 'Win %' in df.columns:
        df['Win %'] = df['Win %'].str.replace('%', '').astype(float)
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['player_id', 'Year', 'Team', 'player'])

    print(f"After cleaning: {len(df_clean)} records")
    print(f"Missing values per column:")
    missing_counts = df_clean.isnull().sum().sort_values(ascending=False)
    print(missing_counts[missing_counts > 0])

    # Save raw individual player data
    df_clean.to_csv('all_lb_data_15_years.csv', index=False)
    print("\nRaw player data saved to 'all_lb_data_15_years.csv'")

    # Calculate weighted averages by team and year
    weighted_avg_df = calculate_weighted_averages_by_team_year(df_clean)

    # Save weighted averages
    weighted_avg_df.to_csv('lb_weighted_averages_by_team_pos_year.csv', index=False)
    print("\nWeighted averages saved to 'lb_weighted_averages_by_team_pos_year.csv'")

    return df_clean, weighted_avg_df

def push_weighted_averages_to_mongo(weighted_avg_df):
    """
    Push weighted averages back to MongoDB
    Creates/updates a collection called 'LB_Weighted_Averages' in the LB database
    """
    print("\nPushing weighted averages to MongoDB...")

    # MongoDB connection
    mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)

    # Access LB database
    lb_db = client['LB']

    # Create or access the weighted averages collection
    weighted_collection = lb_db['LB_Weighted_Averages']

    # Clear existing data (optional - remove if you want to keep old data)
    weighted_collection.delete_many({})
    print("  Cleared existing data from LB_Weighted_Averages collection")

    # Convert DataFrame to list of dictionaries
    records = weighted_avg_df.to_dict('records')

    # Convert NaN values to None for MongoDB compatibility
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None

    # Insert records
    if len(records) > 0:
        result = weighted_collection.insert_many(records)
        print(f"  Inserted {len(result.inserted_ids)} records into LB_Weighted_Averages collection")
    else:
        print("  No records to insert")

    client.close()
    print("  MongoDB connection closed")

def pull_lb_data_by_years(start_year=2010, end_year=2024):
    """
    Pull LB data for specific year range
    """
    mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    lb_db = client['LB']
    team_collections = lb_db.list_collection_names()

    all_lb_data = []

    for team in team_collections:
        collection = lb_db[team]
        cursor = collection.find({
            "Year": {"$gte": start_year, "$lte": end_year}
        })

        for doc in cursor:
            all_lb_data.append(doc)

    df = pd.DataFrame(all_lb_data)
    print(f"Pulled {len(df)} LB records from {start_year}-{end_year}")
    return df

if __name__ == "__main__":
    # Pull all LB data for the past 15 years
    lb_data, weighted_avg_data = pull_all_lb_data()

    # Display sample raw data
    print("\n" + "="*80)
    print("SAMPLE RAW PLAYER DATA:")
    print("="*80)
    print(lb_data.head())
    print(f"\nRaw data shape: {lb_data.shape}")

    # Display weighted averages
    print("\n" + "="*80)
    print("WEIGHTED AVERAGES BY TEAM, POSITION, AND YEAR:")
    print("="*80)
    print(weighted_avg_data.head(20))  # Show first 20 rows
    print(f"\nWeighted averages shape: {weighted_avg_data.shape}")
    print(f"\nColumns in weighted averages: {len(weighted_avg_data.columns)} total")
    print(f"\nSample columns:")
    print(f"  - Grouping: Team, position, Year")
    print(f"  - Sums: sum_Cap_Space, sum_adjusted_value")
    print(f"  - Team stats: Win_Percent, Net_EPA")
    print(f"  - Current weighted averages: weighted_avg_* (31 stats)")
    print(f"  - Previous year weighted averages: prev_weighted_avg_* (31 stats)")
    print(f"  - Previous year summary: prev_total_snap_counts_defense, prev_total_players, prev_sum_Cap_Space, prev_sum_adjusted_value")

    # Push weighted averages back to MongoDB
    push_weighted_averages_to_mongo(weighted_avg_data)

    print("\n" + "="*80)
    print("PROCESS COMPLETE!")
    print("="*80)
    print(f"✓ Raw data saved to: all_lb_data_15_years.csv")
    print(f"✓ Weighted averages saved to: lb_weighted_averages_by_team_pos_year.csv")
    print(f"✓ Weighted averages pushed to MongoDB collection: LB_Weighted_Averages")
    print(f"\nGrouping: Team × Position × Year")
    print(f"Aggregations: Sum (Cap_Space, adjusted_value), Keep (Win %, EPA), Weighted Avg (31 stats)")
    print(f"Lagged Features: Previous year's data for all weighted averages (for LSTM input)")
