#!/usr/bin/env python3
"""
Calculate weighted averages for Edge (ED) players by team and year.
For each team-year combination, calculate weighted averages of all stats,
where each player's stat is weighted by their snap_counts_defense.
Store results in MongoDB.
"""

from pymongo import MongoClient
import numpy as np
from collections import defaultdict

# MongoDB connection
mongo_uri = "mongodb+srv://pranaynandkeolyar:nfl@cluster0.4nbxj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)

# Connect to ED database
ed_db = client['ED']
all_teams = ed_db.list_collection_names()

print(f"Found {len(all_teams)} team collections in ED database")

# Define numeric fields to calculate weighted averages for
numeric_fields = [
    'assists', 'batted_passes', 'forced_fumbles', 'fumble_recoveries',
    'grades_defense', 'grades_defense_penalty', 'grades_pass_rush_defense',
    'grades_run_defense', 'grades_tackle', 'hits', 'hurries',
    'missed_tackle_rate', 'missed_tackles', 'penalties', 'sacks',
    'snap_counts_pass_rush', 'snap_counts_run_defense', 'snap_counts_dl',
    'snap_counts_dl_outside_t', 'snap_counts_dl_over_t',
    'stops', 'tackles', 'tackles_for_loss', 'total_pressures'
]

# Dictionary to store team-year data: {(team, year): [players]}
team_year_data = defaultdict(list)

# Collect all player data grouped by team and year
print("\nCollecting player data from MongoDB...")
for team in all_teams:
    collection = ed_db[team]
    cursor = collection.find({'Year': {'$exists': True}})
    
    for doc in cursor:
        team_name = doc.get('Team')
        year = doc.get('Year')
        
        if team_name and year is not None:
            # Store player document
            team_year_data[(team_name, year)].append(doc)

print(f"Collected data for {len(team_year_data)} team-year combinations")

# Calculate weighted averages for each team-year combination
print("\nCalculating weighted averages...")
weighted_results = []

for (team, year), players in team_year_data.items():
    # Calculate total snap counts for this team-year
    total_snaps = 0
    valid_players = []
    
    for player in players:
        snap_count = player.get('snap_counts_defense')
        if snap_count is not None and snap_count > 0:
            try:
                snap_count = float(snap_count)
                total_snaps += snap_count
                valid_players.append((player, snap_count))
            except (ValueError, TypeError):
                continue
    
    if total_snaps == 0 or len(valid_players) == 0:
        continue
    
    # Initialize result dictionary
    result = {
        'Team': team,
        'Year': year,
        'Position': 'ED',
        'snap_counts_defense': total_snaps,
        'player_count': len(valid_players)
    }
    
    # Calculate weighted averages for each numeric field
    for field in numeric_fields:
        weighted_sum = 0
        valid_count = 0
        
        for player, snap_count in valid_players:
            value = player.get(field)
            if value is not None:
                try:
                    value = float(value)
                    if not np.isnan(value):
                        weighted_sum += value * snap_count
                        valid_count += 1
                except (ValueError, TypeError):
                    continue
        
        if valid_count > 0 and total_snaps > 0:
            result[field] = weighted_sum / total_snaps
        else:
            result[field] = 0
    
    # Add Current_PFF (same as grades_defense for compatibility with predictions notebook)
    result['Current_PFF'] = result.get('grades_defense', 0)
    
    weighted_results.append(result)

print(f"Calculated weighted averages for {len(weighted_results)} team-year combinations")

# Add Previous_ columns by shifting data by year for each team
print("\nCreating Previous_ columns...")

# Group results by team and sort by year
team_results = defaultdict(list)
for result in weighted_results:
    team_results[result['Team']].append(result)

# Sort each team's results by year
for team in team_results:
    team_results[team].sort(key=lambda x: x['Year'])

# Create Previous_ columns
fields_to_shift = numeric_fields + ['snap_counts_defense', 'grades_defense', 'Current_PFF']
fields_to_shift = [f for f in fields_to_shift if f in numeric_fields or f in ['snap_counts_defense', 'grades_defense', 'Current_PFF']]

# Update weighted_results with Previous_ columns
weighted_results = []  # Rebuild the list with Previous_ columns
for team, results in team_results.items():
    for i in range(len(results)):
        if i > 0:  # First year has no previous data
            prev_result = results[i-1]
            for field in fields_to_shift:
                prev_value = prev_result.get(field)
                if prev_value is not None:
                    results[i][f'Previous_{field}'] = prev_value
                else:
                    results[i][f'Previous_{field}'] = 0
        weighted_results.append(results[i])

print("Previous_ columns created")

# Store results in MongoDB
print("\nStoring weighted averages in MongoDB...")

# Create or access the weighted average database
wa_db = client['ED_Weighted_Average']

# Clear existing data (optional - comment out if you want to keep existing data)
# print("Clearing existing weighted average data...")
# for collection_name in wa_db.list_collection_names():
#     wa_db[collection_name].drop()

# Store results grouped by team (similar structure to original ED database)
for result in weighted_results:
    team = result['Team']
    year = result['Year']
    
    # Use team name as collection name
    team_collection = wa_db[team]
    
    # Check if document for this year already exists
    existing = team_collection.find_one({'Year': year, 'Team': team})
    
    if existing:
        # Update existing document
        team_collection.update_one(
            {'Year': year, 'Team': team},
            {'$set': result}
        )
        print(f"Updated: {team} - {year}")
    else:
        # Insert new document
        team_collection.insert_one(result)
        print(f"Inserted: {team} - {year}")

print(f"\n✅ Successfully stored weighted averages in MongoDB!")
print(f"Database: ED_Weighted_Average")
print(f"Structure: Each team is a collection, documents contain team-year weighted averages")

# Print summary statistics
if weighted_results:
    print(f"\nSummary:")
    print(f"  Total team-year combinations: {len(weighted_results)}")
    print(f"  Teams: {len(set(r['Team'] for r in weighted_results))}")
    print(f"  Years: {sorted(set(r['Year'] for r in weighted_results))}")
    
    # Show sample result
    print(f"\nSample weighted average result:")
    sample = weighted_results[0]
    print(f"  Team: {sample['Team']}")
    print(f"  Year: {sample['Year']}")
    print(f"  Total snaps: {sample['snap_counts_defense']:.1f}")
    print(f"  Player count: {sample['player_count']}")
    print(f"  Sample stats:")
    print(f"    grades_defense: {sample.get('grades_defense', 0):.2f}")
    print(f"    sacks: {sample.get('sacks', 0):.2f}")
    print(f"    total_pressures: {sample.get('total_pressures', 0):.2f}")
    print(f"    stops: {sample.get('stops', 0):.2f}")

client.close()
print("\n✅ Script completed!")

