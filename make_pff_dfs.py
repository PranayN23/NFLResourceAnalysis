import pandas as pd

def get_pff():
    years = list(range(2010,2025))
    counts = {'QB' : 'passing_snaps', 'HB' : 'total_touches', 'WR' : 'total_snaps', 'TE' : 'total_snaps',
              'T' : 'snap_counts_offense', 'G' : 'snap_counts_offense', 'C' : 'snap_counts_offense', 
              'DI' : 'snap_counts_defense', 'ED' : 'snap_counts_defense', 'LB' : 'snap_counts_defense',
              'CB' : 'snap_counts_defense', 'S' : 'snap_counts_defense'}
    positions = {'QB' : [
                'player', 'player_id', 'team_name', 'position', 'accuracy_percent', 'aimed_passes', 'attempts', 'avg_depth_of_target',
                'avg_time_to_throw', 'bats', 'big_time_throws', 'btt_rate', 
                'completion_percent', 'completions', 'declined_penalties', 
                'def_gen_pressures', 'drop_rate', 'dropbacks', 'drops', 
                'first_downs', 'franchise_id', 'grades_hands_fumble', 
                'grades_offense', 'grades_pass', 'grades_run', 'hit_as_threw', 
                'interceptions', 'passing_snaps', 'penalties', 
                'pressure_to_sack_rate', 'qb_rating', 'sack_percent', 
                'sacks', 'scrambles', 'spikes', 'thrown_aways', 
                'touchdowns', 'turnover_worthy_plays', 'twp_rate', 
                'yards', 'ypa'
            ], 'HB' : [
            'player', 'player_id', 'team_name', 'position', 'attempts', 'avoided_tackles', 'breakaway_attempts', 'breakaway_percent',
            'breakaway_yards', 'declined_penalties', 'designed_yards', 'drops',
            'elu_recv_mtf', 'elu_rush_mtf', 'elu_yco', 'elusive_rating', 'explosive',
            'first_downs', 'franchise_id', 'fumbles', 'gap_attempts', 'grades_hands_fumble',
            'grades_offense', 'grades_offense_penalty', 'grades_pass', 'grades_pass_block',
            'grades_pass_route', 'grades_run', 'grades_run_block', 'longest', 'penalties',
            'rec_yards', 'receptions', 'routes', 'run_plays', 'scramble_yards', 'scrambles',
            'targets', 'total_touches', 'touchdowns', 'yards', 'yards_after_contact',
            'yco_attempt', 'ypa', 'yprr', 'zone_attempts'
        ], 'WR' : [
            'player', 'player_id', 'team_name', 'position', 'avg_depth_of_target', 'avoided_tackles', 'caught_percent', 'contested_catch_rate',
            'contested_receptions', 'contested_targets', 'declined_penalties', 'drop_rate', 'drops',
            'first_downs', 'franchise_id', 'fumbles', 'grades_hands_drop', 'grades_hands_fumble',
            'grades_offense', 'grades_pass_block', 'grades_pass_route', 'inline_rate',
            'interceptions', 'longest', 'pass_block_rate', 'pass_blocks', 'pass_plays',
            'penalties', 'receptions', 'route_rate', 'routes', 'slot_rate', 'targeted_qb_rating',
            'targets', 'touchdowns', 'wide_rate', 'yards', 'yards_after_catch',
            'yards_after_catch_per_reception', 'yards_per_reception', 'yprr', 'inline_snaps',
            'slot_snaps', 'wide_snaps'
        ], 'TE': [
        'player', 'player_id', 'team_name', 'position', "avg_depth_of_target", "avoided_tackles", "caught_percent",
        "contested_catch_rate", "contested_receptions", "contested_targets",
        "declined_penalties", "drop_rate", "drops", "first_downs",
        "franchise_id", "fumbles", "grades_hands_drop", "grades_hands_fumble",
        "grades_offense", "grades_pass_block", "grades_pass_route",
        "inline_rate", "inline_snaps", "interceptions", "longest",
        "pass_block_rate", "pass_blocks", "pass_plays", "penalties",
        "receptions", "route_rate", "routes", "slot_rate", "slot_snaps",
        "targeted_qb_rating", "targets", "touchdowns", "wide_rate",
        "wide_snaps", "yards", "yards_after_catch", "yards_after_catch_per_reception",
        "yards_per_reception", "yprr", "snap_counts_pass_block", "snap_counts_run_block"
        ], 'T' : [
            'player', 'player_id', 'team_name', 'position', "block_percent", "declined_penalties", "franchise_id", "grades_offense",
            "grades_pass_block", "grades_run_block", "hits_allowed", "hurries_allowed",
            "non_spike_pass_block", "non_spike_pass_block_percentage", "pass_block_percent",
            "pbe", "penalties", "pressures_allowed", "sacks_allowed", "snap_counts_block",
            "snap_counts_ce", "snap_counts_lg", "snap_counts_lt", "snap_counts_offense",
            "snap_counts_pass_block", "snap_counts_pass_play", "snap_counts_rg",
            "snap_counts_rt", "snap_counts_run_block", "snap_counts_te"
        ], 'G' : [
            'player', 'player_id', 'team_name', 'position', "block_percent", "declined_penalties", "franchise_id", "grades_offense",
            "grades_pass_block", "grades_run_block", "hits_allowed", "hurries_allowed",
            "non_spike_pass_block", "non_spike_pass_block_percentage", "pass_block_percent",
            "pbe", "penalties", "pressures_allowed", "sacks_allowed", "snap_counts_block",
            "snap_counts_ce", "snap_counts_lg", "snap_counts_lt", "snap_counts_offense",
            "snap_counts_pass_block", "snap_counts_pass_play", "snap_counts_rg",
            "snap_counts_rt", "snap_counts_run_block", "snap_counts_te"
        ], 'C' : [
            'player', 'player_id', 'team_name', 'position', "block_percent", "declined_penalties", "franchise_id", "grades_offense",
            "grades_pass_block", "grades_run_block", "hits_allowed", "hurries_allowed",
            "non_spike_pass_block", "non_spike_pass_block_percentage", "pass_block_percent",
            "pbe", "penalties", "pressures_allowed", "sacks_allowed", "snap_counts_block",
            "snap_counts_ce", "snap_counts_lg", "snap_counts_lt", "snap_counts_offense",
            "snap_counts_pass_block", "snap_counts_pass_play", "snap_counts_rg",
            "snap_counts_rt", "snap_counts_run_block", "snap_counts_te"
        ],  "DI": [  # Defensive Interior (Defensive Tackles, Nose Tackles)
        'player', 'team_name', 'position', "snap_counts_defense", "player_id", "player_game_count", 
        "assists", "batted_passes", "declined_penalties", "forced_fumbles", "franchise_id", 
        "fumble_recoveries", "fumble_recovery_touchdowns", "grades_defense", "grades_defense_penalty", 
        "grades_pass_rush_defense", "grades_run_defense", "grades_tackle", "hits", "hurries", 
        "missed_tackle_rate", "missed_tackles", "penalties", "qb_rating_against", "sacks", 
        "snap_counts_dl", "snap_counts_dl_a_gap", "snap_counts_dl_b_gap", "snap_counts_dl_outside_t", 
        "snap_counts_dl_over_t", "stops", "tackles", "tackles_for_loss", "total_pressures"
    ],

    "ED": [  # Edge Rushers (Defensive Ends, Outside Linebackers in a 3-4)
        'player', 'team_name', 'position', "snap_counts_defense", "player_id", "player_game_count", 
        "assists", "batted_passes", "declined_penalties", "forced_fumbles", "franchise_id", 
        "fumble_recoveries", "grades_defense", "grades_defense_penalty", "grades_pass_rush_defense", 
        "grades_run_defense", "grades_tackle", "hits", "hurries", "missed_tackle_rate", 
        "missed_tackles", "penalties", "sacks", "snap_counts_pass_rush", "snap_counts_run_defense", 
        "snap_counts_dl", "snap_counts_dl_outside_t", "snap_counts_dl_over_t", "stops", 
        "tackles", "tackles_for_loss", "total_pressures"
    ],

    "LB": [  # Linebackers (Inside Linebackers, Off-Ball Linebackers in 4-3)
        'player', 'team_name', 'position', "snap_counts_defense", "player_id", "player_game_count", 
        "assists", "declined_penalties", "forced_fumbles", "franchise_id", "fumble_recoveries", 
        "grades_coverage_defense", "grades_defense", "grades_defense_penalty", "grades_pass_rush_defense", 
        "grades_run_defense", "grades_tackle", "hits", "hurries", "interceptions", 
        "missed_tackle_rate", "missed_tackles", "pass_break_ups", "penalties", "sacks", 
        "snap_counts_box", "snap_counts_offball", "snap_counts_pass_rush", "snap_counts_run_defense", 
        "stops", "tackles", "tackles_for_loss", "targets", "total_pressures"
    ],

    "CB": [  # Cornerbacks
        'player', 'team_name', 'position', "snap_counts_defense", "player_id", "player_game_count", 
        "assists", "declined_penalties", "forced_fumbles", "franchise_id", "fumble_recoveries", 
        "grades_coverage_defense", "grades_defense", "grades_defense_penalty", "grades_tackle", 
        "interceptions", "interception_touchdowns", "missed_tackle_rate", "missed_tackles", 
        "pass_break_ups", "penalties", "qb_rating_against", "receptions", "snap_counts_corner", 
        "snap_counts_coverage", "snap_counts_slot", "stops", "tackles", "tackles_for_loss", 
        "targets", "touchdowns", "yards"
    ],

    "S": [  # Safeties (Free Safety, Strong Safety)
        'player', 'team_name', 'position', "snap_counts_defense", "player_id", "player_game_count", 
        "assists", "declined_penalties", "forced_fumbles", "franchise_id", "fumble_recoveries", 
        "grades_coverage_defense", "grades_defense", "grades_defense_penalty", "grades_tackle", 
        "interceptions", "interception_touchdowns", "missed_tackle_rate", "missed_tackles", 
        "pass_break_ups", "penalties", "qb_rating_against", "receptions", "snap_counts_fs", 
        "snap_counts_box", "snap_counts_coverage", "snap_counts_slot", "stops", "tackles", 
        "tackles_for_loss", "targets", "touchdowns", "yards"
    ]
}


    for pos, output_columns in positions.items(): 
        pff = []

        for year in years:
            if pos == 'TE':
                df = pd.read_csv('PFF/Receiving'+ str(year) + '.csv')
                df2 = pd.read_csv('PFF/OL'+ str(year) + '.csv')
                df = df.merge(df2, on=['player', 'team_name', 'position'], how='inner')
                df = df.loc[:, ~df.columns.str.endswith('_y')]
                # Rename columns that end with '_x' to remove the suffix
                df.columns = df.columns.str.replace('_x', '', regex=False)
            else:
                read = pos
                if pos == "HB":
                    read = 'RB'
                if pos == 'WR':
                    read = "Receiving"
                if pos == 'T':
                    read = 'OL'
                if pos == 'G':
                    read = 'OL'
                if pos == 'C':
                    read = 'OL'
                if pos in ['DI', 'ED', 'LB', 'CB', 'S']:
                    read = 'Defense'
                df = pd.read_csv('PFF/' + read + str(year) + '.csv')
                positions_of_interest = [pos]
            df = df[df['position'] == pos]
            

            # Group by team_name and position, summing the weighted columns and total snaps
            # Create the team mapping
            team_mapping = {
                'WAS': 'Commanders', 'TEN': 'Titans', 'TB': 'Buccaneers', 
                'SF': '49ers', 'SEA': 'Seahawks', 'PIT': 'Steelers', 
                'PHI': 'Eagles', 'NYJ': 'Jets', 'NYG': 'Giants', 
                'NO': 'Saints', 'NE': 'Patriots', 'MIN': 'Vikings', 
                'MIA': 'Dolphins', 'LV': 'Raiders', 'LAC': 'Chargers', 
                'LA': 'Rams', 'KC': 'Chiefs', 'JAX': 'Jaguars', 
                'IND': 'Colts', 'HST': 'Texans', 'GB': 'Packers', 
                'DET': 'Lions', 'DEN': 'Broncos', 'DAL': 'Cowboys', 
                'CLV': 'Browns', 'CIN': 'Bengals', 'CHI': 'Bears', 
                'CAR': 'Panthers', 'ATL': 'Falcons', 'ARZ': 'Cardinals', 
                'BLT': 'Ravens', 'BUF': 'Bills', 'OAK': 'Raiders',
                'SL' : 'Rams', 'SD' : 'Chargers'
            }
            df = df[output_columns]
            if pos == 'WR':
                df['total_snaps'] = df['inline_snaps'] + df['slot_snaps'] + df['wide_snaps']
            if pos == 'TE':
                df['total_snaps'] = df['inline_snaps'] + df['slot_snaps'] + df['wide_snaps']
            
            df['Team'] = df['team_name'].replace(team_mapping)
            df['Year'] = year
            if pos in ['QB', 'HB', 'WR', 'TE', 'G', 'T', 'C']:
                df['weighted_grade'] = df['grades_offense'] * df[counts[pos]]
            else:
                df['weighted_grade'] = df['grades_defense'] * df[counts[pos]]
            # Ensure column names are stripped of whitespace
            df.columns = df.columns.str.strip()

            # Print columns to check if 'passing_snaps' exists

            # Perform the groupby and transformation
            df['weighted_average_grade'] = df.groupby(['Team', 'position', 'Year']).apply(
                lambda group: (group['weighted_grade'].sum() / group[counts[pos]].sum()) if group[counts[pos]].sum() > 0 else 0
            ).reset_index(level=[0, 1, 2], drop=True)



            # Append the result to the pff list
            pff.append(df)

        pd.concat(pff)

        
        result = pd.concat(pff, ignore_index=True)

        result.to_csv(str(pos) + 'PFF.csv')
get_pff()
