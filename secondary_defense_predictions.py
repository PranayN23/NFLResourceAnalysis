import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def main():
    # Load the secondary defense data and filter for relevant positions
    secondary_data = pd.read_csv('SecondaryDefenseData.csv')
    secondary_data = secondary_data[(secondary_data['position'] == 'CB') | (secondary_data['position'] == 'S') | 
                                     (secondary_data['position'] == 'DI') | (secondary_data['position'] == 'ED') | 
                                     (secondary_data['position'] == 'LB')]
    
    # Replace CB and S with DB, and DI and ED with DL
    rename_mapping = {'CB': 'DB', 'S': 'DB', 'DI': 'DL', 'ED': 'DL'}
    secondary_data['position'] = secondary_data['position'].replace(rename_mapping)

    # Define the weighted average function
    def weighted_avg(group, position):
        total_snap_counts = group['snap_counts_defense'].sum()
        weighted_data = {
            'Year': group['Year'].iloc[0],
            'Team': group['Team'].iloc[0],
            'Position': position,
            'snap_counts_defense': total_snap_counts
        }
        # Calculate weighted average for each metric, including both weighted and previous columns
        weighted_columns = [col for col in group.columns if col.startswith('weighted_avg_')]
        previous_columns = [col for col in group.columns if col.startswith('Previous_')]
        
        for col in weighted_columns + previous_columns:
            weighted_data[col] = (group[col] * group['snap_counts_defense']).sum() / total_snap_counts if total_snap_counts > 0 else 0
        
        return pd.Series(weighted_data)

    # Apply weighted_avg function separately for DB and DL
    db_data = secondary_data[secondary_data['position'] == 'DB']
    dl_data = secondary_data[secondary_data['position'] == 'DL']
    lb_data = secondary_data[secondary_data['position'] == 'LB']

    db_grouped = db_data.groupby(['Year', 'Team']).apply(weighted_avg, position='DB').reset_index(drop=True)
    dl_grouped = dl_data.groupby(['Year', 'Team']).apply(weighted_avg, position='DL').reset_index(drop=True)

    # LB data remains as is, no weighted averaging applied
    lb_grouped = lb_data.copy()
    lb_grouped = lb_grouped.rename(columns={'position': 'Position'})  # Ensure consistent naming

    # Combine all results into a single dataframe
    final_data = pd.concat([db_grouped, dl_grouped, lb_grouped], ignore_index=True)

    # Standardize column names for merging
    final_data = final_data.rename(columns={
        'weighted_avg_grades_defense': 'Current_PFF'  # Rename grade column for merging
    })

    # Save the final combined data to a CSV
    final_data.to_csv("separate_positions_secondary_data.csv", index=False)


        # Load additional data and filter for DB position
    additional_data = pd.read_csv('data.csv')
    additional_data = additional_data[additional_data['Position'].isin(["DB", "LB", "DL"])]
    additional_data.to_csv("test.csv", index=False)  # Save intermediate file for verification
    
    # Remove any unnamed or blank columns
    #secondary_data = secondary_data.loc[:, ~secondary_data.columns.str.contains('^Unnamed')]
    #secondary_data = secondary_data.loc[:, secondary_data.columns != '']
    #additional_data = additional_data.loc[:, ~additional_data.columns.str.contains('^Unnamed')]
    #additional_data = additional_data.loc[:, additional_data.columns != '']
    
    # Merge additional data with the weighted-averaged secondary data on key columns
    combined_data = additional_data.merge(
        final_data,
        on=['Team', 'Year', 'Position']
    )
    # Select relevant columns for the final combined_data DataFrame, including Previous_ columns
    combined_data = combined_data[
    [
        'Team', 'Year', 'Position', 'Value_cap_space', 'Value_draft_data', 
        'Current_AV', 'Current_PFF_x', 'Total DVOA', 
        'win-loss-pct', 'Net EPA', 'snap_counts_defense', 'weighted_avg_player', 
        'weighted_avg_player_id', 'weighted_avg_player_game_count', 'weighted_avg_assists', 
        'weighted_avg_batted_passes', 'weighted_avg_declined_penalties', 
        'weighted_avg_forced_fumbles', 'weighted_avg_franchise_id', 'weighted_avg_fumble_recoveries', 
        'weighted_avg_fumble_recovery_touchdowns', 'weighted_avg_grades_coverage_defense', 
        'weighted_avg_grades_defense_penalty', 'weighted_avg_grades_pass_rush_defense', 
        'weighted_avg_grades_run_defense', 'weighted_avg_grades_tackle', 'weighted_avg_hits', 
        'weighted_avg_hurries', 'weighted_avg_interception_touchdowns', 'weighted_avg_interceptions', 'weighted_avg_missed_tackle_rate', 'weighted_avg_missed_tackles', 
        'weighted_avg_pass_break_ups', 'weighted_avg_penalties', 'weighted_avg_qb_rating_against', 
        'weighted_avg_receptions', 'weighted_avg_sacks', 'weighted_avg_safeties', 
        'weighted_avg_snap_counts_box', 'weighted_avg_snap_counts_corner', 'weighted_avg_snap_counts_coverage', 
        'weighted_avg_snap_counts_dl', 'weighted_avg_snap_counts_dl_a_gap', 'weighted_avg_snap_counts_dl_b_gap', 
        'weighted_avg_snap_counts_dl_outside_t', 'weighted_avg_snap_counts_dl_over_t', 'weighted_avg_snap_counts_fs', 
        'weighted_avg_snap_counts_offball', 'weighted_avg_snap_counts_pass_rush', 'weighted_avg_snap_counts_run_defense', 
        'weighted_avg_snap_counts_slot', 'weighted_avg_stops', 'weighted_avg_tackles', 'weighted_avg_tackles_for_loss', 
        'weighted_avg_targets', 'weighted_avg_total_pressures', 'weighted_avg_touchdowns', 'weighted_avg_yards'
    ] + [col for col in combined_data.columns if col.startswith('Previous_')]
]

    combined_data = combined_data.rename(columns={'Current_PFF_x': 'Current_PFF'})
    combined_data = combined_data.rename(columns={'Previous_grades_defense': 'Previous_PFF'})
    # Save the final combined data to a CSV
    combined_data.to_csv("Combined_Secondary_Defense.csv")

    db_data = combined_data[combined_data['Position'] == 'DB']
    lb_data = combined_data[combined_data['Position'] == 'LB']
    dl_data = combined_data[combined_data['Position'] == 'DL']

    # Save each subset to a separate file
    db_data.to_csv("DB_Secondary_Defense.csv", index=False)
    lb_data.to_csv("LB_Secondary_Defense.csv", index=False)
    dl_data.to_csv("DL_Secondary_Defense.csv", index=False)


def sklearn_mlp(df, metric):
    # Selecting features and target variable
    feature_columns = ['Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Previous_PFF']
    target_column = f'Current_{metric}'
    
    # Train-test split based on year
    features_train = df[df['Year'] <= 2021][feature_columns]
    labels_train = df[df['Year'] <= 2021][target_column]
    features_test = df[df['Year'] == 2022][feature_columns]
    labels_test = df[df['Year'] == 2022][target_column]
    
    # One-hot encoding for categorical features (if any)
    features_train = pd.get_dummies(features_train)
    features_test = pd.get_dummies(features_test)
    features_train, features_test = features_train.align(features_test, join='left', axis=1, fill_value=0)

    # Standardizing features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Model configurations
    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]
    
    for learning_rate in learning_rates:
        for size in sizes:
            print(f'Scikit-learn MLP for {metric} - Learning Rate {learning_rate}, Size {size}')
            mlp = MLPRegressor(hidden_layer_sizes=size, max_iter=1000, alpha=0.01, learning_rate_init=learning_rate)
            mlp.fit(features_train, labels_train)
            
            # Predict and calculate R² scores
            train_r2 = r2_score(labels_train, mlp.predict(features_train))
            test_r2 = r2_score(labels_test, mlp.predict(features_test))
            
            print(f'    Training R²: {train_r2:.2f}')
            print(f'    Test R²: {test_r2:.2f}')

def tensorflow_mlp(df, metric):
    # Selecting features and target variable
    feature_columns = ['Value_cap_space', 'Value_draft_data', 'Previous_AV', 'Previous_PFF']
    target_column = f'Current_{metric}'
    
    # Train-test split based on year
    features_train = df[df['Year'] <= 2021][feature_columns]
    labels_train = df[df['Year'] <= 2021][target_column]
    features_test = df[df['Year'] == 2022][feature_columns]
    labels_test = df[df['Year'] == 2022][target_column]
    
    # One-hot encoding and alignment for categorical features
    features_train = pd.get_dummies(features_train)
    features_test = pd.get_dummies(features_test)
    features_train, features_test = features_train.align(features_test, join='left', axis=1, fill_value=0)

    # Standardizing features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Model configurations
    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10)]
    
    for learning_rate in learning_rates:
        for size in sizes:
            print(f'TensorFlow MLP for {metric} - Learning Rate {learning_rate}, Size {size}')
            
            # Building the model
            model = tf.keras.Sequential([tf.keras.layers.Dense(neurons, activation='relu') for neurons in size] + [tf.keras.layers.Dense(1)])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            
            # Train the model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(features_train, labels_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

            # Calculate R² scores
            train_r2 = r2_score(labels_train, model.predict(features_train).flatten())
            test_r2 = r2_score(labels_test, model.predict(features_test).flatten())
            
            print(f'    Training R²: {train_r2:.2f}')
            print(f'    Test R²: {test_r2:.2f}')

if __name__ == "__main__":
    main()
