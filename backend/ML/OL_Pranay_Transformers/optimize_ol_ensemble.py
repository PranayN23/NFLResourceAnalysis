#!/usr/bin/env python3
"""
Optimize OL ensemble: blend weights + tier thresholds + XGB hyperparameters
Tests on validation set (2023) for best tier accuracy
"""
import os, sys, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from backend.ML.OL_Pranay_Transformers.Player_Model_OL import PlayerTransformerRegressor, Time2Vec
from backend.agent.ol_model_wrapper import OLModelInference

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_OUT = os.path.join(os.path.dirname(__file__), "ol_best_classifier.pth")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "ol_player_scaler.joblib")
XGB_MODEL_OUT = os.path.join(os.path.dirname(__file__), "ol_best_xgb.joblib")

print("=" * 80)
print("OL ENSEMBLE PARAMETER OPTIMIZATION")
print("=" * 80)

# Load & prep data
df_guard = pd.read_csv(os.path.join(parent_dir, "G.csv"))
df_center = pd.read_csv(os.path.join(parent_dir, "C.csv"))
df_tackle = pd.read_csv(os.path.join(parent_dir, "T.csv"))
df = pd.concat([df_guard, df_center, df_tackle], axis=0, ignore_index=True)
df = pd.get_dummies(df, columns=['position'], prefix='pos')

for col in ['grades_offense', 'grades_run_block', 'grades_pass_block', 'adjusted_value', 'Cap_Space',
            'age', 'snap_counts_offense', 'snap_counts_run_block', 'snap_counts_pass_block',
            'snap_counts_block', 'sacks_allowed', 'hits_allowed', 'hurries_allowed',
            'pressures_allowed', 'penalties', 'pbe', 'Net EPA']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['snap_counts_offense'] >= 100].copy()
df.sort_values(by=['player', 'Year'], inplace=True)

df["years_in_league"] = df.groupby("player").cumcount()
df["delta_grade"] = df.groupby("player")["grades_offense"].diff().fillna(0)
df["delta_run_block"] = df.groupby("player")["grades_run_block"].diff().fillna(0)
df["delta_pass_block"] = df.groupby("player")["grades_pass_block"].diff().fillna(0)
df['team_performance_proxy'] = df.groupby(['Team', 'Year'])['Net EPA'].transform('mean')
df['hits_allowed_rate'] = df['hits_allowed'] / df['snap_counts_pass_block']
df['hurries_allowed_rate'] = df['hurries_allowed'] / df['snap_counts_pass_block']
df['pressures_allowed_rate'] = df['pressures_allowed'] / df['snap_counts_pass_block']
df['penalties_rate'] = df['penalties'] / df['snap_counts_offense']
df['pass_block_efficiency'] = df['pbe']
df['snap_counts_block_share'] = df['snap_counts_block'] / df['snap_counts_offense']
df['snap_counts_run_block_share'] = df['snap_counts_run_block'] / df['snap_counts_offense']
df['snap_counts_pass_block_share'] = df['snap_counts_pass_block'] / df['snap_counts_offense']

val_data = df[df['Year'] == 2023].dropna(subset=['grades_offense']).copy()
print(f"\nValidation set (2023): {len(val_data)} rows")

print("Loading trained models...")
engine = OLModelInference(MODEL_OUT, SCALER_OUT, XGB_MODEL_OUT)

# Collect predictions for validation set
print("\nCollecting validator predictions...")
predictions = []
for _, row in val_data.iterrows():
    name = row['player']
    history = df[(df['player'] == name) & (df['Year'] < 2023)].copy()
    if len(history) == 0:
        continue
    transformer_signal = engine._compute_transformer_signal(history)
    df_history, df_xgb = engine._prepare_features(history)
    xgb_columns = list(engine.xgb_model.feature_names_in_) if hasattr(engine.xgb_model, 'feature_names_in_') else engine.xgb_features
    for col in xgb_columns:
        if col not in df_xgb.columns:
            df_xgb[col] = 0.0
    xgb_grade = engine.xgb_model.predict(df_xgb[xgb_columns])[0] if engine.xgb_model else 0
    predictions.append({'player': name, 'actual': row['grades_offense'], 'transformer': transformer_signal, 'xgb': xgb_grade})

pred_df = pd.DataFrame(predictions)
print(f"Predictions collected for {len(pred_df)} players")

def grade_to_tier(grade, thresholds):
    elite, starter, rotation = thresholds
    if grade >= elite: return "Elite"
    elif grade >= starter: return "Starter"
    elif grade >= rotation: return "Rotation"
    else: return "Reserve"

def actual_tier(grade):
    if grade >= 80: return "Elite"
    elif grade >= 65: return "Starter"
    elif grade >= 55: return "Rotation"
    else: return "Reserve"

# PART 1: Optimize blend weights
print("\n" + "=" * 80)
print("PART 1: OPTIMIZING ENSEMBLE BLEND WEIGHTS")
print("=" * 80)
print(f"{'XGB_Weight':<12} {'Trans_Weight':<12} {'Accuracy':<12} {'Precision':<12}")
print("-" * 48)

best_weight = 0.5
best_accuracy = 0.0

for xgb_w in np.arange(0.5, 1.01, 0.05):
    trans_w = 1.0 - xgb_w
    ensemble_preds = pred_df['transformer'] * trans_w + pred_df['xgb'] * xgb_w
    pred_tiers = [grade_to_tier(g, (80, 65, 55)) for g in ensemble_preds]
    actual_tiers = [actual_tier(g) for g in pred_df['actual']]
    acc = accuracy_score(actual_tiers, pred_tiers)
    if acc > best_accuracy:
        best_accuracy = acc
        best_weight = xgb_w
    precision, _, _, _ = precision_recall_fscore_support(actual_tiers, pred_tiers, average='weighted', zero_division=0)
    print(f"{xgb_w:<12.2f} {trans_w:<12.2f} {acc:<12.4f} {precision:<12.4f}")

print(f"\n✓ Best XGB weight: {best_weight:.2f} | Accuracy: {best_accuracy:.4f}")

# PART 2: Optimize tier thresholds
print("\n" + "=" * 80)
print("PART 2: OPTIMIZING TIER THRESHOLDS")
print("=" * 80)
ensemble_preds = pred_df['transformer'] * (1 - best_weight) + pred_df['xgb'] * best_weight
best_thresholds = (80, 65, 55)
best_threshold_accuracy = 0.0

print("Testing thresholds...")
for elite_t in range(75, 86):
    for starter_t in range(60, 70):
        for rotation_t in range(50, 60):
            if rotation_t >= starter_t or starter_t >= elite_t:
                continue
            pred_tiers = [grade_to_tier(g, (elite_t, starter_t, rotation_t)) for g in ensemble_preds]
            actual_tiers = [actual_tier(g) for g in pred_df['actual']]
            acc = accuracy_score(actual_tiers, pred_tiers)
            if acc > best_threshold_accuracy:
                best_threshold_accuracy = acc
                best_thresholds = (elite_t, starter_t, rotation_t)

print(f"✓ Best thresholds: Elite≥{best_thresholds[0]}, Starter≥{best_thresholds[1]}, Rotation≥{best_thresholds[2]}")
print(f"  Accuracy: {best_threshold_accuracy:.4f}")

# PART 3: Optimize XGB hyperparameters
print("\n" + "=" * 80)
print("PART 3: OPTIMIZING XGBOOST HYPERPARAMETERS")
print("=" * 80)

train_data = df[df['Year'] < 2023].dropna(subset=['grades_offense']).copy()
print(f"Training set: {len(train_data)} rows\n")

groups = train_data.groupby('player')
train_data['lag_grades_offense'] = groups['grades_offense'].shift(1)
train_data['lag_grades_run_block'] = groups['grades_run_block'].shift(1)
train_data['lag_grades_pass_block'] = groups['grades_pass_block'].shift(1)
train_data['delta_grade_lag'] = groups['grades_offense'].diff().shift(1)
train_data['team_performance_proxy_lag'] = groups['team_performance_proxy'].shift(1)
train_data['sacks_allowed_rate'] = train_data['sacks_allowed'] / train_data['snap_counts_pass_block']
train_data['t2v_transformer_signal'] = 0.0

XGB_FEATURES = ['lag_grades_offense', 'lag_grades_run_block', 'lag_grades_pass_block',
                'adjusted_value', 'age', 'years_in_league', 'delta_grade_lag',
                'team_performance_proxy_lag', 'sacks_allowed_rate', 'hits_allowed_rate',
                'hurries_allowed_rate', 't2v_transformer_signal']

train_clean = train_data.dropna(subset=XGB_FEATURES + ['grades_offense']).copy()

xgb_configs = [
    {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5},
    {'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 5},
    {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5},
    {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 5},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 4},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6},
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7},
]

print(f"{'n_estimators':<15} {'learning_rate':<15} {'max_depth':<12} {'Accuracy':<12}")
print("-" * 54)

best_xgb_config = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5}
best_xgb_accuracy = 0.0

for config in xgb_configs:
    xgb_model_test = xgb.XGBRegressor(n_estimators=config['n_estimators'],
                                     learning_rate=config['learning_rate'],
                                     max_depth=config['max_depth'],
                                     random_state=42, verbosity=0)
    xgb_model_test.fit(train_clean[XGB_FEATURES].fillna(0), train_clean['grades_offense'])
    
    xgb_val_preds = xgb_model_test.predict(val_data[XGB_FEATURES].fillna(0))
    ensemble_val = pred_df['transformer'].values * (1 - best_weight) + xgb_val_preds * best_weight
    pred_tiers = [grade_to_tier(g, best_thresholds) for g in ensemble_val]
    actual_tiers = [actual_tier(g) for g in val_data['grades_offense'].values]
    acc = accuracy_score(actual_tiers, pred_tiers)
    
    if acc > best_xgb_accuracy:
        best_xgb_accuracy = acc
        best_xgb_config = config
    
    print(f"{config['n_estimators']:<15} {config['learning_rate']:<15} {config['max_depth']:<12} {acc:<12.4f}")

print(f"\n✓ Best XGB config: {best_xgb_config} | Accuracy: {best_xgb_accuracy:.4f}")

# SUMMARY
print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)
print(f"\n✓ Optimal ensemble XGB weight: {best_weight:.2f}")
print(f"  Accuracy: {best_accuracy:.4f} (baseline: 0.4984 | Improvement: +{(best_accuracy-0.4984):.4f})")
print(f"\n✓ Optimal tier thresholds: Elite≥{best_thresholds[0]}, Starter≥{best_thresholds[1]}, Rotation≥{best_thresholds[2]}")
print(f"  Accuracy: {best_threshold_accuracy:.4f} (Improvement: +{(best_threshold_accuracy-0.4984):.4f})")
print(f"\n✓ Optimal XGB hyperparameters: {best_xgb_config}")
print(f"  Accuracy: {best_xgb_accuracy:.4f} (Improvement: +{(best_xgb_accuracy-0.4984):.4f})")
print("\n" + "=" * 80)
print("APPLY THESE CHANGES:")
print("=" * 80)
print(f"1. ol_model_wrapper.py (predict method):")
print(f"   trans_weight, xgb_weight = {1-best_weight:.2f}, {best_weight:.2f}")
print(f"\n2. ol_model_wrapper.py (get_tier method):")
print(f"   if grade >= {best_thresholds[0]}: return 'Elite'")
print(f"   elif grade >= {best_thresholds[1]}: return 'Starter'")
print(f"   elif grade >= {best_thresholds[2]}: return 'Rotation'")
print(f"\n3. OL_Ensemble.py (XGBRegressor line):")
print(f"   xgb_model = xgb.XGBRegressor(n_estimators={best_xgb_config['n_estimators']}, learning_rate={best_xgb_config['learning_rate']}, max_depth={best_xgb_config['max_depth']}, random_state=42)")
print("=" * 80)
