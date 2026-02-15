import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Fix for Mac OpenMP deadlock
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.getcwd())
from backend.ML.TightEnds.TETime2Vec import load_and_engineer_features as load_xgb_data

DATA_FILE = "backend/ML/TightEnds/TE.csv"
TARGET = "weighted_grade"

def weighted_r2_score(y_true, y_pred, weights=None):
    if weights is None:
        return r2_score(y_true, y_pred)
    
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
        
    return 1 - (numerator / denominator)

def tune_xgboost():
    print("Loading data...")
    data, _ = load_xgb_data(DATA_FILE)
    
    # Features
    predictors = [c for c in data.columns if "_prev" in c or "_trend" in c or "_rolling" in c]
    predictors += [
        "age", "age_sq", "Cap_Space",
        "time_linear", "time_sin_1", "time_cos_1", 
        "time_sin_2", "time_cos_2", "time_sin_3", "time_cos_3",
        "career_year", "career_year_sq",
        "is_prime", "is_decline",
        "Growth_Potential", "Team_TE_EPA_History"
    ]
    predictors = [c for c in predictors if c in data.columns]
    
    # Split for 2024 Validation Optimization
    train_data = data[data["Year"] < 2024].copy()
    test_data = data[data["Year"] == 2024].copy()
    
    X_train = train_data[predictors]
    y_train = train_data[TARGET]
    X_test = test_data[predictors]
    y_test = test_data[TARGET]
    
    # Create Sample Weights for Training (weigh better players higher)
    # Using weighted_grade as base. Clip negative values to small positive.
    train_weights = train_data[TARGET].clip(lower=1.0)
    
    # Create Test Weights for Evaluation (Focus on top performers)
    test_weights = test_data[TARGET].clip(lower=1.0)
    # Enhance weights for top 25% performers in test set
    top_quartile = test_data[TARGET].quantile(0.75)
    test_weights = np.where(test_data[TARGET] >= top_quartile, test_weights * 2.0, test_weights)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples (2024): {len(X_test)}")
    
    # Define Parameter Grid
    param_grid = {
        'n_estimators': [500, 1000, 2000],
        'learning_rate': [0.005, 0.01, 0.05],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    import itertools
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Randomly sample 60 combinations
    import random
    random.seed(42)
    masked_combinations = random.sample(combinations, min(60, len(combinations)))
    
    print(f"Evaluated {len(masked_combinations)} random configurations...")
    
    best_score = -float('inf')
    best_params = None
    
    for i, params in enumerate(masked_combinations):
        model = xgb.XGBRegressor(
            **params,
            n_jobs=1,
            random_state=42,
            verbosity=0
        )
        
        # Fit with weights
        model.fit(X_train, y_train, sample_weight=train_weights)
        preds = model.predict(X_test)
        
        # Weighted R2 Score
        score = weighted_r2_score(y_test, preds, weights=test_weights)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New Best Weighted R2: {best_score:.4f} with params: {params}")
            
    print("\n==== BEST HYPERPARAMETERS ====")
    print(best_params)
    print(f"Best Weighted R2 on 2024: {best_score:.4f}")
    
    # Train final model with best params and show metrics
    print("\nValidating Best Model...")
    final_model = xgb.XGBRegressor(
        **best_params,
        n_jobs=1,
        random_state=42
    )
    final_model.fit(X_train, y_train, sample_weight=train_weights)
    final_preds = final_model.predict(X_test)
    
    # Standard Metrics
    mae = np.mean(np.abs(y_test - final_preds))
    r2 = r2_score(y_test, final_preds)
    
    # Weighted Metrics
    w_r2 = weighted_r2_score(y_test, final_preds, weights=test_weights)
    
    # Top 20 Error
    temp_df = test_data.copy()
    temp_df["Pred"] = final_preds
    temp_df["Error"] = np.abs(temp_df[TARGET] - temp_df["Pred"])
    top_20 = temp_df.sort_values(TARGET, ascending=False).head(20)
    top_20_mae = top_20["Error"].mean()
    
    print(f"Final Standard R2: {r2:.4f}")
    print(f"Final Weighted R2: {w_r2:.4f}")
    print(f"Final Global MAE: {mae:.4f}")
    print(f"Top 20 Performers MAE: {top_20_mae:.4f}")

if __name__ == "__main__":
    tune_xgboost()
