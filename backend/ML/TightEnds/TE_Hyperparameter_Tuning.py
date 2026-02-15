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

def tune_xgboost():
    print("Loading data...")
    data, _ = load_xgb_data(DATA_FILE)
    
    # Features (same as used in TE_Ensemble)
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
    # Train on everything before 2024
    train_data = data[data["Year"] < 2024].copy()
    test_data = data[data["Year"] == 2024].copy()
    
    X_train = train_data[predictors]
    y_train = train_data[TARGET]
    X_test = test_data[predictors]
    y_test = test_data[TARGET]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples (2024): {len(X_test)}")
    
    # Define Parameter Grid
    param_grid = {
        'n_estimators': [100, 500, 1000, 1500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=1,
        random_state=42,
        verbosity=0
    )
    
    # Custom scoring (R2 or RMSE). GridSearchCV defaults to score method of estimator (R2 for regressor)
    # But usually we might want to prioritize RMSE or just good old R2.
    
    print("\nStarting Grid Search (this may take a while)...")
    
    # We can use a randomized search if grid is too big, but let's try a smaller grid or RandomizedSearch
    from sklearn.model_selection import RandomizedSearchCV
    
    search = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=param_grid,
        n_iter=50,
        scoring='r2',
        cv=3, # 3-fold CV within the training set? 
              # Better to just optimize for the holdout 2024? 
              # Standard CV on time-series is tricky. 
              # If we just fit on Train and score on Test, we can't use GridSearchCV directly without a custom CV splitter.
              # Let's write a simple loop instead to be explicit and avoid leakage/bad splits.
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Actually, for time series/strict walk-forward, random CV is bad.
    # But here we are just tuning for the model that generalizes best to 2024 specifically? 
    # Or generally?
    # Let's try to find params that work best for 2024 given <2024 data.
    
    # Simple Loop Implementation for exact control
    best_score = -float('inf')
    best_params = None
    
    import itertools
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Randomly sample 50 combinations
    import random
    random.seed(42)
    masked_combinations = random.sample(combinations, min(50, len(combinations)))
    
    print(f"Evaluated {len(masked_combinations)} random configurations...")
    
    for i, params in enumerate(masked_combinations):
        model = xgb.XGBRegressor(
            **params,
            n_jobs=1,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        score = r2_score(y_test, preds)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New Best R2: {best_score:.4f} with params: {params}")
            
    print("\n==== BEST HYPERPARAMETERS ====")
    print(best_params)
    print(f"Best R2 on 2024: {best_score:.4f}")
    
    # Train final model with best params and show metrics
    print("\nValidating Best Model...")
    final_model = xgb.XGBRegressor(
        **best_params,
        n_jobs=1,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)
    
    mae = np.mean(np.abs(y_test - final_preds))
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    
    print(f"Final R2: {r2_score(y_test, final_preds):.4f}")
    print(f"Final MAE: {mae:.4f}")
    print(f"Final RMSE: {rmse:.4f}")

if __name__ == "__main__":
    tune_xgboost()
