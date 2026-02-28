import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(min_correlation=0.15):
    """Load data and prepare features"""
    print("="*80)
    print("SIMPLE BASELINE MODELS FOR LB PREDICTION")
    print("="*80)
    
    df = pd.read_csv('lb_weighted_averages_by_team_pos_year.csv')
    df_lb = df[df['position'] == 'LB'].copy()
    
    print(f"\nLoaded {len(df_lb)} LB records")
    
    # Target
    target = 'weighted_avg_grades_defense'
    
    # Get previous year features
    prev_features = [col for col in df_lb.columns if col.startswith('prev_weighted_avg_')]
    prev_features.extend(['prev_total_snap_counts_defense', 'prev_total_players', 
                         'prev_sum_Cap_Space', 'prev_sum_adjusted_value'])
    
    # Clean data
    required_cols = prev_features + [target, 'Year']
    df_clean = df_lb[required_cols].dropna()
    
    print(f"Clean records: {len(df_clean)}")
    
    # Filter by correlation
    X_all = df_clean[prev_features].values
    y = df_clean[target].values
    
    print(f"\nFiltering features by correlation (min = {min_correlation})...")
    correlations = []
    for i, feat in enumerate(prev_features):
        corr = np.corrcoef(X_all[:, i], y)[0, 1]
        if not np.isnan(corr) and abs(corr) >= min_correlation:
            correlations.append((feat, corr, abs(corr)))
    
    correlations.sort(key=lambda x: x[2], reverse=True)
    selected_features = [f[0] for f in correlations]
    
    print(f"Selected {len(selected_features)} features with |corr| >= {min_correlation}")
    print("\nTop 10 features:")
    for feat, corr, abs_corr in correlations[:10]:
        print(f"  {feat}: {corr:.3f}")
    
    X = df_clean[selected_features].values
    
    return X, y, selected_features, df_clean

def time_based_split(X, y, df_clean, test_size=0.2, val_size=0.2):
    """Split by year"""
    years = sorted(df_clean['Year'].unique())
    n_years = len(years)
    
    n_test_years = max(1, int(n_years * test_size))
    n_val_years = max(1, int(n_years * val_size))
    
    test_years = years[-n_test_years:]
    val_years = years[-(n_test_years + n_val_years):-n_test_years]
    train_years = years[:-(n_test_years + n_val_years)]
    
    print(f"\nTime-based split:")
    print(f"  Train: {train_years}")
    print(f"  Val: {val_years}")
    print(f"  Test: {test_years}")
    
    train_mask = df_clean['Year'].isin(train_years).values
    val_mask = df_clean['Year'].isin(val_years).values
    test_mask = df_clean['Year'].isin(test_years).values
    
    return (X[train_mask], X[val_mask], X[test_mask],
            y[train_mask], y[val_mask], y[test_mask])

def evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Try multiple simple models"""
    print("\n" + "="*80)
    print("EVALUATING MULTIPLE MODELS")
    print("="*80)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, 
                                                        learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Metrics
        results[name] = {
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'model': model,
            'test_pred': test_pred
        }
        
        print(f"\n{name}:")
        print(f"  Train R²: {results[name]['train_r2']:.4f}, MAE: {results[name]['train_mae']:.2f}")
        print(f"  Val R²:   {results[name]['val_r2']:.4f}, MAE: {results[name]['val_mae']:.2f}")
        print(f"  Test R²:  {results[name]['test_r2']:.4f}, MAE: {results[name]['test_mae']:.2f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_model_name]
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"  Test R²: {best_result['test_r2']:.4f}")
    print(f"  Test MAE: {best_result['test_mae']:.2f}")
    print("="*80)
    
    return results, best_model_name

def plot_results(y_test, results, best_model_name):
    """Plot predictions for best model"""
    best_pred = results[best_model_name]['test_pred']
    best_r2 = results[best_model_name]['test_r2']
    best_mae = results[best_model_name]['test_mae']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.scatter(y_test, best_pred, alpha=0.6)
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual grades_defense')
    ax1.set_ylabel('Predicted grades_defense')
    ax1.set_title(f'{best_model_name} - Test Set')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    textstr = f'R² = {best_r2:.4f}\nMAE = {best_mae:.2f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residuals
    residuals = y_test - best_pred
    ax2.scatter(best_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted grades_defense')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{best_model_name} - Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_models_results.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: baseline_models_results.png")
    plt.show()

def plot_model_comparison(results):
    """Compare all models"""
    model_names = list(results.keys())
    test_r2s = [results[name]['test_r2'] for name in model_names]
    test_maes = [results[name]['test_mae'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    colors = ['green' if r2 > 0 else 'red' for r2 in test_r2s]
    bars1 = ax1.barh(model_names, test_r2s, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Test R² Score')
    ax1.set_title('Model Comparison - R²')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{test_r2s[i]:.3f}', 
                ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # MAE comparison
    ax2.barh(model_names, test_maes, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Test MAE')
    ax2.set_title('Model Comparison - MAE (lower is better)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: model_comparison.png")
    plt.show()

def main():
    # Load and prepare data
    X, y, features, df_clean = load_and_prepare_data(min_correlation=0.15)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        X, y, df_clean, test_size=0.2, val_size=0.2
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Evaluate models
    results, best_model_name = evaluate_models(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Plot results
    plot_results(y_test, results, best_model_name)
    plot_model_comparison(results)
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nWith max correlation of ~0.37, we achieved:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"  Test MAE: {results[best_model_name]['test_mae']:.2f}")
    
    if results[best_model_name]['test_r2'] < 0:
        print("\n⚠️  Even the best model has negative R².")
        print("   This confirms that predicting next year's LB performance")
        print("   from last year's stats is very difficult with this data.")
        print("\n   Recommendations:")
        print("   1. Add more features (opponent quality, coaching changes, draft picks)")
        print("   2. Try classification instead (Good/Bad LB units)")
        print("   3. Predict something more stable (snap counts, not grades)")
    elif results[best_model_name]['test_r2'] < 0.15:
        print("\n⚠️  Best R² is < 0.15 - very weak predictive power")
        print("   The model explains less than 15% of variance.")
    else:
        print("\n✓ Model shows some predictive ability!")
        print(f"  Explains {results[best_model_name]['test_r2']*100:.1f}% of variance")
    
    print("\n   Note: LSTM is likely overkill for this problem.")
    print("   Simple linear models are probably best given the weak correlations.")

if __name__ == "__main__":
    main()