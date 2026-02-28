import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(min_correlation=0.15):
    """Load data with feature selection"""
    print("="*80)
    print("TRANSFORMER VS OTHER MODELS - TESTING NON-LINEAR PATTERNS")
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
    selected_features = [f[0] for f in correlations[:12]]  # Use top 12 features
    
    print(f"\nSelected top 12 features:")
    for feat, corr, _ in correlations[:12]:
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
    print(f"  Train: {train_years} ({len(train_years)} years)")
    print(f"  Val: {val_years} ({len(val_years)} years)")
    print(f"  Test: {test_years} ({len(test_years)} years)")
    
    train_mask = df_clean['Year'].isin(train_years).values
    val_mask = df_clean['Year'].isin(val_years).values
    test_mask = df_clean['Year'].isin(test_years).values
    
    return (X[train_mask], X[val_mask], X[test_mask],
            y[train_mask], y[val_mask], y[test_mask])

def build_transformer_model(input_dim, d_model=32, num_heads=4, ff_dim=64, dropout=0.4):
    """
    Build a lightweight transformer encoder
    Designed for small tabular data (not sequences)
    """
    print("\n" + "="*80)
    print("TRANSFORMER MODEL")
    print("="*80)
    print(f"  Embedding dim: {d_model}, Heads: {num_heads}, FF dim: {ff_dim}")
    
    inputs = layers.Input(shape=(input_dim,))
    
    # Project to embedding space
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout)(x)
    
    # Reshape for attention (batch, 1, d_model)
    x_reshaped = layers.Reshape((1, d_model))(x)
    
    # Multi-head self-attention block
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x_reshaped, x_reshaped)
    
    # Add & Norm
    x_reshaped = layers.Add()([x_reshaped, attention_output])
    x_reshaped = layers.LayerNormalization(epsilon=1e-6)(x_reshaped)
    
    # Feed-forward network
    ff = layers.Dense(ff_dim, activation="relu")(x_reshaped)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    
    # Add & Norm
    x_reshaped = layers.Add()([x_reshaped, ff])
    x_reshaped = layers.LayerNormalization(epsilon=1e-6)(x_reshaped)
    
    # Flatten and output
    x_flat = layers.Flatten()(x_reshaped)
    x_flat = layers.Dense(32, activation="relu")(x_flat)
    x_flat = layers.Dropout(dropout)(x_flat)
    x_flat = layers.Dense(16, activation="relu")(x_flat)
    x_flat = layers.Dropout(dropout)(x_flat)
    outputs = layers.Dense(1)(x_flat)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_mlp(input_dim):
    """Simple MLP for comparison"""
    print("\n" + "="*80)
    print("MLP MODEL (for comparison)")
    print("="*80)
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=300):
    """Train model with early stopping"""
    print(f"\nTraining {model_name}...")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True,
        verbose=0
    )
    
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-7,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,  # Small batch for small data
        callbacks=[early_stopping, lr_scheduler],
        verbose=0
    )
    
    print(f"  Stopped at epoch {len(history.history['loss'])}")
    print(f"  Best val loss: {min(history.history['val_loss']):.4f}")
    
    return history

def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Compare Ridge, MLP, and Transformer"""
    print("\n" + "="*80)
    print("EVALUATING ALL MODELS")
    print("="*80)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Ridge Regression (Linear Baseline)
    print("\n1. RIDGE REGRESSION (Linear)")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    for split, X, y_true in [('train', X_train_scaled, y_train),
                              ('val', X_val_scaled, y_val),
                              ('test', X_test_scaled, y_test)]:
        pred = ridge.predict(X)
        results[f'Ridge_{split}'] = {
            'r2': r2_score(y_true, pred),
            'mae': mean_absolute_error(y_true, pred),
            'pred': pred
        }
    
    print(f"  Test R²: {results['Ridge_test']['r2']:.4f}, MAE: {results['Ridge_test']['mae']:.2f}")
    
    # 2. MLP (Non-linear neural network)
    print("\n2. MLP")
    mlp = build_mlp(X_train_scaled.shape[1])
    mlp_history = train_model(mlp, X_train_scaled, y_train, X_val_scaled, y_val, 
                               'MLP', epochs=300)
    
    for split, X, y_true in [('train', X_train_scaled, y_train),
                              ('val', X_val_scaled, y_val),
                              ('test', X_test_scaled, y_test)]:
        pred = mlp.predict(X, verbose=0).flatten()
        results[f'MLP_{split}'] = {
            'r2': r2_score(y_true, pred),
            'mae': mean_absolute_error(y_true, pred),
            'pred': pred
        }
    
    print(f"  Test R²: {results['MLP_test']['r2']:.4f}, MAE: {results['MLP_test']['mae']:.2f}")
    
    # 3. Transformer (Non-linear with attention)
    print("\n3. TRANSFORMER")
    transformer = build_transformer_model(X_train_scaled.shape[1])
    transformer_history = train_model(transformer, X_train_scaled, y_train, 
                                     X_val_scaled, y_val, 'Transformer', epochs=300)
    
    for split, X, y_true in [('train', X_train_scaled, y_train),
                              ('val', X_val_scaled, y_val),
                              ('test', X_test_scaled, y_test)]:
        pred = transformer.predict(X, verbose=0).flatten()
        results[f'Transformer_{split}'] = {
            'r2': r2_score(y_true, pred),
            'mae': mean_absolute_error(y_true, pred),
            'pred': pred
        }
    
    print(f"  Test R²: {results['Transformer_test']['r2']:.4f}, MAE: {results['Transformer_test']['mae']:.2f}")
    
    return results, mlp_history, transformer_history, y_test

def plot_results(results, y_test, mlp_history, transformer_history):
    """Visualize results"""
    
    # Model comparison
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Test R² comparison
    ax1 = plt.subplot(2, 3, 1)
    models = ['Ridge', 'MLP', 'Transformer']
    r2_scores = [results[f'{m}_test']['r2'] for m in models]
    colors = ['green' if r2 > 0 else 'red' for r2 in r2_scores]
    bars = ax1.bar(models, r2_scores, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Test R² Score')
    ax1.set_title('Model Comparison - R²')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, r2 in zip(bars, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 2. MAE comparison
    ax2 = plt.subplot(2, 3, 2)
    mae_scores = [results[f'{m}_test']['mae'] for m in models]
    ax2.bar(models, mae_scores, alpha=0.7, color='steelblue')
    ax2.set_ylabel('Test MAE')
    ax2.set_title('Model Comparison - MAE (lower better)')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, mae in enumerate(mae_scores):
        ax2.text(i, mae, f'{mae:.2f}', ha='center', va='bottom')
    
    # 3. Training curves - MLP
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(mlp_history.history['loss'], label='Train Loss', alpha=0.7)
    ax3.plot(mlp_history.history['val_loss'], label='Val Loss', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('MLP Training Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Training curves - Transformer
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(transformer_history.history['loss'], label='Train Loss', alpha=0.7)
    ax4.plot(transformer_history.history['val_loss'], label='Val Loss', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (MSE)')
    ax4.set_title('Transformer Training Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Predictions scatter - best model
    best_model = max(models, key=lambda m: results[f'{m}_test']['r2'])
    ax5 = plt.subplot(2, 3, 5)
    pred = results[f'{best_model}_test']['pred']
    r2 = results[f'{best_model}_test']['r2']
    mae = results[f'{best_model}_test']['mae']
    
    ax5.scatter(y_test, pred, alpha=0.6)
    min_val = min(y_test.min(), pred.min())
    max_val = max(y_test.max(), pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax5.set_xlabel('Actual')
    ax5.set_ylabel('Predicted')
    ax5.set_title(f'{best_model} - Test Set\nR²={r2:.3f}, MAE={mae:.2f}')
    ax5.grid(True, alpha=0.3)
    
    # 6. Residuals - best model
    ax6 = plt.subplot(2, 3, 6)
    residuals = y_test - pred
    ax6.scatter(pred, residuals, alpha=0.6)
    ax6.axhline(y=0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Residuals')
    ax6.set_title(f'{best_model} - Residuals')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  Saved: transformer_comparison.png")
    plt.show()

def main():
    # Load data
    X, y, features, df_clean = load_and_prepare_data(min_correlation=0.15)
    
    print(f"\nData shape: {X.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        X, y, df_clean, test_size=0.2, val_size=0.2
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Evaluate all models
    results, mlp_hist, trans_hist, y_test = evaluate_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Plot results
    plot_results(results, y_test, mlp_hist, trans_hist)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print("\nTest Set Performance:")
    for model in ['Ridge', 'MLP', 'Transformer']:
        r2 = results[f'{model}_test']['r2']
        mae = results[f'{model}_test']['mae']
        print(f"\n{model}:")
        print(f"  R²:  {r2:>7.4f}")
        print(f"  MAE: {mae:>7.2f}")
    
    # Determine winner
    models = ['Ridge', 'MLP', 'Transformer']
    best = max(models, key=lambda m: results[f'{m}_test']['r2'])
    
    print(f"\n{'='*80}")
    print(f"WINNER: {best}")
    print(f"  Test R²: {results[f'{best}_test']['r2']:.4f}")
    print(f"{'='*80}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    
    ridge_r2 = results['Ridge_test']['r2']
    transformer_r2 = results['Transformer_test']['r2']
    
    if transformer_r2 > ridge_r2 + 0.05:
        print("✓ Transformer found non-linear patterns that linear model missed!")
        print("  → Complex feature interactions matter for this problem")
    elif transformer_r2 > ridge_r2:
        print("≈ Transformer slightly better, but marginal improvement")
        print("  → Some non-linearity exists but effect is small")
    else:
        print("✗ Transformer did not beat linear baseline")
        print("  → Relationship is mostly linear, or too little data to learn non-linear patterns")
        print("  → The 0.37 correlation captures most available signal")
    
    if all(results[f'{m}_test']['r2'] < 0 for m in models):
        print("\n⚠️  ALL MODELS have negative R²")
        print("   → Even with non-linear models, features don't predict target well")
        print("   → Need better features (personnel changes, injuries, coaching, etc.)")

if __name__ == "__main__":
    main()