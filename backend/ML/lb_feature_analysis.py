import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load data and show basic statistics"""
    print("="*80)
    print("FEATURE ANALYSIS FOR LB PREDICTION")
    print("="*80)
    
    df = pd.read_csv('lb_weighted_averages_by_team_pos_year.csv')
    df_lb = df[df['position'] == 'LB'].copy()
    
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(df_lb)}")
    print(f"  Years: {sorted(df_lb['Year'].unique())}")
    print(f"  Teams: {df_lb['Team'].nunique()}")
    print(f"  Columns: {len(df_lb.columns)}")
    
    return df_lb

def analyze_target(df_lb, target='weighted_avg_grades_defense'):
    """Analyze target variable"""
    print(f"\n{'='*80}")
    print(f"TARGET VARIABLE ANALYSIS: {target}")
    print(f"{'='*80}")
    
    target_data = df_lb[target].dropna()
    
    print(f"\nBasic Statistics:")
    print(f"  Count: {len(target_data)}")
    print(f"  Mean: {target_data.mean():.2f}")
    print(f"  Std: {target_data.std():.2f}")
    print(f"  Min: {target_data.min():.2f}")
    print(f"  25%: {target_data.quantile(0.25):.2f}")
    print(f"  Median: {target_data.median():.2f}")
    print(f"  75%: {target_data.quantile(0.75):.2f}")
    print(f"  Max: {target_data.max():.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(target_data, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {target}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(target_data)
    plt.ylabel(target)
    plt.title('Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: target_distribution.png")
    plt.close()

def categorize_features(df_lb):
    """Categorize all available features"""
    all_cols = df_lb.columns.tolist()
    
    categories = {
        'target': [],
        'identifiers': [],
        'current_year_weighted': [],
        'previous_year_weighted': [],
        'previous_year_summary': [],
        'current_year_context': [],
        'other': []
    }
    
    for col in all_cols:
        if col == 'weighted_avg_grades_defense' or col.startswith('weighted_avg_'):
            if not col.startswith('prev_'):
                categories['target'].append(col)
        elif col in ['Team', 'Year', 'position']:
            categories['identifiers'].append(col)
        elif col.startswith('prev_weighted_avg_'):
            categories['previous_year_weighted'].append(col)
        elif col.startswith('prev_'):
            categories['previous_year_summary'].append(col)
        elif col in ['Win_Percent', 'Net_EPA', 'sum_Cap_Space', 'sum_adjusted_value']:
            categories['current_year_context'].append(col)
        else:
            categories['other'].append(col)
    
    print(f"\n{'='*80}")
    print("FEATURE CATEGORIZATION")
    print(f"{'='*80}")
    
    for category, features in categories.items():
        if features:
            print(f"\n{category.upper()} ({len(features)} features):")
            for feat in features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")
    
    return categories

def analyze_correlations(df_lb, target='weighted_avg_grades_defense'):
    """Analyze correlations between all features and target"""
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Get all numeric columns except identifiers
    exclude_cols = ['Team', 'Year', 'position']
    numeric_cols = df_lb.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations
    correlations = []
    for col in numeric_cols:
        if col == target:
            continue
        
        # Get clean data
        clean_data = df_lb[[col, target]].dropna()
        if len(clean_data) < 10:  # Need at least 10 samples
            continue
        
        corr, pval = pearsonr(clean_data[col], clean_data[target])
        
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': pval,
            'n_samples': len(clean_data),
            'is_prev_year': col.startswith('prev_')
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    # Show top correlations
    print(f"\nTOP 20 FEATURES BY CORRELATION:")
    print("-" * 80)
    print(corr_df.head(20)[['feature', 'correlation', 'abs_correlation', 'p_value', 'is_prev_year']].to_string(index=False))
    
    # Statistics by category
    print(f"\n\nCORRELATION STATISTICS:")
    print("-" * 80)
    print(f"Total features analyzed: {len(corr_df)}")
    print(f"\nPrevious Year Features:")
    prev_year = corr_df[corr_df['is_prev_year'] == True]
    print(f"  Count: {len(prev_year)}")
    print(f"  Max |correlation|: {prev_year['abs_correlation'].max():.4f}")
    print(f"  Mean |correlation|: {prev_year['abs_correlation'].mean():.4f}")
    print(f"  With |corr| > 0.3: {(prev_year['abs_correlation'] > 0.3).sum()}")
    print(f"  With |corr| > 0.2: {(prev_year['abs_correlation'] > 0.2).sum()}")
    print(f"  With |corr| > 0.1: {(prev_year['abs_correlation'] > 0.1).sum()}")
    
    print(f"\nCurrent Year Features:")
    curr_year = corr_df[corr_df['is_prev_year'] == False]
    print(f"  Count: {len(curr_year)}")
    if len(curr_year) > 0:
        print(f"  Max |correlation|: {curr_year['abs_correlation'].max():.4f}")
        print(f"  Mean |correlation|: {curr_year['abs_correlation'].mean():.4f}")
    
    # Warnings
    print(f"\n\nWARNINGS:")
    print("-" * 80)
    if prev_year['abs_correlation'].max() < 0.3:
        print("⚠️  No previous year features have strong correlation (>0.3)")
    if prev_year['abs_correlation'].max() < 0.2:
        print("⚠️  No previous year features have moderate correlation (>0.2)")
        print("   This suggests predicting next year's performance from last year is very difficult!")
    
    # Check for high correlations with current year (data leakage risk)
    high_curr = curr_year[curr_year['abs_correlation'] > 0.5]
    if len(high_curr) > 0:
        print(f"\n⚠️  {len(high_curr)} current-year features have high correlation (>0.5)")
        print("   These should NOT be used for prediction (data leakage):")
        for feat in high_curr['feature'].tolist()[:5]:
            print(f"     - {feat}")
    
    # Save correlation plot
    plot_top_correlations(corr_df.head(20), target)
    
    return corr_df

def plot_top_correlations(top_corr_df, target):
    """Plot top correlations"""
    plt.figure(figsize=(12, 8))
    
    colors = ['red' if not prev else 'blue' for prev in top_corr_df['is_prev_year']]
    
    plt.barh(range(len(top_corr_df)), top_corr_df['correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_corr_df)), top_corr_df['feature'], fontsize=9)
    plt.xlabel('Correlation with ' + target)
    plt.title('Top 20 Features by Correlation\n(Blue = Previous Year, Red = Current Year)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: feature_correlations.png")
    plt.close()

def analyze_feature_pairs(df_lb, target='weighted_avg_grades_defense', top_n=5):
    """Analyze relationships between top features and target"""
    print(f"\n{'='*80}")
    print("TOP FEATURE VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Get top correlated previous year features
    exclude_cols = ['Team', 'Year', 'position']
    numeric_cols = df_lb.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    prev_cols = [col for col in numeric_cols if col.startswith('prev_')]
    
    correlations = []
    for col in prev_cols:
        clean_data = df_lb[[col, target]].dropna()
        if len(clean_data) >= 10:
            corr = np.corrcoef(clean_data[col], clean_data[target])[0, 1]
            correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in correlations[:top_n]]
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features[:6]):
        clean_data = df_lb[[feature, target]].dropna()
        
        axes[idx].scatter(clean_data[feature], clean_data[target], alpha=0.5, s=20)
        axes[idx].set_xlabel(feature.replace('prev_weighted_avg_', '').replace('_', ' ').title(), fontsize=8)
        axes[idx].set_ylabel(target.replace('weighted_avg_', '').replace('_', ' ').title(), fontsize=8)
        
        # Add trend line
        z = np.polyfit(clean_data[feature], clean_data[target], 1)
        p = np.poly1d(z)
        axes[idx].plot(clean_data[feature], p(clean_data[feature]), "r--", alpha=0.8, linewidth=2)
        
        # Add correlation
        corr = np.corrcoef(clean_data[feature], clean_data[target])[0, 1]
        axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[idx].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      verticalalignment='top', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top_feature_relationships.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: top_feature_relationships.png")
    plt.close()

def main():
    """Run complete feature analysis"""
    # Load data
    df_lb = load_and_explore_data()
    
    # Analyze target
    analyze_target(df_lb)
    
    # Categorize features
    categories = categorize_features(df_lb)
    
    # Analyze correlations
    corr_df = analyze_correlations(df_lb)
    
    # Visualize top features
    analyze_feature_pairs(df_lb)
    
    # Save correlation results
    corr_df.to_csv('feature_correlations.csv', index=False)
    print(f"\n  Saved: feature_correlations.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - target_distribution.png")
    print("  - feature_correlations.png")
    print("  - feature_correlations.csv")
    print("  - top_feature_relationships.png")
    
    print("\n\nRECOMMENDATIONS:")
    print("-" * 80)
    
    # Get previous year correlations
    prev_corr = corr_df[corr_df['is_prev_year'] == True]
    max_corr = prev_corr['abs_correlation'].max()
    
    if max_corr < 0.15:
        print("❌ VERY WEAK PREDICTIVE POWER")
        print("   Max correlation is < 0.15. Consider:")
        print("   1. Adding more features (opponent strength, coaching, injuries)")
        print("   2. Using a different target (classification, simpler metrics)")
        print("   3. Accepting that year-to-year LB performance is highly random")
    elif max_corr < 0.25:
        print("⚠️  WEAK PREDICTIVE POWER")
        print("   Max correlation is < 0.25. You may get limited accuracy.")
        print("   Consider enriching features or simplifying the problem.")
    elif max_corr < 0.4:
        print("✓ MODERATE PREDICTIVE POWER")
        print("   Max correlation is < 0.4. Model should show some predictive ability.")
        print("   Focus on the highest correlated features.")
    else:
        print("✓✓ GOOD PREDICTIVE POWER")
        print("   Max correlation is >= 0.4. Model should perform reasonably well.")
    
    print("\nUse only features from 'previous_year' categories to avoid data leakage!")

if __name__ == "__main__":
    main()