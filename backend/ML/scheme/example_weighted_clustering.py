"""
Example: How to use feature weights in K-means clustering.

This shows how to weight certain features more heavily than others when clustering.
Features with higher weights have more influence in determining cluster assignments.

Usage (from repo root):
    python -m backend.ML.scheme.example_weighted_clustering
"""
from __future__ import annotations

from backend.ML.scheme.fix_and_validate_schemes import fix_and_cluster_all_years

# Example 1: Weight motion_rate and play_action_rate 2x more than other features
# This means teams will be grouped more by motion/play-action similarity
WEIGHTS_EXAMPLE_1 = {
    "motion_rate": 2.0,  # 2x weight
    "play_action_rate": 2.0,  # 2x weight
    # All other features default to 1.0 (equal weight)
}

# Example 2: Weight personnel packages more heavily
# This groups teams more by personnel usage patterns
WEIGHTS_EXAMPLE_2 = {
    "personnel_11_rate": 1.5,
    "personnel_12_rate": 1.5,
    "personnel_13_rate": 1.5,
    "personnel_21_rate": 1.5,
    "motion_rate": 1.2,
    "play_action_rate": 1.2,
    # All other features default to 1.0
}

# Example 3: Weight formation-specific rates more
# This emphasizes under-center vs shotgun differences
WEIGHTS_EXAMPLE_3 = {
    "under_center_rate": 2.0,
    "shotgun_rate": 2.0,
    "under_center_play_action_rate": 1.5,
    "shotgun_play_action_rate": 1.5,
    # All other features default to 1.0
}

if __name__ == "__main__":
    print("Example: Clustering with weighted features")
    print("=" * 60)
    print("\nThis example shows how to weight motion_rate and play_action_rate")
    print("2x more heavily than other features.\n")
    
    # Uncomment to run with custom weights:
    # fix_and_cluster_all_years(n_clusters=4, feature_weights=WEIGHTS_EXAMPLE_1)
    
    print("To use custom weights, modify this script and uncomment the line above.")
    print("\nOr call directly:")
    print("  from backend.ML.scheme.fix_and_validate_schemes import fix_and_cluster_all_years")
    print("  weights = {'motion_rate': 2.0, 'play_action_rate': 2.0}")
    print("  fix_and_cluster_all_years(feature_weights=weights)")
