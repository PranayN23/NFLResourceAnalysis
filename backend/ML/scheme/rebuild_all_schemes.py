"""
Rebuild all yearly scheme CSV files with read option rate included.

This script:
1. Rebuilds PBP scheme data (includes read_option_rate)
2. Splits by year
3. Merges Sharp data for 2025
4. Merges Sumer personnel data for 2022-2025
5. Re-clusters all years using all available metrics
6. Normalizes column order (scheme_cluster at end)

Usage (from repo root):
    python -m backend.ML.scheme.rebuild_all_schemes
"""
from __future__ import annotations

from pathlib import Path

try:
    from .scheme_from_pbp import build_and_save_pbp_scheme
    from .split_by_year import split_scheme_by_year
    from .enrich_2025_schemes import enrich_2025_with_sharp_and_personnel
    from .merge_sumer_personnel import merge_sumer_personnel
    from .fix_and_validate_schemes import fix_and_cluster_all_years
    from .normalize_schemes import normalize_schemes
    from .config import SCHEME_DATA_DIR
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scheme.scheme_from_pbp import build_and_save_pbp_scheme
    from scheme.split_by_year import split_scheme_by_year
    from scheme.enrich_2025_schemes import enrich_2025_with_sharp_and_personnel
    from scheme.merge_sumer_personnel import merge_sumer_personnel
    from scheme.fix_and_validate_schemes import fix_and_cluster_all_years
    from scheme.normalize_schemes import normalize_schemes
    import os
    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def rebuild_all_schemes() -> None:
    """
    Complete rebuild of all yearly scheme CSV files with read option rate.
    """
    print("[rebuild_all_schemes] Step 1: Rebuilding PBP scheme data...")
    pbp_path = build_and_save_pbp_scheme()
    print(f"[rebuild_all_schemes] Saved PBP data to {pbp_path}")
    
    print("\n[rebuild_all_schemes] Step 2: Splitting by year...")
    year_files = split_scheme_by_year(pbp_path)
    print(f"[rebuild_all_schemes] Split into {len(year_files)} yearly files")
    
    print("\n[rebuild_all_schemes] Step 3: Enriching 2025 with Sharp + Sumer data...")
    try:
        enrich_2025_with_sharp_and_personnel()
        print("[rebuild_all_schemes] 2025 enrichment complete")
    except FileNotFoundError as e:
        print(f"[rebuild_all_schemes] Warning: Could not enrich 2025: {e}")
        print("[rebuild_all_schemes] Continuing without 2025 enrichment...")
    
    print("\n[rebuild_all_schemes] Step 4: Merging Sumer personnel data for 2022-2024...")
    try:
        merge_sumer_personnel(years=[2022, 2023, 2024])
        print("[rebuild_all_schemes] Personnel merge complete")
    except Exception as e:
        print(f"[rebuild_all_schemes] Warning: Could not merge personnel: {e}")
        print("[rebuild_all_schemes] Continuing without personnel data...")
    
    print("\n[rebuild_all_schemes] Step 5: Re-clustering all years with all available metrics...")
    fix_and_cluster_all_years(n_clusters=4)
    print("[rebuild_all_schemes] Clustering complete")
    
    print("\n[rebuild_all_schemes] Step 6: Normalizing column order...")
    normalize_schemes()
    print("[rebuild_all_schemes] Normalization complete")
    
    print("\n[rebuild_all_schemes] ✅ All steps complete! Yearly CSV files updated with read_option_rate.")


if __name__ == "__main__":
    rebuild_all_schemes()
