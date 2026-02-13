"""
Enrich the 2025 schemes file with Sharp Football and SumerSports data.

What this does (for the 2025 NFL season):
- Take `2025_schemes.csv` (PBP-derived features)
- Overwrite its scheme columns with Sharp Football 2025 numbers:
    motion_rate, play_action_rate, air_yards_per_att,
    shotgun_rate, no_huddle_rate, under_center_rate
- Then run the existing Sumer personnel merge for 2025
  and fold the result back into `2025_schemes.csv`,
  so that file now contains:
    - All PBP metrics
    - Sharp motion / play-action / shotgun / no-huddle / air yards
    - Personnel usage rates (10, 11, 12, 13, 20, 21, 22, 01, 31)

Usage (from backend/ML):

    python -m scheme.enrich_2025_schemes
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR
    from .merge_sumer_personnel import merge_sumer_personnel_for_year
except ImportError:  # pragma: no cover
    import os

    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    from merge_sumer_personnel import merge_sumer_personnel_for_year  # type: ignore


SHARP_FILENAME = "sharp_scheme_2026.csv"  # This actually represents the 2025 NFL season.


def merge_sharp_into_2025() -> Path:
    """
    Merge Sharp Football 2025 scheme stats into `2025_schemes.csv`.

    Returns the path to the updated `2025_schemes.csv`.
    """
    base_dir = Path(SCHEME_DATA_DIR)
    schemes_path = base_dir / "2025_schemes.csv"
    sharp_path = base_dir / SHARP_FILENAME

    if not schemes_path.exists():
        raise FileNotFoundError(f"Base schemes file not found: {schemes_path}")
    if not sharp_path.exists():
        raise FileNotFoundError(
            f"Sharp schemes file not found: {sharp_path}\n"
            "This should be the Sharp Football table for the 2025 NFL season."
        )

    schemes = pd.read_csv(schemes_path)
    sharp = pd.read_csv(sharp_path)

    # Columns to take from Sharp and splice into schemes
    sharp_cols = [
        "motion_rate",
        "play_action_rate",
        "air_yards_per_att",
        "shotgun_rate",
        "no_huddle_rate",
        "under_center_rate",
    ]

    # Ensure the required join key and feature columns exist in Sharp
    required_sharp_cols = ["team_abbr"] + sharp_cols
    missing = [c for c in required_sharp_cols if c not in sharp.columns]
    if missing:
        raise ValueError(
            f"Sharp CSV is missing expected columns {missing}. "
            f"Columns present: {list(sharp.columns)}"
        )

    sharp_trim = sharp[required_sharp_cols].copy()

    # Normalize team_abbr in schemes to match Sharp (e.g. LA -> LAR for Rams)
    schemes_normalized = schemes.copy()
    schemes_normalized["team_abbr"] = schemes_normalized["team_abbr"].replace(
        {
            "LA": "LAR",  # Make sure Rams join correctly
        }
    )

    # Drop any existing versions of these columns from schemes so Sharp values win
    schemes_wo_sharp = schemes_normalized.drop(columns=sharp_cols, errors="ignore")

    merged = schemes_wo_sharp.merge(sharp_trim, on="team_abbr", how="left")

    # Save back to the same path, so downstream code naturally picks up enriched data
    merged.to_csv(schemes_path, index=False)
    return schemes_path


def enrich_2025_with_sharp_and_personnel() -> Path:
    """
    Full enrichment pipeline for 2025:
    - Merge Sharp stats into `2025_schemes.csv`
    - Run Sumer personnel merge for 2025
    - Overwrite `2025_schemes.csv` with the personnel-enriched result

    Returns the path to the final `2025_schemes.csv`.
    """
    base_dir = Path(SCHEME_DATA_DIR)

    # 1) Sharp -> 2025_schemes.csv
    schemes_path = merge_sharp_into_2025()

    # 2) Sumer personnel merge writes 2025_schemes_with_personnel.csv
    with_personnel_path = merge_sumer_personnel_for_year(2025)

    # 3) Overwrite 2025_schemes.csv with the with-personnel version
    enriched = pd.read_csv(with_personnel_path)
    enriched.to_csv(schemes_path, index=False)

    return schemes_path


if __name__ == "__main__":
    final_path = enrich_2025_with_sharp_and_personnel()
    print(f"Enriched 2025 schemes with Sharp + Sumer personnel into {final_path}")

