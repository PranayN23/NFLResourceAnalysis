"""
Normalize yearly scheme CSVs so that:

- 2022–2024 `*_schemes.csv` include Sumer personnel columns
  (copied from `*_schemes_with_personnel.csv`).
- 2025 `2025_schemes.csv` keeps the Sharp-enriched + personnel data.
- In all years, `scheme_cluster` appears **after all personnel columns**
  (or at the very end if no personnel are present).

Usage (from repo root):

    python -m backend.ML.scheme.normalize_schemes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from .config import SCHEME_DATA_DIR
except ImportError:  # pragma: no cover
    import os

    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _list_year_scheme_files() -> Dict[int, Path]:
    """Return mapping of year -> {year}_schemes.csv path."""
    base = Path(SCHEME_DATA_DIR)
    out: Dict[int, Path] = {}
    for p in base.glob("*_schemes.csv"):
        # Skip any non-year prefixes (e.g. sharp_*)
        stem = p.stem
        if stem.startswith("sharp_"):
            continue
        try:
            year = int(stem.split("_")[0])
        except ValueError:
            continue
        out[year] = p
    return out


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so that all personnel_* columns come before scheme_cluster,
    and scheme_cluster (if present) is the last column.
    """
    personnel_cols = [c for c in df.columns if c.startswith("personnel_")]
    has_cluster = "scheme_cluster" in df.columns

    core_cols = [c for c in df.columns if c not in personnel_cols and c != "scheme_cluster"]

    new_order = core_cols + personnel_cols
    if has_cluster:
        new_order.append("scheme_cluster")

    # Only keep columns that actually exist (defensive)
    new_order = [c for c in new_order if c in df.columns]
    return df.loc[:, new_order]


def normalize_schemes() -> None:
    """
    Normalize all yearly scheme CSVs in SCHEME_DATA_DIR.
    """
    base = Path(SCHEME_DATA_DIR)
    year_files = _list_year_scheme_files()

    for year, schemes_path in sorted(year_files.items()):
        with_personnel_path = base / f"{year}_schemes_with_personnel.csv"

        # Decide which source to load:
        # - 2022–2024: prefer the *_schemes_with_personnel.csv version
        # - 2025: keep the already-enriched 2025_schemes.csv (has Sharp + personnel)
        # - other years: just use the base {year}_schemes.csv
        if year in (2022, 2023, 2024) and with_personnel_path.exists():
            df = pd.read_csv(with_personnel_path)
            source = with_personnel_path
        else:
            df = pd.read_csv(schemes_path)
            source = schemes_path

        df_norm = _reorder_columns(df)
        df_norm.to_csv(schemes_path, index=False)
        print(f"[normalize_schemes] Year {year}: wrote normalized {schemes_path} (source={source.name})")


if __name__ == "__main__":
    normalize_schemes()

